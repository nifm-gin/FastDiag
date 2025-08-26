import csv
import pandas as pd
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer, confusion_matrix, roc_auc_score, f1_score, brier_score_loss

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from get_outcomes import *
from get_data import *
from utility.utility import align_X_y_and_clean
from utility.utility import false_neg_scorer, false_pos_scorer

import matplotlib.pyplot as plt

DATA_DIRECTORY = "data/"

OUTCOMES = ["mortality_6m", "mortality_30d", "mortality_7d",
            "gose_6m", "gose_30d", "TIER", "TIL"]

FEATURES_old = ["traumatrix", "segmentation", "traumatrix_and_segmentation", # old naming convention for features
            "all_prehospital", "all_prehospital_and_segmentation", "all_DCA",
            "all_DCA_and_segmentation"]

FEATURES = ["PREHOSP", "CT-TIQUA", "MULTI", 
            "PREHOSP-X", "MULTI-PRE", "RESUS-X", "MULTI-RESUS"]

def compute_shap(outcome, feature_set, hyperparams):
    # get outcome ground truth
    if outcome == "mortality_6m":
        y = get_mortality_6m()
    elif outcome == "mortality_30d":
        y = get_mortality_30d()
    elif outcome == "mortality_7d": 
        y = get_mortality_7d()
    elif outcome == "gose_6m":
        y = get_gose_6m()
    elif outcome == "gose_30d":
        y = get_gose_30d()
    elif outcome == "TIER":
        y = get_tier()
    elif outcome == "TIL":
        y = get_tier()

    # get input features
    if feature_set == "PREHOSP":
        X = get_traumatrix(with_name=True)

    elif feature_set == "CT-TIQUA":
        X = get_segmentation(with_name=True)

    elif feature_set == "MULTI":
        X = get_traumatrix_and_segmentation(with_name=True)

    elif feature_set == "PREHOSP-X":
        X = get_all_prehospital(with_name=True)

    elif feature_set == "MULTI-PRE":
        X = get_all_prehospital_and_segmentation(with_name=True)

    elif feature_set == "RESUS-X":
        X = get_all_DCA(with_name=True)

    elif feature_set == "MULTI-RESUS":
        X = get_all_DCA_and_segmentation(with_name=True)

    # align, clean and imputation 
    X, y = align_X_y_and_clean(X, y)

    # model pipeline (minority class oversampling + majority class undersampling + model)
    pipeline_smote_under = Pipeline(steps=[('over', SMOTE(sampling_strategy=hyperparams["over__sampling_strategy"], k_neighbors=hyperparams["over__k_neighbors"])), 
                                           ('under', RandomUnderSampler(sampling_strategy=hyperparams["under__sampling_strategy"])), 
                                           ('model', HistGradientBoostingClassifier(learning_rate=hyperparams["model__learning_rate"]))])

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    ftwo_scorer = make_scorer(fbeta_score, beta=2)

    nested_scores_smote_undersampling = cross_validate(pipeline_smote_under, X, y, 
                                                    scoring={'F2':ftwo_scorer, 'ROC_AUC':'roc_auc', 'Recall':'recall_macro', 'F1':'f1', 'Brier':"neg_brier_score", 'False_neg_scorer':false_neg_scorer, 'False_pos_scorer':false_pos_scorer}, 
                                                    cv=outer_cv, n_jobs=-1, return_estimator=True, return_indices=True)


    for type in ["False Negative", "False Positive", "True Negative", "True Positive"]:
        
        for fold in range(len(nested_scores_smote_undersampling["indices"]["train"])):
            y_true_train = np.asarray(y)[nested_scores_smote_undersampling["indices"]["train"][fold]]
            y_true_test = np.asarray(y)[nested_scores_smote_undersampling["indices"]["test"][fold]]

            fitted = nested_scores_smote_undersampling["estimator"][0].fit(X.loc[nested_scores_smote_undersampling["indices"]["train"][fold]], y_true_train) #-#=#

            pred_test = fitted.predict(X.loc[nested_scores_smote_undersampling["indices"]["test"][fold]])
            

            index = -1
            for i in range(len(y_true_test)):

                if type=="False Negative" and y_true_test[i] == 0 and pred_test[i] == 0:
                    index = i
                elif type=="False Positive" and y_true_test[i] == 1 and pred_test[i] == 1:
                    index = i
                elif type=="True Negative" and y_true_test[i] == 1 and pred_test[i] == 0:
                    index = i
                elif type=="True Positive" and y_true_test[i] == 0 and pred_test[i] == 1:
                    index = i
            
            if index != -1: # we found a TN or TP or whatever we are looking for inside (FN, FP, TN, TP)
                explainer = shap.TreeExplainer(fitted['model']) 
                shap_values = explainer(X.loc[nested_scores_smote_undersampling["indices"]["test"][fold]])
                #plt.figure(figsize=(20, 5))
                plt.title(f"{type} SHAP values for {outcome} outcome and {feature_set} data.")
                shap.plots.waterfall(shap_values[index], show=False)
                plt.savefig(f"SHAP_plots/individual/shap_{outcome}_{feature_set}_{type}.png", dpi=300, bbox_inches='tight')
                plt.clf()
                # so we can exit this loop and go for the next type inside (FN, FP, TN, TP) 
                break


if __name__ == "__main__":
    for outcome in OUTCOMES:
        for i, feature in enumerate(FEATURES):
            df = pd.read_csv(f"results_summary/results_{FEATURES_old[i]}_{outcome}.csv")
            hyperparams = {"model__learning_rate": df["model__learning_rate"].iloc[0],
                           "under__sampling_strategy": df["under__sampling_strategy"].iloc[0],
                           "over__sampling_strategy": df["over__sampling_strategy"].iloc[0],
                           "over__k_neighbors": df["over__k_neighbors"].iloc[0],}
            
            compute_shap(outcome, feature, hyperparams)