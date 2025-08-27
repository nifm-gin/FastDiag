import csv
import os
import pandas as pd
import shap

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, make_scorer, confusion_matrix, roc_auc_score, f1_score, brier_score_loss

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from collections import Counter

from get_outcomes import *
from get_data import *
from utility.utility import align_X_y_and_clean, translate_column_names
from utility.utility import false_neg_scorer, false_pos_scorer

import matplotlib.pyplot as plt

DATA_DIRECTORY = "data/"

#OUTCOMES = ["mortality_6m", "mortality_30d", "mortality_7d",
#            "gose_6m", "gose_30d", "TIER", "TIL"]
OUTCOMES = ["mortality_6m"]

FEATURES_old = ["traumatrix", "segmentation", "traumatrix_and_segmentation", # old naming convention for features
            "all_prehospital", "all_prehospital_and_segmentation", "all_DCA",
            "all_DCA_and_segmentation"]

FEATURES = ["PREHOSP", "CT-TIQUA", "MULTI", 
            "PREHOSP-X", "MULTI-PRE", "RESUS-X", "MULTI-RESUS"]

def compute_shap_conservative(outcome, feature_set):

# returns only the shap values on the test set of the fold from the best model

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
    X = translate_column_names(X)

    # model pipeline (minority class oversampling + majority class undersampling + model)
    pipeline_smote_under = Pipeline(steps=[('over', SMOTE()), ('under', RandomUnderSampler(sampling_strategy=0.5)), ('model', HistGradientBoostingClassifier())])
    
    inner_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)


    # hyperparameter grid search
    ftwo_scorer = make_scorer(fbeta_score, beta=2) # the search is optimized for F2 score
    
    
    p_grid = {"model__learning_rate": [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 1], "over__sampling_strategy": [0.1, 0.2, 0.3], "over__k_neighbors":[3,5,8], "under__sampling_strategy":[0.3, 0.5, 0.7]}

    # light version for testing
    #p_grid = {"model__learning_rate": [0.2, 0.3, 0.5, 1], "over__sampling_strategy": [0.2], "over__k_neighbors":[5], "under__sampling_strategy":[0.5]}

    clf = GridSearchCV(estimator=pipeline_smote_under, param_grid=p_grid, scoring={'F2':ftwo_scorer}, refit='F2', cv=inner_cv)

    outer_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

    nested_scores_smote_undersampling = cross_validate(clf, X, y, 
                                                    scoring={'F2':ftwo_scorer, 'ROC_AUC':'roc_auc', 'Recall':'recall_macro', 'F1':'f1', 'Brier':"neg_brier_score", 'False_neg_scorer':false_neg_scorer, 'False_pos_scorer':false_pos_scorer}, 
                                                    cv=outer_cv, n_jobs=-1, return_estimator=True, return_indices=True)

    # once the best hyperparameters are found, we can extract them and train the final model
    best_params_list = []

    for estimator in nested_scores_smote_undersampling["estimator"]:
        best_params_list.append(frozenset(estimator.best_params_.items()))

    # Convert dict to a hashable tuple and count occurrences
    most_common_params = Counter(best_params_list).most_common(1)[0][0]

    # final params hold the most common occurence of a hyperparameter combination
    final_best_params = dict(most_common_params)  # Convert back to dict

    # nested_scores_smote_undersampling = cross_validate(..., return_estimator=True, ...)
    estimators = nested_scores_smote_undersampling['estimator']  # list of fitted GridSearchCV per outer fold

    # Try to get test indices returned by cross_validate (if present)
    test_indices_from_cv = nested_scores_smote_undersampling['indices']['test']

    #print("test_indices_from_cv")
    #print(test_indices_from_cv)

    # Now iterate and access the test samples and predictions of the best inner model
    results = []
    for i, estimator in enumerate(estimators):

        if final_best_params == estimator.best_params_:
            
            if not os.path.exists(f"SHAP_plots/all/shap_{outcome}_{feature_set}.png"):

                print("Found matching best params in fold", i)

                # estimator should be a fitted GridSearchCV; best_estimator_ is the refit pipeline
                best_pipeline = estimator.best_estimator_ if isinstance(estimator, GridSearchCV) else estimator

                idx = test_indices_from_cv[i]

                explainer = shap.TreeExplainer(best_pipeline['model']) 

                # for all test set predictions
                shap_values = explainer(X.loc[idx])

                plt.title(f"SHAP values for {outcome} outcome and {feature_set} data.")
                shap.plots.beeswarm(shap_values, max_display=20, show=False)

                plt.savefig(f"SHAP_plots/all/shap_{outcome}_{feature_set}.png", dpi=300, bbox_inches='tight')
                plt.clf()




                # for TP, TN, FP, FN
                y_true_train = np.asarray(y)[nested_scores_smote_undersampling["indices"]["train"][i]]
                y_true_test = np.asarray(y)[nested_scores_smote_undersampling["indices"]["test"][i]]

                fitted = best_pipeline.fit(X.loc[nested_scores_smote_undersampling["indices"]["train"][i]], y_true_train) #-#=#

                pred_test = fitted.predict(X.loc[nested_scores_smote_undersampling["indices"]["test"][i]])

                # Identify TP, TN, FP, FN indices
                tp_idx = [idx for idx, (yt, yp) in zip(test_indices_from_cv[i], zip(y_true_test, pred_test)) if yt == 1 and yp == 1]
                tn_idx = [idx for idx, (yt, yp) in zip(test_indices_from_cv[i], zip(y_true_test, pred_test)) if yt == 0 and yp == 0]
                fp_idx = [idx for idx, (yt, yp) in zip(test_indices_from_cv[i], zip(y_true_test, pred_test)) if yt == 0 and yp == 1]
                fn_idx = [idx for idx, (yt, yp) in zip(test_indices_from_cv[i], zip(y_true_test, pred_test)) if yt == 1 and yp == 0]

                X_tp = X.loc[tp_idx]
                X_tn = X.loc[tn_idx]
                X_fp = X.loc[fp_idx]
                X_fn = X.loc[fn_idx]

                shap_values_tp = explainer(X_tp)
                shap_values_tn = explainer(X_tn)
                shap_values_fp = explainer(X_fp)
                shap_values_fn = explainer(X_fn)

                try:
                    plt.title(f"SHAP values true positive for {outcome} outcome and {feature_set} data.")
                    shap.plots.beeswarm(shap_values_tp, max_display=20, show=False)
                
                    plt.savefig(f"SHAP_plots/all_by_type/shap_true_positive_{outcome}_{feature_set}.png", dpi=300, bbox_inches='tight')
                    plt.clf()
                except:
                    print("No true positives to plot for this fold.")

                try:
                    plt.title(f"SHAP values true negative for {outcome} outcome and {feature_set} data.")
                    shap.plots.beeswarm(shap_values_tn, max_display=20, show=False)

                    plt.savefig(f"SHAP_plots/all_by_type/shap_true_negative_{outcome}_{feature_set}.png", dpi=300, bbox_inches='tight')
                    plt.clf()
                except:
                    print("No true negatives to plot for this fold.")

                try:
                    plt.title(f"SHAP values false positive for {outcome} outcome and {feature_set} data.")
                    shap.plots.beeswarm(shap_values_fp, max_display=20, show=False)

                    plt.savefig(f"SHAP_plots/all_by_type/shap_false_positive_{outcome}_{feature_set}.png", dpi=300, bbox_inches='tight')
                    plt.clf()
                except:
                    print("No false positives to plot for this fold.")

                try:
                    plt.title(f"SHAP values false negative for {outcome} outcome and {feature_set} data.")
                    shap.plots.beeswarm(shap_values_fn, max_display=20, show=False)

                    plt.savefig(f"SHAP_plots/all_by_type/shap_false_negative_{outcome}_{feature_set}.png", dpi=300, bbox_inches='tight')
                    plt.clf()
                except:
                    print("No false negatives to plot for this fold.")

            break
        


if __name__ == "__main__":
    for outcome in OUTCOMES:
        for i, feature in enumerate(FEATURES):
            compute_shap_conservative(outcome, feature)