import os
import csv
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import fbeta_score, confusion_matrix, roc_auc_score, f1_score, brier_score_loss

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from utility.utility import false_neg_scorer, false_pos_scorer
from sklearn.metrics import fbeta_score, make_scorer

from collections import Counter

from get_outcomes import *
from get_data import *


DATA_DIRECTORY = "data/"

OUTCOMES = ["mortality_6m", "mortality_30d", "mortality_7d",
            "gose_6m", "gose_30d", "TIER", "TIL"]

FEATURES = ["traumatrix", "segmentation", "traumatrix_and_segmentation",
            "all_prehospital", "all_prehospital_and_segmentation", "all_DCA",
            "all_DCA_and_segmentation"
            ]

FOLDS = 5
N_REPEATS = 3

def align_X_y_and_clean(X,y):
    # Identify indexes where 'mortality' is NaN in y
    nan_indexes = y.loc[pd.isna(y.iloc[:,1]), :].index
    print(f"Indexes with NaN in 'mortality': {nan_indexes}")

    # Drop rows with NaN in y
    y = y.drop(index=nan_indexes)

    # Ensure 'name' is the index or used for alignment
    X = X[~X['name'].isin(y.loc[nan_indexes, 'name'])]

    # Verify the shapes after cleaning
    print(f"Shape of y after cleaning: {y.shape}")
    print(f"Shape of X after cleaning: {X.shape}")

    # Number of unique values of "name" in y
    print("Unique 'name' values in y:", y['name'].nunique())

    # Number of unique values of "name" in X
    print("Unique 'name' values in X:", X['name'].nunique())

    missing_in_X = set(y['name']) - set(X['name'])
    print(f"Names in y but not in X: {missing_in_X}")

    missing_in_y = set(X['name']) - set(y['name'])
    print(f"Names in X but not in y: {missing_in_y}")

    # Find rows with names that are common to both y and X
    common_names = set(y['name']).intersection(set(X['name']))

    # Keep only rows where the name is both in X and y
    y = y[y['name'].isin(common_names)]
    X = X[X['name'].isin(common_names)]

    print(f"Shape of y after alignment: {y.shape}")
    print(f"Shape of X after alignment: {X.shape}")

    # Check for duplicates in 'name' in y
    print("Number of duplicate names in y:", y['name'].duplicated().sum())

    # Check for duplicates in 'name' in X
    print("Number of duplicate names in X:", X['name'].duplicated().sum())

    # Remove duplicates from both DataFrames
    y = y.drop_duplicates(subset=['name'])
    X = X.drop_duplicates(subset=['name'])

    # Verify the shapes again
    print(f"Shape of y after removing duplicates: {y.shape}")
    print(f"Shape of X after removing duplicates: {X.shape}")

    # Check if 'name' values are the same in both DataFrames
    common_names = set(y['name']) == set(X['name'])
    print(f"Are 'name' values aligned between y and X {common_names}")

    # Sort y and X by 'name'
    y = y.sort_values(by='name').reset_index(drop=True)
    X = X.sort_values(by='name').reset_index(drop=True)

    # Verify that the names are aligned
    print((y['name'].values == X['name'].values).all())

    # Drop the 'name' column from X
    #X_features = X.drop(columns=['name', 'mortalité J7'])
    X_features = X.drop(columns=['name'])

    # imputation with median 
    X_features_imputed = X_features.fillna(X_features.median())

    # Drop the 'name' column from y (if it exists)
    y_outcome = y.drop(columns=['name'])

    # Ensure y_outcome is a 1D array
    y_outcome = y_outcome.values.ravel()  # Convert to 1D if using pandas DataFrame
    return X_features_imputed, y_outcome


def compute_model(feature_set, outcome):
    """
    Trains a model for a given feature set and outcome.
    Performs hyperparameter tuning using grid search with nested cross-validation.
    Returns a dictionnary with the best hyperparameter combination and the 
    performance metrics along with their 95% confidence intervals.
    """

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
    if feature_set == "traumatrix":
        X = get_traumatrix(with_name=True)

    elif feature_set == "segmentation":
        X = get_segmentation(with_name=True)

    elif feature_set == "traumatrix_and_segmentation":
        X = get_traumatrix_and_segmentation(with_name=True)

    elif feature_set == "all_prehospital":
        X = get_all_prehospital(with_name=True)

    elif feature_set == "all_prehospital_and_segmentation":
        X = get_all_prehospital_and_segmentation(with_name=True)

    elif feature_set == "all_DCA":
        X = get_all_DCA(with_name=True)

    elif feature_set == "all_DCA_and_segmentation":
        X = get_all_DCA_and_segmentation(with_name=True)

    # align, clean and imputation 
    X, y = align_X_y_and_clean(X, y)

    # model pipeline (minority class oversampling + majority class undersampling + model)
    pipeline_smote_under = Pipeline(steps=[('over', SMOTE()), ('under', RandomUnderSampler(sampling_strategy=0.5)), ('model', HistGradientBoostingClassifier())])
    
    inner_cv = RepeatedStratifiedKFold(n_splits=FOLDS, n_repeats=5, random_state=1)


    # hyperparameter grid search
    ftwo_scorer = make_scorer(fbeta_score, beta=2) # the search is optimized for F2 score
    # parameter grid
    p_grid = {"model__learning_rate": [0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.5, 1], "over__sampling_strategy": [0.1, 0.2, 0.3], "over__k_neighbors":[3,5,8], "under__sampling_strategy":[0.3, 0.5, 0.7]}
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
    final_params = dict(most_common_params)  # Convert back to dict
    
    # The final model can then be trained
    print("Final chosen hyperparameters:", final_params)
    pipeline = Pipeline(steps=[
    ('over', SMOTE(k_neighbors=final_params["over__k_neighbors"], sampling_strategy=final_params["over__sampling_strategy"])), 
    ('under', RandomUnderSampler(sampling_strategy=final_params["under__sampling_strategy"])), 
    ('model', HistGradientBoostingClassifier(learning_rate=final_params["model__learning_rate"]))])

    y_pred = cross_val_predict(pipeline, X, y, cv=20, method='predict_proba')[:,1]

    y_pred_binary = [1 if i>=0.5 else 0 for i in y_pred]

    # compute confidence intervals by bootstraping the test set
    rng = np.random.RandomState(seed=12345)
    idx = np.arange(len(y))

    y_pred = np.asarray(y_pred)
    y_pred_binary = np.asarray(y_pred_binary)
    y = np.asarray(y)

    test_roc_auc = []
    test_f1 = []
    test_ftwos = []
    test_brier = []
    test_false_neg = []
    test_false_pos = []
    test_sensitivity = []
    test_specificity = []
    test_PPV = []
    test_NPV = []
    test_lr_plus = []
    test_lr_minus = []
    test_youden_index = []
    test_tn = []
    test_fp = []
    test_fn = []
    test_tp = []

    for i in range(200): 
        # bootstrap with 200 rounds: random sampling with replacement of the predictions

        pred_idx = rng.choice(idx, size=len(idx), replace=True)
        
        roc_auc_test_boot = roc_auc_score(y_score=y_pred[pred_idx], y_true=y[pred_idx])
        f1_test_boot = f1_score(y_pred=y_pred_binary[pred_idx], y_true=y[pred_idx])
        f2_test_boot = fbeta_score(y_pred=y_pred_binary[pred_idx], y_true=y[pred_idx], beta=2)
        brier_test_boot = brier_score_loss(y_proba=y_pred[pred_idx], y_true=y[pred_idx])
        false_neg_test_boot = confusion_matrix(y[pred_idx], y_pred_binary[pred_idx])[1,0]
        false_pos_test_boot = confusion_matrix(y[pred_idx], y_pred_binary[pred_idx])[0,1]
        # Sensitivity (Recall) and Specificity
        tn, fp, fn, tp = confusion_matrix(y[pred_idx], y_pred_binary[pred_idx]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        # Positive Predictive Value (PPV) and Negative Predictive Value (NPV)
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        npv = tn / (tn + fn) if (tn + fn) > 0 else np.nan
        # Likelihood Ratios
        lr_plus = sensitivity / (1 - specificity) if (1 - specificity) > 0 else np.nan
        lr_minus = (1 - sensitivity) / specificity if specificity > 0 else np.nan

        # Youden index (sensitivity + specificity - 1)
        youden_index = sensitivity + specificity - 1

        test_roc_auc.append(roc_auc_test_boot)
        test_f1.append(f1_test_boot)
        test_ftwos.append(f2_test_boot)
        test_brier.append(brier_test_boot)
        test_false_neg.append(false_neg_test_boot/len(idx)*100)
        test_false_pos.append(false_pos_test_boot/len(idx)*100)
        test_sensitivity.append(sensitivity * 100)  # Convert to percentage
        test_specificity.append(specificity * 100)
        test_PPV.append(ppv * 100)  # Convert to percentage
        test_NPV.append(npv * 100)  # Convert to percentage
        test_lr_plus.append(lr_plus)
        test_lr_minus.append(lr_minus)
        test_youden_index.append(youden_index)
        test_tn.append(tn)
        test_fp.append(fp)
        test_fn.append(fn)
        test_tp.append(tp)


    print("Classification performance\n")
    output = {"Feature set": feature_set, "Outcome": outcome}

    # Compute the mean and 95% confidence intervals for each metric
    # 95% confidence intervals are computed using the 2.5th and 97.5th percentiles of the bootstrap samples
    bootstrap_roc_auc_test_mean = np.mean(test_roc_auc)
    ci_lower = np.percentile(test_roc_auc, 2.5)     # 2.5 percentile (alpha=0.025)
    ci_upper = np.percentile(test_roc_auc, 97.5)
    output["ROC AUC"] = f"{bootstrap_roc_auc_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"ROC AUC:         {bootstrap_roc_auc_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_f1_test_mean = np.mean(test_f1)
    ci_lower = np.percentile(test_f1, 2.5)
    ci_upper = np.percentile(test_f1, 97.5)
    output["F1"] = f"{bootstrap_f1_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"F1:              {bootstrap_f1_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_f2_test_mean = np.mean(test_ftwos)
    ci_lower = np.percentile(test_ftwos, 2.5)
    ci_upper = np.percentile(test_ftwos, 97.5)
    output["F2"] = f"{bootstrap_f2_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"F2:              {bootstrap_f2_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_brier_test_mean = np.mean(test_brier)
    ci_lower = np.percentile(test_brier, 2.5)
    ci_upper = np.percentile(test_brier, 97.5)
    output["Brier loss"] = f"{bootstrap_brier_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Brier loss:      {bootstrap_brier_test_mean:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_false_neg_test_mean = np.mean(test_false_neg)
    ci_lower = np.percentile(test_false_neg, 2.5)
    ci_upper = np.percentile(test_false_neg, 97.5)
    output["False negatives"] = f"{bootstrap_false_neg_test_mean:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"False negatives: {bootstrap_false_neg_test_mean:.2f}%  - 95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_false_pos_test_mean = np.mean(test_false_pos)
    ci_lower = np.percentile(test_false_pos, 2.5)
    ci_upper = np.percentile(test_false_pos, 97.5)
    output["False positives"] = f"{bootstrap_false_pos_test_mean:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"False positives: {bootstrap_false_pos_test_mean:.2f}%  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_sensitivity = np.mean(test_sensitivity)
    ci_lower = np.percentile(test_sensitivity, 2.5)
    ci_upper = np.percentile(test_sensitivity, 97.5)
    output["Sensitivity"] = f"{bootstrap_sensitivity:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Sensitivity:     {bootstrap_sensitivity:.2f}%  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_specificity = np.mean(test_specificity)
    ci_lower = np.percentile(test_specificity, 2.5)
    ci_upper = np.percentile(test_specificity, 97.5)
    output["Specificity"] = f"{bootstrap_specificity:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Specificity:     {bootstrap_specificity:.2f}%  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_PPV = np.mean(test_PPV)
    ci_lower = np.percentile(test_PPV, 2.5)
    ci_upper = np.percentile(test_PPV, 97.5)
    output["PPV"] = f"{bootstrap_PPV:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"PPV:     {bootstrap_PPV:.2f}%  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_NPV = np.mean(test_NPV)
    ci_lower = np.percentile(test_NPV, 2.5)
    ci_upper = np.percentile(test_NPV, 97.5)
    output["NPV"] = f"{bootstrap_NPV:.2f}%  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"NPV:     {bootstrap_NPV:.2f}%  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_lr_plus = np.mean(test_lr_plus)
    ci_lower = np.percentile(test_lr_plus, 2.5)
    ci_upper = np.percentile(test_lr_plus, 97.5)
    output["LR+"] = f"{bootstrap_lr_plus:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Likelihood Ratio +: {bootstrap_lr_plus:.2f}  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_lr_minus = np.mean(test_lr_minus)
    ci_lower = np.percentile(test_lr_minus, 2.5)
    ci_upper = np.percentile(test_lr_minus, 97.5)
    output["LR-"] = f"{bootstrap_lr_minus:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Likelihood Ratio +: {bootstrap_lr_minus:.2f}  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    bootstrap_youden_index = np.mean(test_youden_index)
    ci_lower = np.percentile(test_youden_index, 2.5)
    ci_upper = np.percentile(test_youden_index, 97.5)
    output["Youden Index"] = f"{bootstrap_youden_index:.2f}  -  95% CI {ci_lower:.2f}-{ci_upper:.2f}"
    #print(f"Youden Index: {bootstrap_youden_index:.2f}  -95% CI {ci_lower:.2f}-{ci_upper:.2f}")

    output["Average TP"] = [np.mean(test_tp)]
    output["Average TN"] = [np.mean(test_tn)]
    output["Average FP"] = [np.mean(test_fp)]
    output["Average FN"] = [np.mean(test_fn)]
    
    output.update({k: str(v) for k, v in final_params.items()})
    
    return output

if __name__ == "__main__":
    
    dataframes_to_concat = []

    for f in FEATURES:
        for o in OUTCOMES:
            print("--------------------------------------------------------")
            print(f"Computing model for feature set: {f} and outcome: {o}")

            output = compute_model(f, o)

            output_dataframe = pd.DataFrame.from_dict(output)
            output_dataframe.to_csv(f"results_summary/results_{f}_{o}.csv", index=True)
            dataframes_to_concat.append(output_dataframe)
    
    final_output_dataframe = pd.concat(dataframes_to_concat, ignore_index=True)
    final_output_dataframe.to_csv("results_summary/final_results.csv", index=True)