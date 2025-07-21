from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import confusion_matrix

import numpy as np
import pandas as pd

ftwo_scorer = make_scorer(fbeta_score, beta=2)

def confusion_matrix_scorer(clf, X, y):

     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)

     return {'tn': cm[0, 0], 'fp': cm[0, 1],
             'fn': cm[1, 0], 'tp': cm[1, 1]}

def false_neg_scorer(clf, X, y):

     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)
     
     return cm[1, 0]

def false_pos_scorer(clf, X, y):

     y_pred = clf.predict(X)
     cm = confusion_matrix(y, y_pred)
     
     return cm[0, 1]

# Helper function to check if a value is numeric
def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def calculate_dtc(row):
     # Condition for dtc = NA
     if ('nr' in [row['DTC Vd'], row['DTC IP']] or
          'non réalisé' in [row['DTC Vd'], row['DTC IP']]):
          return np.nan
     # Condition for dtc = 1
     if (row['DTC Vd'] == 'Pathologique' or row['DTC IP'] == 'Pathologique' or
          (is_numeric(row['DTC Vd']) and float(row['DTC Vd']) < 30) or
          (is_numeric(row['DTC IP']) and float(row['DTC IP']) > 1.2)):
          return 1
     # Default case for dtc = 0
     return 0

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
