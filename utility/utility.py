from sklearn.metrics import fbeta_score, make_scorer
from sklearn.metrics import confusion_matrix

import numpy as np

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