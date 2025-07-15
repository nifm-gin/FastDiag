import pandas as pd
import numpy as np
from utility.utility import calculate_dtc

DATA_DIRECTORY = "data/"

def get_segmentation(with_name=True):
    """
    Load, preprocess and return the 14 features obtained from the segmentation of the CT scans.
    
    Features: 
        supratentorial_IPH, supratentorial_SAH, supratentorial_Petechiae,
        supratentorial_Edema, infratentorial_IPH, infratentorial_SAH,
        infratentorial_Petechiae, infratentorial_Edema, brainstem_IPH
    """

    columns_to_load = list(range(101, 115))
    if with_name:
        columns_to_load.append(2)

    X_segmentation = pd.read_csv(DATA_DIRECTORY+"final_database_clinical_segmentation.csv", usecols=columns_to_load)

    # sort by name
    #X_traumatrix = X_traumatrix.sort_values(by='name').reset_index(drop=True)

    return X_segmentation

def get_traumatrix(with_name=True):
    """
    Load, preprocess and return the 15 features used in the Traumatrix studies.
    
    Features: 
        age, hemocue_initial, fracas_du_bassin, catecholamines,
        pression_arterielle_systolique_PAS_arrivee_du_smur,
        pression_arterielle_diastolique_PAD_arrivee_du_smur,
        score_glasgow_initial, score_glasgow_moteur_initial, 
        anomalie_pupillaire_prehospitalier, frequence_cardiaque_FC_arrivee_du_smur,
        arret_cardio_respiratoire_massage, penetrant_objet, ischemie_du_membre,
        hemorragie_externe, amputation
    """

    columns_to_load = list(range(115, 130))

    if with_name:
        columns_to_load.append(2)

    X_traumatrix = pd.read_csv(DATA_DIRECTORY+"final_database_clinical_segmentation.csv", usecols=columns_to_load)

    # sort by name
    #X_traumatrix = X_traumatrix.sort_values(by='name').reset_index(drop=True)

    return X_traumatrix

def get_traumatrix_and_segmentation(with_name=True):
    """
    Load, preprocess and return the 29 features obtained from the combination of Traumatrix and segmentation.
    
    Features: 
        supratentorial_IPH, supratentorial_SAH, supratentorial_Petechiae,
        supratentorial_Edema, infratentorial_IPH, infratentorial_SAH,
        infratentorial_Petechiae, infratentorial_Edema, brainstem_IPH,
        age, hemocue_initial, fracas_du_bassin, catecholamines,
        pression_arterielle_systolique_PAS_arrivee_du_smur,
        pression_arterielle_diastolique_PAD_arrivee_du_smur,
        score_glasgow_initial, score_glasgow_moteur_initial, 
        anomalie_pupillaire_prehospitalier, frequence_cardiaque_FC_arrivee_du_smur,
        arret_cardio_respiratoire_massage, penetrant_objet, ischemie_du_membre,
        hemorragie_externe, amputation
    """

    X_traumatrix = get_traumatrix(with_name=with_name)
    X_segmentation = get_segmentation(with_name=False)

    X_traumatrix_segmentation = pd.concat([X_traumatrix, X_segmentation], axis=1)

    return X_traumatrix_segmentation

def get_all_prehospital(with_name=True):

    """Load, preprocess and return the 20 features obtained from the prehospital data."""

    data_full = pd.read_csv(DATA_DIRECTORY+"final_database_clinical_segmentation.csv")

    X_prehosp = data_full.iloc[:, 15:35]

    # Convert all columns to numeric, replacing non-numeric values and empty strings with NaN
    X_prehosp = X_prehosp.apply(pd.to_numeric, errors='coerce')

    if with_name:
        # Include column name
        X_prehosp = pd.concat([X_prehosp, data_full.iloc[:, 2]], axis=1)

    return X_prehosp

def get_all_prehospital_and_segmentation(with_name=True):

    X_all_prehosp = get_all_prehospital(with_name=with_name)
    X_segmentation = get_segmentation(with_name=False)

    X_prehosp_segmentation = pd.concat([X_all_prehosp, X_segmentation], axis=1)

    return X_prehosp_segmentation


def get_all_DCA(with_name=True):
    """Load, preprocess and return the 14 features obtained from the DCA (Déchocage or resuscitation room)"""
    
    data_full = pd.read_csv(DATA_DIRECTORY+"final_database_clinical_segmentation.csv")
    X_dca = data_full.iloc[:, 35:49]

    # Define the conditions for dtc
    # Apply the function row-wise to create the new column
    X_dca['dtc'] = X_dca.apply(calculate_dtc, axis=1)

    X_dca = X_dca.apply(pd.to_numeric, errors='coerce')

    if with_name:
        X_dca = pd.concat([X_dca, data_full.iloc[:, 2]], axis=1)
    
    return X_dca

def get_all_DCA_and_segmentation(with_name=True):
    """
    Load, preprocess and return the 28 features obtained from the combination of DCA and segmentation.
    
    """

    X_dca = get_all_DCA(with_name=with_name)
    X_segmentation = get_segmentation(with_name=False)

    X_dca_segmentation = pd.concat([X_dca, X_segmentation], axis=1)

    return X_dca_segmentation

