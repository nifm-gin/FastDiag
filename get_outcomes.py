import pandas as pd
import numpy as np

DATA_DIRECTORY = "data/"

def get_gose_30d():
    columns_to_load = [96, 1]

    gose_30d = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    print(gose_30d.head())
    # Drop the first row
    gose_30d = gose_30d.iloc[1:, :]

    # Rename the first column to 'name'
    gose_30d.rename(columns={gose_30d.columns[0]: 'name'}, inplace=True)

    gose_30d.rename(columns={gose_30d.columns[1]: 'mortality'}, inplace=True)

    # Drop rows with NaN in 'name'
    gose_30d = gose_30d.dropna(subset=['name'])


    data = gose_30d.copy()

    # Création de la colonne 'gose'
    data['mortality'] = data['mortality'].apply(
        lambda x: 0 if x in ['8 Upper Good Recovery (Upper GR)', 
                            '7 Lower Good Recovery (Lower GR)', 
                            '6 Upper Moderate Disability (Upper MD)',
                            'max'] else 
                np.nan if pd.isnull(x) or x in ['', 'nd', 'NaN'] else 
                1
    ).astype(float)


    # Compter le nombre de 1, de 0 et de NaN dans la colonne 'mortality'
    count_1 = (data['mortality'] == 1).sum()  # Nombre de 1
    count_0 = (data['mortality'] == 0).sum()  # Nombre de 0
    count_nan = data['mortality'].isna().sum()  # Nombre de NaN

    # Afficher les résultats
    print(f"Nombre de 1 : {count_1}")
    print(f"Nombre de 0 : {count_0}")
    print(f"Nombre de NaN : {count_nan}")

    # Create the 'tier_bin' column based on conditions
    y = data[['name']].copy()
    y['mortality'] = (
        (data.iloc[:, 1] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['mortality'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y

def get_gose_6m():
    columns_to_load = [97, 1]

    gose_6m = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    print(gose_6m.head())
    # Drop the first row
    gose_6m = gose_6m.iloc[1:, :]

    # Rename the first column to 'name'
    gose_6m.rename(columns={gose_6m.columns[0]: 'name'}, inplace=True)

    gose_6m.rename(columns={gose_6m.columns[1]: 'mortality'}, inplace=True)

    # Drop rows with NaN in 'name'
    gose_6m = gose_6m.dropna(subset=['name'])


    data = gose_6m.copy()

    # Création de la colonne 'gose'
    data['mortality'] = data['mortality'].apply(
        lambda x: 0 if x in ['8 Upper Good Recovery (Upper GR)', 
                            '7 Lower Good Recovery (Lower GR)', 
                            '6 Upper Moderate Disability (Upper MD)',
                            'max'] else 
                np.nan if pd.isnull(x) or x in ['', 'nd', 'NaN'] else 
                1
    ).astype(float)


    # Compter le nombre de 1, de 0 et de NaN dans la colonne 'mortality'
    count_1 = (data['mortality'] == 1).sum()  # Nombre de 1
    count_0 = (data['mortality'] == 0).sum()  # Nombre de 0
    count_nan = data['mortality'].isna().sum()  # Nombre de NaN

    # Afficher les résultats
    print(f"Nombre de 1 : {count_1}")
    print(f"Nombre de 0 : {count_0}")
    print(f"Nombre de NaN : {count_nan}")

    # Create the 'tier_bin' column based on conditions
    y = data[['name']].copy()
    y['mortality'] = (
        (data.iloc[:, 1] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['mortality'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y


def get_mortality_7d():
    
    columns_to_load = [93, 1] 

    mortality_7d = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    print(mortality_7d.head())
    
    # Drop the first row
    #mortality_7d = mortality_7d.iloc[1:, :]

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[0]: 'name'}, inplace=True)


    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[1]: 'mortality'}, inplace=True)

    data = mortality_7d.copy()
    
    # Exemple : appliquer la transformation sur les colonnes en position 1 et 2
    cols = [data.columns[1]]  # Noms des colonnes en position 1 et 3

    # Appliquer les transformations à chaque colonne
    for col in cols:
        data[col] = data[col].replace({'1': 1, '0': 0, 'nd': np.nan}).astype(float)
        data[col] = pd.to_numeric(data[col], errors='coerce')


    # Create the 'mortality' column based on conditions
    y = data[['name']].copy()
    y['mortality'] = (
        (data.iloc[:, 1] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['mortality'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y

def get_mortality_30d():
    
    columns_to_load = [93, 94, 1] 

    mortality_7d = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    print(mortality_7d.head())
    
    # Drop the first row
    #mortality_7d = mortality_7d.iloc[1:, :]

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[0]: 'name'}, inplace=True)

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[1]: 'mortality D7'}, inplace=True)

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[2]: 'mortality'}, inplace=True)

    data = mortality_7d.copy()
    
    # Exemple : appliquer la transformation sur les colonnes en position 1 et 2
    cols = [data.columns[1], data.columns[2]]  # Noms des colonnes en position 1 et 3

    # Appliquer les transformations à chaque colonne
    for col in cols:
        data[col] = data[col].replace({'1': 1, '0': 0, 'nd': np.nan}).astype(float)
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Recodage de la colonne mortality
    data['mortality'] = data['mortality D7'].where(data['mortality D7'] == 1, data['mortality'])

    # Create the 'mortality' column based on conditions
    y = data[['name']].copy()
    y['mortality'] = (
        (data.iloc[:, 2] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['mortality'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y
    

def get_mortality_6m():

    columns_to_load = [93, 94, 95, 1] 

    mortality_7d = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    
    # Drop the first row
    #mortality_7d = mortality_7d.iloc[1:, :]

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[0]: 'name'}, inplace=True)

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[1]: 'mortality D7'}, inplace=True)

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[2]: 'mortality D30'}, inplace=True)

    # Rename the first column to 'IPP'
    mortality_7d.rename(columns={mortality_7d.columns[3]: 'mortality'}, inplace=True)

    data = mortality_7d.copy()
    
    # Exemple : appliquer la transformation sur les colonnes en position 1 et 2
    cols = [data.columns[1], data.columns[2], data.columns[3]]  # Noms des colonnes en position 1 et 3

    # Appliquer les transformations à chaque colonne
    for col in cols:
        data[col] = data[col].replace({'1': 1, '0': 0, 'nd': np.nan, '8 Upper Good Recovery (Upper GR)': np.nan}).astype(float)
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # Recodage de la colonne mortality
    data['mortality D30'] = data['mortality D7'].where(data['mortality D7'] == 1, data['mortality D30'])
    data['mortality'] = data['mortality D30'].where(data['mortality D30'] == 1, data['mortality'])

    # Create the 'mortality' column based on conditions
    y = data[['name']].copy()
    y['mortality'] = (
        (data.iloc[:, 3] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['mortality'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y


def get_tier():
    # Load the columns 71 to 75 and the 6th column
    columns_to_load = list(range(58, 69)) + [1]

    TIER = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)

    # Colonnes à traiter
    columns_to_convert = TIER.columns[7:12]

    # Remplacer uniquement '1' et '0' par des nombres
    for col in columns_to_convert:
        TIER[col] = TIER[col].replace({'1': 1, '0': 0})  # Convertir '1' et '0' en nombres
        TIER[col] = pd.to_numeric(TIER[col], errors='coerce')  # Convertir le reste en numérique, NaN pour les non-convertibles

    # Create the 'tier_bin' column based on conditions
    y = TIER[['name']].copy()
    y['tier_bin'] = (
        (TIER.iloc[:, 7] == 1) | 
        (TIER.iloc[:, 8] == 1) | 
        (TIER.iloc[:, 9] == 1) | 
        (TIER.iloc[:, 10] == 1) | 
        (TIER.iloc[:, 11] == 1)
    ).astype(int)

    # Display the resulting DataFrame
    print(y.head())

    # Outcome event
    event_count = (y['tier_bin'] == 1).sum()  # Count the number of events
    print(f"Outcome events : {event_count}")

    return y

def get_til():
    # Load the columns 71 to 75 and the 6th column
    columns_to_load = list(range(69, 74)) + [1]

    TIL = pd.read_csv(DATA_DIRECTORY + "clinical_data_anonymized.csv", usecols=columns_to_load)
    

    # Drop the first row
    TIL = TIL.iloc[1:, :]

    # Rename the first column to 'name'
    TIL.rename(columns={TIL.columns[0]: 'name'}, inplace=True)
    TIL.rename(columns={TIL.columns[1]: 'TIL 0'}, inplace=True)
    TIL.rename(columns={TIL.columns[2]: 'TIL 1'}, inplace=True)
    TIL.rename(columns={TIL.columns[3]: 'TIL 2'}, inplace=True)
    TIL.rename(columns={TIL.columns[4]: 'TIL 3'}, inplace=True)
    TIL.rename(columns={TIL.columns[5]: 'TIL 4'}, inplace=True)

    TIL = TIL.dropna(subset=['name'])

    # Display the resulting DataFrame
    print(TIL.head())

    y = TIL[['name']].copy()  # Include 'name' in y
    y["TIL_bin"] = ((TIL.iloc[:, 3] == 1) | (TIL.iloc[:, 4] == 1) | (TIL.iloc[:, 5] == 1)).astype(int)

    # Verify the first few rows of y
    print(y.head())

    # Outcome event
    event_count = (y["TIL_bin"] == 1).sum()  # Count the number of events (y = 1)
    print(f"Outcome events: {event_count}")

    return y

def get_traumatrix():
    pass