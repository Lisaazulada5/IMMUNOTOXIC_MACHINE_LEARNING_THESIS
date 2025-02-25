import pubchempy as pcp


import urllib.error

def obtener_iupac_desde_pubchem(nombre_comun):
    try:
        compuestos = pcp.get_compounds(nombre_comun, 'name', timeout=15)  # Timeout de 30 segundos
        if compuestos:
            return compuestos[0].iupac_name
        return None
    except urllib.error.URLError as e:
        print(f"Error de conexión: {e}")
        return None


import requests
import pandas as pd
import time


def obtener_smiles_pubchem(df, columna_dtxsid, columna_smiles='SMILES', delay=0.5):
    """
    Consulta PubChem para obtener los SMILES asociados a los identificadores DTXSID en un DataFrame.

    Parámetros:
    df: pd.DataFrame - DataFrame que contiene la columna con los identificadores DTXSID.
    columna_dtxsid: str - Nombre de la columna que contiene los identificadores DTXSID.
    columna_smiles: str - Nombre de la nueva columna donde se guardarán los SMILES (default: 'SMILES').
    delay: float - Tiempo de espera (en segundos) entre cada consulta para evitar bloqueos (default: 0.5).

    Retorna:
    pd.DataFrame - DataFrame con una nueva columna que contiene los SMILES obtenidos.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"

    # Crear una nueva columna de SMILES inicializada con valores vacíos
    df[columna_smiles] = None

    for index, row in df.iterrows():
        dtxsid = row[columna_dtxsid]
        if pd.isnull(dtxsid):
            continue  # Ignorar valores nulos en la columna DTXSID

        try:
            # Hacer la consulta a PubChem
            url = base_url.format(dtxsid)
            response = requests.get(url)

            if response.status_code == 200:
                # Parsear la respuesta JSON
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                df.at[index, columna_smiles] = smiles
            else:
                print(f"Error {response.status_code} para DTXSID: {dtxsid}")
        except Exception as e:
            print(f"Error al obtener SMILES para DTXSID {dtxsid}: {e}")

        # Pausa para evitar bloquear el servidor
        time.sleep(delay)

    return df

import requests
import pandas as pd
import time

def obtener_smiles_pubchem_cas(df, columna_cas, delay=0.5):
    """
    Consulta PubChem para obtener los SMILES asociados a los números CAS en un DataFrame.

    Parámetros:
    df: pd.DataFrame - DataFrame que contiene la columna con los números CAS.
    columna_cas: str - Nombre de la columna que contiene los números CAS.
    delay: float - Tiempo de espera (en segundos) entre cada consulta para evitar bloqueos (default: 0.5).

    Retorna:
    pd.DataFrame - DataFrame con una nueva columna 'SMILES' que contiene los valores obtenidos.
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/CanonicalSMILES/JSON"

    # Crear una nueva columna de SMILES inicializada con valores vacíos
    df["SMILES"] = None

    for index, row in df.iterrows():
        cas_number = row[columna_cas]
        if pd.isnull(cas_number):
            continue  # Ignorar valores nulos en la columna CAS

        try:
            # Hacer la consulta a PubChem
            url = base_url.format(cas_number)
            response = requests.get(url)

            if response.status_code == 200:
                # Parsear la respuesta JSON
                data = response.json()
                smiles = data['PropertyTable']['Properties'][0]['CanonicalSMILES']
                df.at[index, "SMILES"] = smiles
            else:
                print(f"Error {response.status_code} para CAS: {cas_number}")
        except Exception as e:
            print(f"Error al obtener SMILES para CAS {cas_number}: {e}")

        # Pausa para evitar bloquear el servidor
        time.sleep(delay)

    return df


import pubchempy as pcp
import pandas as pd
import time

def obtener_iupac_desde_pubchem(df, columna_smiles, columna_iupac='IUPAC', delay=0.5):
    """
    Consulta PubChem para obtener los nombres IUPAC asociados a los SMILES en un DataFrame.

    Parámetros:
    df: pd.DataFrame - DataFrame que contiene la columna con los SMILES.
    columna_smiles: str - Nombre de la columna que contiene los SMILES.
    columna_iupac: str - Nombre de la nueva columna donde se guardarán los nombres IUPAC (default: 'IUPAC').
    delay: float - Tiempo de espera (en segundos) entre cada consulta para evitar bloqueos (default: 0.5).

    Retorna:
    pd.DataFrame - DataFrame con una nueva columna que contiene los nombres IUPAC obtenidos.
    """
    df[columna_iupac] = None

    for index, row in df.iterrows():
        smiles = row[columna_smiles]
        if pd.isnull(smiles):
            continue  # Ignorar valores nulos en la columna SMILES

        try:
            compuestos = pcp.get_compounds(smiles, 'smiles', timeout=15)
            if compuestos:
                df.at[index, columna_iupac] = compuestos[0].iupac_name
        except Exception as e:
            print(f"Error al obtener IUPAC para SMILES {smiles}: {e}")

        time.sleep(delay)  # Pausa para evitar bloqueos del servidor

    return df
