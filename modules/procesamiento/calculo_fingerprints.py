from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
"""
Calculo de fingerprints
"""

def calcular_fingerprints_maccs(df, columna_smiles):
    """
    Calcula los fingerprints de MACCS para las moléculas en formato SMILES de una columna
    del DataFrame y los guarda en una nueva columna 'MACCS'.

    :param df: DataFrame que contiene las moléculas en formato SMILES.
    :param columna_smiles: Nombre de la columna que contiene las moléculas en formato SMILES.
    :return: DataFrame con una nueva columna 'MACCS' que contiene los fingerprints de las moléculas.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no se encuentra en el DataFrame.")

        # Crear una lista para almacenar los fingerprints ECFP

    def obtener_fingerprint(smile):
        """Calcula el fingerprint de MACCS para una molécula en formato SMILES."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        return MACCSkeys.GenMACCSKeys(mol)

    # Aplicar la función a cada fila de la columna SMILES y guardar el resultado en la columna 'MACCS'
    df['MACCS'] = df[columna_smiles].apply(obtener_fingerprint)

    return df


import numpy as np

from rdkit import DataStructs
import numpy as np
import pandas as pd
from rdkit.Chem import MACCSkeys

from rdkit.Chem import DataStructs
import numpy as np

"""
CONVERTIR MACCS (BIRVECTOR) A UN ARRAY NUMPY
"""

# Función para convertir ExplicitBitVect a numpy array
def convertir_a_numpy(columna_fingerprint):
    fingerprint = []
    for i in columna_fingerprint:
        # Crea un array vacío de tamaño igual al número de bits del fingerprint (en este caso, 167)
        arr = np.zeros((0,), dtype=np.int8)
        # Convierte el fingerprint (ExplicitBitVect) a un array de NumPy
        DataStructs.ConvertToNumpyArray(i, arr)
        fingerprint.append(arr)
    return fingerprint





# Función para aplicar la conversión a una columna del DataFrame
def convertir_MACCS_a_numpy(df, columna_MACCS):
    # Aplica la función de conversión a cada elemento de la columna 'columna_MACCS'
    df['MACCS_array'] = df[columna_MACCS].apply(convertir_a_numpy)
    return df


"""
MORGAN FINGERPRINTS
"""


def calcular_ecfp(df, columna_smiles, radio=2, nBits=2048):
    """
    Calcula los fingerprints ECFP para cada molécula en el DataFrame y agrega una nueva columna con los resultados.

    :param df: DataFrame que contiene una columna de SMILES.
    :param columna_smiles: El nombre de la columna que contiene las cadenas SMILES.
    :param radio: Radio para calcular los fingerprints (2 para ECFP4, 3 para ECFP6).
    :param nBits: Número de bits para la longitud del vector binario.
    :return: DataFrame con una nueva columna 'ECFP' que contiene los fingerprints ECFP.
    """

    # Crear una lista para almacenar los fingerprints ECFP
    ecfp_list = []

    for smiles in df[columna_smiles]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            # Calcular el fingerprint ECFP como un vector binario
            ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radio, nBits=nBits)
            ecfp_list.append(ecfp)
        else:
            # Si la molécula no es válida, añadir un valor nulo
            ecfp_list.append(None)

    # Agregar la lista de fingerprints al DataFrame como una nueva columna
    df['ECFP'] = ecfp_list

    return df

#def convertir_ECFP_a_numpy(df, columna_ECFP):
    # Aplica la función de conversión a cada elemento de la columna 'columna_MACCS'
    #df['ECFP_array'] = df[columna_ECFP].apply(convertir_a_numpy)
#    return df

def convertir_a_numpy(bitvector):
    arr = np.zeros((bitvector.GetNumBits(),), dtype=np.int8)  # Asegurar el tamaño correcto
    DataStructs.ConvertToNumpyArray(bitvector, arr)
    return arr


import numpy as np
from rdkit import DataStructs


def convertir_bitvector_a_numpy(bitvector):
    """
    Convierte un objeto RDKit ExplicitBitVect en un array de NumPy.

    Parámetros:
    - bitvector: rdkit.DataStructs.cDataStructs.ExplicitBitVect

    Retorna:
    - Un array de NumPy con valores 0 y 1.
    """
    arr = np.zeros((bitvector.GetNumBits(),), dtype=np.int8)  # Asegurar el tamaño correcto
    DataStructs.ConvertToNumpyArray(bitvector, arr)
    return arr


def convertir_ECFP_a_numpy(df, columna_ECFP):
    """
    Convierte la columna de fingerprints en arrays de NumPy.

    Parámetros:
    - df: DataFrame que contiene la columna con fingerprints.
    - columna_ECFP: Nombre de la columna que contiene los ExplicitBitVect.

    Retorna:
    - DataFrame con una nueva columna 'ECFP_array' con los fingerprints como arrays de NumPy.
    """
    df['ECFP_array'] = df[columna_ECFP].apply(convertir_bitvector_a_numpy)
    return df



