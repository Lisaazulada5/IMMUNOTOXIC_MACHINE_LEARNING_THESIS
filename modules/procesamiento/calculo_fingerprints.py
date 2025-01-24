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
def convertir_a_numpy(fingerprint):
    # Crea un array vacío de tamaño igual al número de bits del fingerprint (en este caso, 167)
    numpy_array = np.zeros((fingerprint.GetNumBits(),), dtype=int)

    # Convierte el fingerprint (ExplicitBitVect) a un array de NumPy
    DataStructs.ConvertToNumpyArray(fingerprint, numpy_array)

    return numpy_array


# Función para aplicar la conversión a una columna del DataFrame
def convertir_MACCS_a_numpy(df, columna_MACCS):
    # Aplica la función de conversión a cada elemento de la columna 'columna_MACCS'
    df['MACCS_array'] = df[columna_MACCS].apply(convertir_a_numpy)
    return df


"""
MORGAN FINGERPRINTS
"""


def calcular_ecfp(df, columna_smiles, radio=3, nBits=1024):
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

def convertir_ECFP_a_numpy(df, columna_ECFP):
    # Aplica la función de conversión a cada elemento de la columna 'columna_MACCS'
    df['ECFP_array'] = df[columna_ECFP].apply(convertir_a_numpy)
    return df