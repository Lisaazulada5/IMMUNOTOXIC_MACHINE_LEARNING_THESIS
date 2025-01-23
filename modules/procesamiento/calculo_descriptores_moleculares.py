from rdkit import Chem
from rdkit.Chem import Crippen
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

from rdkit import Chem

from rdkit import Chem

from rdkit import Chem

def convertir_smiles_a_smarts(df, columna_smiles="SMILES", columna_smarts="SMARTS"):
    """
    Convierte una columna de SMILES a SMARTS en un DataFrame.

    Parámetros:
    - df (DataFrame): DataFrame con la columna de SMILES.
    - columna_smiles (str): Nombre de la columna con SMILES (default: "SMILES").
    - columna_smarts (str): Nombre de la columna donde se guardarán los SMARTS (default: "SMARTS").

    Retorna:
    - DataFrame: DataFrame con la nueva columna de SMARTS.
    """
    def convertir(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.MolToSmarts(mol) if mol else None
        except:
            return None

    df[columna_smarts] = df[columna_smiles].apply(lambda x: convertir(x) if pd.notna(x) else None)
    return df

from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors

from rdkit import Chem
from rdkit.Chem import Descriptors

def calcular_descriptores(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * 8  # Devuelve None si el SMILES no es válido

        # Cálculo de descriptores moleculares
        masa_molecular = Descriptors.ExactMolWt(mol)
        numero_atomos_pesados = Descriptors.HeavyAtomCount(mol)
        h_bond_acceptors = Descriptors.NumHAcceptors(mol)
        h_bond_donors = Descriptors.NumHDonors(mol)

        # Subestructuras específicas usando solo SMILES
        enlaces_dobles = len([bond for bond in mol.GetBonds() if bond.GetBondType().name == 'DOUBLE'])
        enlaces_triples = len([bond for bond in mol.GetBonds() if bond.GetBondType().name == 'TRIPLE'])
        carboxilo = len([atom for atom in mol.GetAtoms() if atom.GetSmarts() == "C(=O)O"])
        hidroxilo = len([atom for atom in mol.GetAtoms() if atom.GetSmarts() == "O[H]"])

        return [
            masa_molecular,
            numero_atomos_pesados,
            h_bond_acceptors,
            h_bond_donors,
            enlaces_dobles,
            enlaces_triples,
            carboxilo,
            hidroxilo,
        ]
    except Exception as e:
        print(f"Error procesando SMILES {smiles}: {e}")
        return [None] * 8

from rdkit import Chem
from rdkit.Chem import Descriptors

def calcular_logp(smiles):
    """
    Calcula el logP de una molécula dada su SMILES.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Devuelve None si el SMILES no es válido
        return Descriptors.MolLogP(mol)
    except Exception as e:
        print(f"Error procesando SMILES {smiles}: {e}")
        return None

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen

# Ejemplo de DataFrame con una columna SMILES

# Función para calcular el LogP
def calcular_logp_crippen(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None  # Devuelve None si el SMILES no es válido
        return Crippen.MolLogP(mol)
    except Exception as e:
        print(f"Error procesando SMILES {smiles}: {e}")
        return None


from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import pandas as pd


def calcular_pesos_moleculares(df, columna_smiles):
    """
    Calcula el peso molecular exacto para una columna SMILES en un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame que contiene la columna de SMILES.
        columna_smiles (str): Nombre de la columna con las cadenas SMILES.

    Returns:
        pd.DataFrame: DataFrame original con una nueva columna 'Peso_Molecular' que contiene los valores calculados.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no está en el DataFrame.")

    def calcular_peso(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"SMILES inválido: {smiles}")
            return ExactMolWt(mol)
        except Exception as e:
            print(f"Error al procesar SMILES '{smiles}': {e}")
            return None

    # Crear nueva columna con los pesos moleculares
    df['Peso_Molecular'] = df[columna_smiles].apply(calcular_peso)
    return df

from rdkit import Chem
from rdkit.Chem import Lipinski

def calcular_numero_aceptores(df, columna_smiles):
    """
    Calcula el número de aceptores de enlaces de hidrógeno para una columna de SMILES en un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columna_smiles (str): Nombre de la columna que contiene las estructuras SMILES.

    Retorna:
        pd.DataFrame: DataFrame con una nueva columna 'NumHAcceptors' con los resultados.
    """
    df['NumHAcceptors'] = df[columna_smiles].apply(
        lambda smile: Lipinski.NumHAcceptors(Chem.MolFromSmiles(smile)) if Chem.MolFromSmiles(smile) else None
    )
    return df

def calcular_numero_donadores(df, columna_smiles):
    """
    Calcula el número de donadores de enlaces de hidrógeno para una columna de SMILES en un DataFrame.

    Parámetros:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columna_smiles (str): Nombre de la columna que contiene las estructuras SMILES.

    Retorna:
        pd.DataFrame: DataFrame con una nueva columna 'NumHAcceptors' con los resultados.
    """
    df['NumHDonors'] = df[columna_smiles].apply(
        lambda smile: Lipinski.NumHDonors(Chem.MolFromSmiles(smile)) if Chem.MolFromSmiles(smile) else None
    )
    return df

"""
calcule el número de enlaces dobles en una molécula a partir de su representación SMILES 
"""
from rdkit import Chem


def calcular_enlaces_dobles(df, columna_smiles):
    """
    Calcula el número de enlaces dobles para cada molécula en la columna SMILES y añade
    una nueva columna al DataFrame con esta información.

    :param df: DataFrame que contiene las moléculas en formato SMILES.
    :param columna_smiles: Nombre de la columna que contiene las moléculas en formato SMILES.
    :return: DataFrame con una nueva columna 'Dobles_Enlaces' que contiene el número de enlaces dobles.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no se encuentra en el DataFrame.")

    # Calcular el número de enlaces dobles para cada molécula
    df['Dobles_Enlaces'] = df[columna_smiles].apply(
        lambda smile: len([bond for bond in Chem.MolFromSmiles(smile).GetBonds()
                           if bond.GetBondType().name == 'DOUBLE']) if Chem.MolFromSmiles(smile) else None
    )
    return df

"""
calcule el número de enlaces triples en una molécula a partir de su representación SMILES 
"""
from rdkit import Chem


def calcular_enlaces_triples(df, columna_smiles):
    """
    Calcula el número de enlaces triples para cada molécula en la columna SMILES y añade
    una nueva columna al DataFrame con esta información.

    :param df: DataFrame que contiene las moléculas en formato SMILES.
    :param columna_smiles: Nombre de la columna que contiene las moléculas en formato SMILES.
    :return: DataFrame con una nueva columna 'triples_Enlaces' que contiene el número de enlaces triples.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no se encuentra en el DataFrame.")

    # Calcular el número de enlaces dobles para cada molécula
    df['Triples_Enlaces'] = df[columna_smiles].apply(
        lambda smile: len([bond for bond in Chem.MolFromSmiles(smile).GetBonds()
                           if bond.GetBondType().name == 'TRIPLE']) if Chem.MolFromSmiles(smile) else None
    )
    return df

"""
NUMERO DE HIDROXILOS
"""

from rdkit import Chem

from rdkit import Chem
import pandas as pd


def calcular_Numero_hidroxilos(df, columna_smiles):
    """
    Calcula el número de grupos hidroxilo (-OH) para cada molécula en la columna SMILES y añade
    una nueva columna al DataFrame con esta información.

    :param df: DataFrame que contiene las moléculas en formato SMILES.
    :param columna_smiles: Nombre de la columna que contiene las moléculas en formato SMILES.
    :return: DataFrame con una nueva columna 'Numero_hidroxilos' que contiene el número de grupos hidroxilo.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no se encuentra en el DataFrame.")

    def contar_hidroxilos(smile):
        """Cuenta los grupos hidroxilo en una molécula dada en formato SMILES."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        # SMARTS para el grupo hidroxilo (-OH)
        hidroxilo_smarts = Chem.MolFromSmarts('[OH]')
        return len(mol.GetSubstructMatches(hidroxilo_smarts))

    # Crear una nueva columna en el DataFrame con el conteo de hidroxilos
    df['Numero_hidroxilos'] = df[columna_smiles].apply(contar_hidroxilos)
    return df

from rdkit import Chem

def calcular_Numero_carboxilos(df, columna_smiles):
    """
    Calcula el número de grupos carboxilo (-COOH) para cada molécula en la columna SMILES y añade
    una nueva columna al DataFrame con esta información.

    :param df: DataFrame que contiene las moléculas en formato SMILES.
    :param columna_smiles: Nombre de la columna que contiene las moléculas en formato SMILES.
    :return: DataFrame con una nueva columna 'Numero_carboxilos' que contiene el número de grupos carboxilo.
    """
    if columna_smiles not in df.columns:
        raise ValueError(f"La columna '{columna_smiles}' no se encuentra en el DataFrame.")

    def contar_carboxilos(smile):
        """Cuenta los grupos carboxilo en una molécula dada en formato SMILES."""
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return None
        # SMARTS para el grupo carboxilo (-COOH)
        carboxilo_smarts = Chem.MolFromSmarts('C(=O)O')
        return len(mol.GetSubstructMatches(carboxilo_smarts))

    # Crear una nueva columna en el DataFrame con el conteo de carboxilos
    df['Numero_carboxilos'] = df[columna_smiles].apply(contar_carboxilos)
    return df

