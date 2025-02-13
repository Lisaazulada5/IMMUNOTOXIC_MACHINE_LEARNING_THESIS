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

"""
TPSA
"""
from rdkit import Chem
from rdkit.Chem import MolSurf


def calcular_tpsa(df, smiles_col):
    """
    Calcula el TPSA para cada molécula en una columna de SMILES y guarda los resultados en una nueva columna.

    Parámetros:
        df (pandas.DataFrame): DataFrame que contiene la columna de SMILES.
        smiles_col (str): Nombre de la columna que contiene las cadenas SMILES.

    Retorna:
        pandas.DataFrame: DataFrame con una nueva columna "TPSA" que contiene los valores calculados.
    """
    # Crear una nueva columna para los valores de TPSA
    df['TPSA'] = df[smiles_col].apply(
        lambda smi: MolSurf.TPSA(Chem.MolFromSmiles(smi)) if Chem.MolFromSmiles(smi) else None)
    return df

"""
CALCULO DEL NÚMERO DE ENLACES ROTABLES 
"""

def calcular_enlaces_rotables(df, smiles_col):
    """
    Calcula el número de enlaces rotables para cada molécula en una columna de SMILES y guarda los resultados en una nueva columna.

    Parámetros:
        df (pandas.DataFrame): DataFrame que contiene la columna de SMILES.
        smiles_col (str): Nombre de la columna que contiene las cadenas SMILES.

    Retorna:
        pandas.DataFrame: DataFrame con una nueva columna "NumRotatableBonds" que contiene los valores calculados.
        Imprime un mensaje si encuentra SMILES inválidos.
    """
    enlaces_rotables = []
    smiles_invalidos = 0
    for smi in df[smiles_col]:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            enlaces_rotables.append(Descriptors.NumRotatableBonds(mol))
        else:
            enlaces_rotables.append(None)
            smiles_invalidos += 1
    if smiles_invalidos > 0:
        print(f"Se encontraron {smiles_invalidos} SMILES inválidos. Se han asignado valores None.")
    df['NumRotatableBonds'] = enlaces_rotables
    return df


from rdkit import Chem


def contar_sulfatos(smile):
    """
    Cuenta los grupos sulfato (-SO4) en una molécula dada en formato SMILES.

    Parameters:
        smile (str): Representación SMILES de la molécula.

    Returns:
        int or None: Número de grupos sulfato en la molécula, o None si el SMILES no es válido.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    # SMARTS para el grupo sulfato (-SO4)
    sulfato_smarts = Chem.MolFromSmarts('O=S(=O)([O])[O]')
    return len(mol.GetSubstructMatches(sulfato_smarts))


# Aplicar la función a una columna del DataFrame
def agregar_numero_sulfatos(df, columna_smiles):
    """
    Agrega una nueva columna al DataFrame con el conteo de grupos sulfato en las moléculas SMILES.

    Parameters:
        df (pandas.DataFrame): DataFrame que contiene una columna con SMILES.
        columna_smiles (str): Nombre de la columna con los SMILES.

    Returns:
        pandas.DataFrame: DataFrame con una nueva columna 'Numero_sulfatos' que contiene el conteo.
    """
    df['Numero_sulfatos'] = df[columna_smiles].apply(contar_sulfatos)
    return df


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def calcular_carga_molecular(smiles):
    """Calcula la carga formal total de una molécula a partir de su SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Manejo de errores si el SMILES es inválido
    return Chem.GetFormalCharge(mol)


def agregar_carga_molecular(df, smiles_col="SMILES"):
    """
    Agrega la carga formal total al DataFrame con base en una columna de SMILES.

    Parámetros:
    - df: DataFrame de pandas con una columna de SMILES.
    - smiles_col: Nombre de la columna que contiene los SMILES (por defecto, "SMILES").

    Retorna:
    - DataFrame con una nueva columna "Carga_Molecular".
    """
    df["Carga_Molecular"] = df[smiles_col].apply(calcular_carga_molecular)
    return df

from rdkit.Chem import AllChem

from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd


def calcular_carga_formal(smiles):
    """
    Calcula la carga formal total de una molécula a partir de su SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None  # Manejo de errores si el SMILES es inválido
    return Chem.GetFormalCharge(mol)


def calcular_carga_gasteiger(smiles):
    """
    Calcula la suma de las cargas parciales de Gasteiger de una molécula.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    AllChem.ComputeGasteigerCharges(mol)
    return sum(float(a.GetProp("_GasteigerCharge")) for a in mol.GetAtoms())


def agregar_cargas_moleculares(df, smiles_col="SMILES"):
    """
    Agrega las cargas formales y de Gasteiger al DataFrame.

    Parámetros:
    - df: DataFrame con una columna de SMILES.
    - smiles_col: Nombre de la columna que contiene los SMILES (por defecto, "SMILES").

    Retorna:
    - DataFrame con nuevas columnas "Carga_Formal" y "Carga_Gasteiger".
    """
    df["Carga_Formal"] = df[smiles_col].apply(calcular_carga_formal)
    df["Carga_Gasteiger"] = df[smiles_col].apply(calcular_carga_gasteiger)
    return df

from rdkit import Chem
from rdkit.Chem import GraphDescriptors

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd


def calcular_molmr(df, columna_smiles):
    """
    Calcula el descriptor MolMR (Wildman-Crippen MR value) para cada molécula en un DataFrame.
    :param df: DataFrame con una columna de SMILES.
    :param columna_smiles: Nombre de la columna que contiene los SMILES.
    :return: DataFrame con la nueva columna 'MolMR'.
    """
    valores_molmr = []

    for smiles in df[columna_smiles]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molmr = Descriptors.MolMR(mol)  # Cálculo del descriptor
        else:
            molmr = None  # Manejo de errores en caso de SMILES inválidos
        valores_molmr.append(molmr)

    df['MolMR'] = valores_molmr
    return df


from rdkit import Chem
from rdkit.Chem import Descriptors, rdPartialCharges
def calcular_maxparcialcharge(df, columna_smiles):
    """
    Calcula el descriptor MolMR (Wildman-Crippen MR value) para cada molécula en un DataFrame.
    :param df: DataFrame con una columna de SMILES.
    :param columna_smiles: Nombre de la columna que contiene los SMILES.
    :return: DataFrame con la nueva columna 'MolMR'.
    """
    valores_calcular_maxparcialcharge = []

    for smiles in df[columna_smiles]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molmr = Chem.Descriptors.MaxPartialCharge(mol)  # Cálculo del descriptor
        else:
            molmr = None  # Manejo de errores en caso de SMILES inválidos
        valores_calcular_maxparcialcharge.append(molmr)

    df['Max_Charge'] = valores_calcular_maxparcialcharge
    return df

def calcular_minparcialcharge(df, columna_smiles):
    """
    Calcula el descriptor MolMR (Wildman-Crippen MR value) para cada molécula en un DataFrame.
    :param df: DataFrame con una columna de SMILES.
    :param columna_smiles: Nombre de la columna que contiene los SMILES.
    :return: DataFrame con la nueva columna 'MolMR'.
    """
    valores_calcular_maxparcialcharge = []

    for smiles in df[columna_smiles]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            molmr = Chem.Descriptors.MinPartialCharge(mol)  # Cálculo del descriptor
        else:
            molmr = None  # Manejo de errores en caso de SMILES inválidos
        valores_calcular_maxparcialcharge.append(molmr)

    df['Min_Charge'] = valores_calcular_maxparcialcharge
    return df


from rdkit import Chem
from rdkit.Chem import rdchem
import numpy as np


def calcular_bcutw(smiles):
    """Calcula BCUTw.1 l y BCUTw.1 h para una molécula a partir de un SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None  # Retorna valores nulos si el SMILES no es válido

    num_atoms = mol.GetNumAtoms()

    # Crear la matriz de adyacencia
    adj_matrix = Chem.rdmolops.GetAdjacencyMatrix(mol)

    # Obtener los pesos atómicos
    pesos_atomicos = np.array(
        [rdchem.GetPeriodicTable().GetAtomicWeight(atom.GetAtomicNum()) for atom in mol.GetAtoms()])

    # Modificar la matriz con los pesos atómicos
    weighted_matrix = adj_matrix * np.sqrt(np.outer(pesos_atomicos, pesos_atomicos))

    # Calcular los eigenvalues
    eigenvalues = np.linalg.eigvalsh(weighted_matrix)

    # Tomar el más bajo (l) y el más alto (h)
    bcutw_l = min(eigenvalues)
    bcutw_h = max(eigenvalues)

    return bcutw_l, bcutw_h


from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
import numpy as np


def calcular_bcutc(smiles):
    """Calcula BCUTc.1 l y BCUTc.1 h para una molécula a partir de un SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None  # Retorna valores nulos si el SMILES no es válido

    # Calcular cargas parciales de Gasteiger
    ComputeGasteigerCharges(mol)

    # Obtener las cargas parciales de los átomos
    cargas_parciales = np.array([atom.GetDoubleProp('_GasteigerCharge') for atom in mol.GetAtoms()])

    # Crear la matriz de adyacencia
    adj_matrix = rdmolops.GetAdjacencyMatrix(mol)

    # Modificar la matriz con las cargas parciales
    weighted_matrix = adj_matrix * np.sqrt(np.outer(cargas_parciales, cargas_parciales))

    # Calcular los eigenvalues
    eigenvalues = np.linalg.eigvalsh(weighted_matrix)

    # Tomar el más bajo (l) y el más alto (h)
    bcutc_l = min(eigenvalues)
    bcutc_h = max(eigenvalues)

    return bcutc_l, bcutc_h

import pandas as pd
from mordred import Calculator, descriptors
from rdkit import Chem
from rdkit import Chem
from mordred import Calculator, BCUT

def calcular_bcut_smr_df(df, smiles_col):
    """
    Calcula el descriptor BCUT_SMR para cada molécula en un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con una columna de SMILES.
        smiles_col (str): Nombre de la columna con los SMILES.

    Returns:
        pd.DataFrame: DataFrame con una nueva columna "BCUT_SMR".
    """
    # Configurar el cálculo de BCUT usando SMR como propiedad atómica
    calc = Calculator(BCUT.BCUT(prop="e", nth=0))  # nth=0 para el mayor eigenvalor

    def calcular_bcut(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                descriptores = calc(mol)
                return descriptores.asdict().get("BCUT_SMR_0", None)  # Extrae el valor correcto
            except Exception as e:
                print(f"Error con {smiles}: {e}")
                return None
        return None

    # Aplicar la función a la columna de SMILES
    df["BCUT_SMR"] = df[smiles_col].apply(calcular_bcut)

    return df  # Retorna el DataFrame con la nueva columna

from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd

from rdkit import Chem
from mordred import Calculator, descriptors
import pandas as pd


def calcular_todos_los_descriptores(df, smiles_col):
    """
    Calcula todos los descriptores de Mordred y los añade al DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con una columna de SMILES.
        smiles_col (str): Nombre de la columna con los SMILES.

    Returns:
        pd.DataFrame: DataFrame con los descriptores calculados agregados.
    """
    calc = Calculator(descriptors, ignore_3D=True)  # Ignorar descriptores 3D
    nombres_descriptores = [str(d) for d in calc.descriptors]  # Obtener nombres de los descriptores

    df_descriptores = []

    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            try:
                valores = calc(mol)
                df_descriptores.append(list(valores))
            except Exception as e:
                print(f"Error al calcular descriptores para {smiles}: {e}")
                df_descriptores.append([None] * len(nombres_descriptores))
        else:
            df_descriptores.append([None] * len(nombres_descriptores))

    # Convertir los resultados a un DataFrame con nombres de columnas
    df_descriptores = pd.DataFrame(df_descriptores, columns=nombres_descriptores)

    # Concatenar el DataFrame original con los descriptores
    df_resultado = pd.concat([df, df_descriptores], axis=1)

    return df_resultado







