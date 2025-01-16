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


