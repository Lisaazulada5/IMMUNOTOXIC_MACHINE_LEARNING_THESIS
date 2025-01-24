from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw



from rdkit.Chem import Draw

from rdkit.Chem import Draw
from rdkit import Chem

from rdkit.Chem import AllChem, Draw
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect


def visualize_ecfp(df, row_index):
    """
    Visualiza el fingerprint ECFP de una molécula seleccionada desde el DataFrame.

    Parámetros:
    - df: DataFrame con las columnas 'SMILES' y 'ECFP' (ExplicitBitVect).
    - row_index: Índice de la fila de la molécula que se desea visualizar.
    """
    try:
        # Extraer el SMILES y el fingerprint correspondiente
        smiles = df.loc[row_index, 'SMILES']
        fingerprint = df.loc[row_index, 'ECFP']

        # Convertir el SMILES a Mol
        mol = AllChem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"El SMILES en la fila {row_index} no se pudo convertir a Mol.")

        # Crear lista de bits activos
        list_bits = [(mol, bit, None) for bit in fingerprint.GetOnBits()]
        legends = [str(bit) for bit in fingerprint.GetOnBits()]

        # Dibujar los bits ECFP
        img = Draw.DrawMorganBits(list_bits, molsPerRow=4, legends=legends)
        img.show()

    except Exception as e:
        print(f"An error occurred: {e}")
