import time
import pubchempy as pcp

def obtener_iupac_desde_smiles(smiles):
    time.sleep(0.1)  # Pausa de 100ms entre solicitudes
    try:
        compuestos = pcp.get_compounds(smiles, 'smiles')
        if compuestos:
            return compuestos[0].iupac_name
        else:
            return 'No se encontró'
    except Exception as e:
        print(f"Error con SMILES {smiles}: {e}")
        return 'Error'

