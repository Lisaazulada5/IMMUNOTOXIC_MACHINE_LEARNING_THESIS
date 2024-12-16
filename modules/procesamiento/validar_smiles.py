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
