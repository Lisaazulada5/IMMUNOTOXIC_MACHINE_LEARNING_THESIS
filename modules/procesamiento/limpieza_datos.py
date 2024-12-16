import pandas as pd


def verificar_conflictos(df):
    """
    Función para verificar si existen datos con valores 'active' e 'inactive'
    en la columna 'HIT CALL' para el mismo 'DTXSID'.

    Parameters:
    df (DataFrame): DataFrame con los datos a analizar.

    Returns:
    DataFrame: Filas que tienen valores 'active' e 'inactive' en la columna 'HIT CALL'.
    """
    # Filtrar conflictos de HIT CALL
    conflictos = df[df.duplicated(subset=['DTXSID'], keep=False) &
                    df['HIT CALL'].isin(['active', 'inactive'])]

    if not conflictos.empty:
        print("Se encontraron conflictos:")
        print(conflictos)
    else:
        print("No se encontraron conflictos entre 'active' e 'inactive'.")

    return conflictos

def limpiar_dtxsid(df, columna):
    """
    Limpia la columna DTXSID para que tenga el formato correcto, eliminando URLs.
    """
    # Extrae únicamente el DTXSID usando una expresión regular
    df[columna] = df[columna].str.extract(r'(DTXSID\d+)', expand=False)
    return df