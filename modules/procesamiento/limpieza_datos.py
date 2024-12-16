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

from data.manejo_archivos import leer_csv


def limpiar_dtxsid(df, columna):
    """
    Limpia los valores en la columna especificada, dejando solo el DTXSID de las URLs.
    """
    # Asegurarse de que la columna sea de tipo cadena
    df[columna] = df[columna].astype(str)

    # Reemplazar cualquier cosa antes de 'DTXSID' (incluido el mismo) por una cadena vacía
    df[columna] = df[columna].str.replace(r'.*?(DTXSID\d+)', r'\1', regex=True)

    # Reemplazar los valores nulos con 'Desconocido' si no se encuentra el DTXSID
    df[columna].fillna('Desconocido', inplace=True)

    # Revisamos si se extrajo correctamente el DTXSID
    print(f"Valores después de la limpieza: {df[columna].head()}")

    return df
