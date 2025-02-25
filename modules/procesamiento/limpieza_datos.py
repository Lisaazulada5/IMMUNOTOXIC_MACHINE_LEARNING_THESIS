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
                    df['HIT CALL'].isin(['Active', 'Inactive'])]

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

from data.manejo_archivos import merge_datasets
def agregar_columna_ats(df):
    """
    Calcula el ATS (Aggregate Toxicity Score) y lo agrega como una columna al DataFrame.
    """
    DTXSID_ACTIVE = pd.DataFrame(df[df['HIT CALL'] == 'Active'].groupby('DTXSID').size())
    print(DTXSID_ACTIVE)

    DTXSID_TOTAL = pd.DataFrame(df.groupby('DTXSID').size())
    print(DTXSID_TOTAL)

    ATS = DTXSID_ACTIVE/DTXSID_TOTAL
    ATS = ATS.fillna(0)
    ATS = pd.DataFrame(ATS).reset_index()
    print(ATS)


    merge_datasets(df, ATS, columna_clave='DTXSID', sep=';', archivo_salida=output_path)

from data.manejo_archivos import guardar_csv


def clasificar_ats(df, columna_ats='ATS'):
    """
    Clasifica los valores de ATS en 'Activo' o 'Inactivo' y agrega una columna nueva 'Clasificación_ATS'.

    Args:
        df (pd.DataFrame): El DataFrame que contiene la columna ATS.
        columna_ats (str): Nombre de la columna que contiene los valores ATS.

    Returns:
        pd.DataFrame: DataFrame con la columna 'Clasificación_ATS' agregada.
    """
    if columna_ats not in df.columns:
        raise ValueError(f"La columna '{columna_ats}' no existe en el DataFrame.")

    # Crear la nueva columna con base en la condición
    df['Clasificacion_ATS'] = df[columna_ats].apply(lambda x: 'Activo' if x >= 0.10 else 'Inactivo')
    return df















