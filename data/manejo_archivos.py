import os
import pandas as pd

import pandas as pd
import os
import pandas as pd

def leer_csv(ruta, sep=';', encoding='utf-8'):
    delimitadores_posibles = [';', ',', '\t']
    for sep in delimitadores_posibles:
        try:
            return pd.read_csv(ruta, sep=sep, encoding=encoding)
        except pd.errors.ParserError:
            continue
        except UnicodeDecodeError:
            return pd.read_csv(ruta, sep=sep, encoding='ISO-8859-1')
    return pd.DataFrame()

def guardar_csv(df, ruta, append=False, sep=';'):
    """
    Guarda un DataFrame en un archivo CSV.
    Si `append` es True, agrega las filas sin sobrescribir el archivo.
    """
    modo = 'a' if append else 'w'
    encabezado = not append or not os.path.exists(ruta)
    df.to_csv(ruta, mode=modo, header=encabezado, index=False, sep=sep)

def concatenar_archivos_en_uno(folder_path, output_path, append=False):
    """
    Función para concatenar múltiples archivos CSV y Excel en un solo archivo.
    """
    import os
    import pandas as pd

    # Verifica si la carpeta de salida existe
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Lista de todos los archivos CSV o Excel en la carpeta
    files = [f for f in os.listdir(folder_path) if f.endswith('.csv') or f.endswith('.xlsx') or f.endswith('.xls')]
    print(f"Archivos detectados en la carpeta {folder_path}: {files}")

    dfs = []

    # Leer cada archivo y agregarlo a la lista de DataFrames
    for file in files:
        file_path = os.path.join(folder_path, file)
        try:
            df = leer_csv(file_path)  # Reutiliza la función leer_csv
            print(f"Archivo leído correctamente: {file}")
            print(df.head())  # Muestra las primeras filas como validación
            dfs.append(df)
        except Exception as e:
            print(f"Error al leer el archivo {file}: {e}")

    if dfs:
        # Concatenar todos los DataFrames en uno solo
        df_combined = pd.concat(dfs, ignore_index=True)

        # Guardar el DataFrame combinado
        guardar_csv(df_combined, output_path, append=append)
        print(f"Archivo combinado guardado en: {output_path}")
    else:
        print("No se han leído archivos válidos. Verifica los formatos y delimitadores.")
"""
JOIN
"""
# manejo_archivos.py

import pandas as pd


def merge_datasets(df_smiles, df_kf3ct, columna_clave='DTXSID', sep=';', archivo_salida='data/merged_file.csv'):
    """
    Realiza un merge entre dos DataFrames, utilizando una columna clave y asegurando que el tamaño del DataFrame resultante
    sea el de df_smiles.

    Parámetros:
    df_smiles: DataFrame - El DataFrame con los SMILES.
    df_kf3ct: DataFrame - El DataFrame con los datos adicionales a añadir.
    columna_clave: str - El nombre de la columna en común para hacer el merge (default es 'DTXSID').
    sep: str - El delimitador para los archivos CSV (default es ';').
    archivo_salida: str - Ruta del archivo donde se guardará el resultado final.

    Retorna:
    DataFrame - El DataFrame resultante del merge.
    """
    try:
        # Asegurarse de que la columna clave sea de tipo string
        df_smiles[columna_clave] = df_smiles[columna_clave].astype(str)
        df_kf3ct[columna_clave] = df_kf3ct[columna_clave].astype(str)

        # Realizar el merge con un left join para mantener el tamaño de df_smiles
        df_merged = pd.merge(df_smiles, df_kf3ct, on=columna_clave, how='right')

        # Guardar el DataFrame resultante en un archivo CSV
        df_merged.to_csv(archivo_salida, index=False, sep=sep)

        # Imprimir tamaño del DataFrame final para verificar
        print(f"Merge completado. Tamaño del DataFrame resultante: {df_merged.shape}")

        return df_merged

    except Exception as e:
        print(f"Error al realizar el merge: {e}")
        return None


"""
la cague
"""
def separar_por_estado(df, columna_estado):
    """
    Separa el DataFrame en dos DataFrames: uno con 'activo' y otro con 'inactivo'.
    """
    activos = df[df[columna_estado] == 'active'].drop_duplicates()
    print(activos)
    inactivos = df[df[columna_estado] == 'inactive'].drop_duplicates()
    print(inactivos)
    return activos, inactivos


import pandas as pd


def dividir_por_smiles(df, columna_smiles='SMILES'):
    """
    Divide un DataFrame en dos según si la columna SMILES está vacía o no.

    Parámetros:
        df (pd.DataFrame): DataFrame de entrada.
        columna_smiles (str): Nombre de la columna a evaluar. Por defecto es 'SMILES'.

    Retorna:
        df_con_smiles (pd.DataFrame): DataFrame con filas donde la columna SMILES tiene datos.
        df_sin_smiles (pd.DataFrame): DataFrame con filas donde la columna SMILES está vacía.
    """
    # Filtrar filas donde la columna SMILES no está vacía
    df_con_smiles = df[df[columna_smiles].notna() & (df[columna_smiles] != '')]

    # Filtrar filas donde la columna SMILES está vacía
    df_sin_smiles = df[df[columna_smiles].isna() | (df[columna_smiles] == '')]

    guardar_csv(df_sin_smiles, 'data/df_sin_smiles.csv')
    guardar_csv(df_con_smiles, 'data/df_con_smiles.csv')

    return df_con_smiles, df_sin_smiles

