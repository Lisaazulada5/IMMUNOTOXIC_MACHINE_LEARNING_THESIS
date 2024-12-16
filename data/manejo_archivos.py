import os
import pandas as pd

import pandas as pd
import os


def leer_csv(ruta, delimitador=';', encoding='utf-8'):
    """
    Lee un archivo CSV con un delimitador específico y maneja inconsistencias en el archivo.
    Retorna un DataFrame.
    """
    try:
        # Intenta leer el archivo directamente con el delimitador proporcionado
        return pd.read_csv(ruta, sep=delimitador, encoding=encoding, error_bad_lines=False, warn_bad_lines=True)
    except pd.errors.ParserError:
        print(f"ParserError al leer {ruta} con delimitador '{delimitador}'. Intentando corregir...")

        # Si hay errores, intenta detectar automáticamente el delimitador
        try:
            import csv
            with open(ruta, 'r', encoding=encoding) as file:
                dialect = csv.Sniffer().sniff(file.read(2048))
                file.seek(0)  # Reinicia el puntero del archivo
                print(f"Delimitador detectado automáticamente: '{dialect.delimiter}'")
                return pd.read_csv(ruta, sep=dialect.delimiter, encoding=encoding, error_bad_lines=False)
        except Exception as e:
            print(f"No se pudo detectar automáticamente el delimitador: {e}")

    except UnicodeDecodeError:
        print(f"Error de codificación al leer {ruta}. Probando con 'ISO-8859-1'.")
        try:
            return pd.read_csv(ruta, sep=delimitador, encoding='ISO-8859-1', error_bad_lines=False)
        except Exception as e:
            print(f"No se pudo leer el archivo con ISO-8859-1: {e}")

    print(f"El archivo {ruta} no pudo ser leído correctamente.")
    return pd.DataFrame()  # Retorna un DataFrame vacío si falla

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


