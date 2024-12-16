import os
import pandas as pd

import pandas as pd
import os
import pandas as pd

def leer_csv(ruta, sep=';'):
    """
    Lee un archivo CSV y lo retorna como un DataFrame.
    """
    try:
        return pd.read_csv(ruta, sep=sep)
    except FileNotFoundError:
        print(f"El archivo {ruta} no existe.")
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


