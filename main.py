import pandas as pd
from modules.procesamiento.validar_smiles import obtener_iupac_desde_pubchem
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.obtener_iupac_desde_smiles_rdkit import obtener_iupac_desde_smiles
import os
import pandas as pd
from modules.procesamiento.validar_smiles import obtener_iupac_desde_pubchem
from data.manejo_archivos import leer_csv, guardar_csv

# Rutas de archivos
input_path = 'data/smiles_all_concatened.csv'
output_path = 'data/smiles_all_concatened_iupac_pubchem.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    # Cargar el CSV con nombres comunes
    df = leer_csv(input_path)

    # Supongamos que la columna de nombres comunes se llama 'PREFERRED_NAME'
    df['iupac_name'] = df['PREFERRED_NAME'].apply(obtener_iupac_desde_pubchem)

    # Guardar resultados acumulativos
    guardar_csv(df, output_path, append=False)  # Cambié append a False para evitar agregar más datos

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

import os
import pandas as pd

# Ruta de los archivos
input_path = 'data/smiles_all_concatened_iupac_pubchem.csv'
output_path = 'data/smiles_all_concatened_iupac_smiles.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    # Cargar el CSV con SMILES
    df = leer_csv(input_path)

    # Suponiendo que la columna de SMILES se llama 'SMILES'
    df['iupac_name_SMILES'] = df['SMILES'].apply(obtener_iupac_desde_smiles)

    # Revisar el DataFrame con los nombres IUPAC añadidos
    print(df)

    # Guardar los resultados acumulativos
    guardar_csv(df, output_path, append=False)  # Cambié append a False para no agregar datos si el archivo no existe

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Concatenar los DF de los ensayos

from data.manejo_archivos import concatenar_archivos_en_uno

folder_path = 'data/HDF3CGF'
output_path = 'data/HDF3CGF_concatenados.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    ## Usar la función
    concatenar_archivos_en_uno(folder_path, output_path)

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

folder_path = 'data/KF3CT'
output_path = 'data/KF3CT_concatenados.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    ## Usar la función
    concatenar_archivos_en_uno(folder_path, output_path)

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

folder_path = 'data/Concatenados_KF3CT_HDF3CGF'
output_path = 'data/KF3CT_HDF3CGF_concatenados.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    ## Usar la función
    concatenar_archivos_en_uno(folder_path, output_path)

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

df = leer_csv('data/KF3CT_HDF3CGF_concatenados.csv')
# print(df)

from modules.procesamiento.limpieza_datos import limpiar_dtxsid

a = limpiar_dtxsid(df, 'DTXSID')

# print(a)

from data.manejo_archivos import merge_datasets

df_smiles = leer_csv('data/smiles_all_concatened_iupac_smiles.csv')
df_kf3ct = a.iloc[:, :9]

# se verifican si hay datos que sean activos e inactivos a la vez
from modules.procesamiento.limpieza_datos import verificar_conflictos

verificar_conflictos(
    df_kf3ct)  # se evidencian que todas las sustancias pueden dar un hit call tanto activo, como inactivo.

"""
Se adiciona la columna ATS para la clasificación del potencial inmunotóxico
"""

from modules.procesamiento.limpieza_datos import agregar_columna_ats

output_path = 'data/ATS.csv'
if not os.path.exists(output_path):
    agregar_columna_ats(df_kf3ct)  # crea una columna ATS
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Se realiza la nueva clasificación de activo e inactivo con base en el ATS
"""
df = leer_csv('data/ATS.csv')
print(df)
from modules.procesamiento.limpieza_datos import clasificar_ats

df = clasificar_ats(df)
output_path = 'data/ATS_CLASIFICACION.csv'
if not os.path.exists(output_path):
    guardar_csv(df, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Se urealiza merge para que queden los SMILES REVISADOS
"""
a = 'data/merge_ATS_SMILES.csv'
if not os.path.exists(output_path):
    df2 = leer_csv('data/ATS_CLASIFICACION.csv')
    df1 = leer_csv('data/smiles_all_concatened_iupac_smiles.csv')
    merge_datasets(df1, df2, columna_clave='DTXSID', sep=';', archivo_salida=a)
    print(f"Archivo generado: {a}")
else:
    print(f"El archivo {a} ya existe. No se ha procesado de nuevo.")

"""
Ahora del archivo resultante se generan dos dataframe, uno con SMILES y otro con las sustancias que no tienen SMILES
"""

# Leer el archivo CSV

df = leer_csv('data/merge_ATS_SMILES.csv')
# Dividir los DataFrames según la columna 'SMILES'
df_sin_smiles = df[df['SMILES'].isna()]
df_con_smiles = df[df['SMILES'].notna()]

# Verificar los tamaños de los DataFrames
print(f"Filas sin SMILES: {df_sin_smiles.shape[0]}")
print(f"Filas con SMILES: {df_con_smiles.shape[0]}")

# Guardar los DataFrames resultantes (opcional)
ruta1 = 'data/sin_smiles.csv'
ruta2 = 'data/con_smiles.csv'
if not os.path.exists(ruta1):
    guardar_csv(df_sin_smiles, ruta1)
    guardar_csv(df_con_smiles, ruta2)
    print(f"Archivo generado: {ruta1}")
else:
    print(f"El archivo {ruta1} ya existe. No se ha procesado de nuevo.")

"""
Revisar el archivo sin smiles a ver si puedo obtener dichos SMILES
"""

from modules.procesamiento.validar_smiles import obtener_smiles_pubchem

df = leer_csv(ruta1)
a = 'data/con_smiles/smiles_pubchem.csv'
if not os.path.exists(a):
    df = obtener_smiles_pubchem(df, columna_dtxsid='DTXSID')
    guardar_csv(df, a)
    print(f"Archivo generado: {a}")
else:
    print(f"El archivo {a} ya existe. No se ha procesado de nuevo.")

"""
Unir los smiles encontrados con los demás datos
"""

df = leer_csv('data/con_smiles/smiles_pubchem.csv')
# print(df[df['SMILES'] != '0'])
df_no_vacios = df[df['SMILES'] != '0']
print(df_no_vacios)
a = 'data/df_no_vacios.csv'
if not os.path.exists(a):
    guardar_csv(df_no_vacios, a)
    print(f"Archivo generado: {a}")
else:
    print(f"El archivo {a} ya existe. No se ha procesado de nuevo.")

"""
Concatenar los SMILES descargados de coptox y los SMILES encontrados en pubchem
"""

"""
def convertir_texto_a_csv(input_file, output_file, delimiter="\t", encoding="utf-8"):

    Convierte un archivo de texto a un archivo CSV.

    Parámetros:
    input_file: str - Ruta del archivo de texto de entrada.
    output_file: str - Ruta donde se guardará el archivo CSV.
    delimiter: str - Delimitador utilizado en el archivo de texto (default: tabulación).
    encoding: str - Codificación del archivo de texto (default: utf-8).

    Retorna:
    None - Guarda el archivo convertido como CSV.
    """
"""
    try:
        # Leer el archivo de texto en un DataFrame
        df = pd.read_csv(input_file, delimiter=delimiter, encoding=encoding)

        # Guardar el DataFrame en formato CSV
        df.to_csv(output_file, index=False, encoding=encoding)

        print(f"Archivo convertido y guardado como: {output_file}")
    except Exception as e:
        print(f"Error al convertir el archivo: {e}")


# Ejemplo de uso
input_file = "data/smiles_pubchem"  # Ruta del archivo de texto
output_file = "data/archivo.csv"  # Ruta del archivo CSV de salida
convertir_texto_a_csv(input_file, output_file, delimiter="\t")
"""
folder_path = 'data/con_smiles/concatenar_smiles_pubchem'
output_path = 'data/smiles_comptox_smiles_pubchem.csv'
# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    ## Usar la función
    concatenar_archivos_en_uno(folder_path, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Voy a hacer la validacion de los nombre IUPAC para el nuevo conjunto de datos
"""
df = leer_csv('data/smiles_comptox_smiles_pubchem.csv')
# Filtrar filas donde iupac_name e iupac_name_SMILES están vacías
df_filas_vacias = df[(df['iupac_name'].isna() | (df['iupac_name'] == '')) &
                     (df['iupac_name_SMILES'].isna() | (df['iupac_name_SMILES'] == ''))]
"""
# Aplicar las funciones de forma iterativa para llenar los datos faltantes
for index, row in df_filas_vacias.iterrows():
    DTXSID = row['DTXSID']
    if DTXSID:  # Solo proceder si hay SMILES
        iupac_name = obtener_iupac_desde_pubchem(DTXSID)  # Usar tu función para buscar en PubChem
# Actualizar el dataframe original con los valores encontrados
df = df.loc[index, 'iupac_name'] = iupac_name
guardar_csv(df, 'data/smiles_comptox_smiles_pubchem_iupac_name.csv')
"""

import pandas as pd

# Diccionario para almacenar los resultados ya obtenidos
iupac_cache = {}

# Función para guardar el DataFrame en un archivo CSV

# Iterar sobre las filas del DataFrame
input_path = 'data/smiles_all_concatened.csv'
output_path = 'data/smiles_comptox_smiles_pubchem_iupac_name.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    for index, row in df_filas_vacias.iterrows():
        DTXSID = row['DTXSID']
        if DTXSID:  # Solo proceder si hay un valor en DTXSID
            if DTXSID not in iupac_cache:
                # Si no está en el caché, buscar y almacenar el resultado
                iupac_name = obtener_iupac_desde_pubchem(DTXSID)
                iupac_cache[DTXSID] = iupac_name
            else:
                # Si ya está en el caché, reutilizar el resultado
                iupac_name = iupac_cache[DTXSID]
            # Actualizar el DataFrame original con el valor encontrado
            df.loc[index, 'iupac_name'] = iupac_name

    # Guardar el DataFrame actualizado en un archivo CSV
    guardar_csv(df, 'data/smiles_comptox_smiles_pubchem_iupac_name.csv')

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""Ahora se revisan si los nombres IUPAC generados con SMILES"""

df = leer_csv('data/smiles_comptox_smiles_pubchem_iupac_name.csv')
# Filtrar filas donde iupac_name_SMILES e iupac_name_SMILES están vacías
df_filas_vacias = df[(df['iupac_name_SMILES'].isna() | (df['iupac_name_SMILES'] == ''))]

input_path = 'data/smiles_comptox_smiles_pubchem_iupac_name.csv'
output_path = 'data/smiles_comptox_smiles_pubchem_iupac_name_smiles_name.csv'

# Verificar si el archivo de salida ya existe
if not os.path.exists(output_path):
    for index, row in df_filas_vacias.iterrows():
        SMILES = row['SMILES']
        if SMILES:  # Solo proceder si hay un valor en DTXSID
            if SMILES not in iupac_cache:
                # Si no está en el caché, buscar y almacenar el resultado
                iupac_name_SMILES = obtener_iupac_desde_smiles(SMILES)
                iupac_cache[SMILES] = iupac_name_SMILES
            else:
                # Si ya está en el caché, reutilizar el resultado
                iupac_name_SMILES = iupac_cache[SMILES]
            # Actualizar el DataFrame original con el valor encontrado
            df.loc[index, 'iupac_name_SMILES'] = iupac_name_SMILES

    # Guardar el DataFrame actualizado en un archivo CSV
    guardar_csv(df, 'data/smiles_comptox_smiles_pubchem_iupac_name_smiles_name.csv')

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")




















