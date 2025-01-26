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

"""
CALCULO DE DESCRIPTORES MOLECULARES
"""

from modules.procesamiento.calculo_descriptores_moleculares import convertir_smiles_a_smarts

df = leer_csv('data/smiles_comptox_smiles_pubchem_iupac_name_smiles_name.csv')
output_path = 'data/SMARTS.csv'

if not os.path.exists(output_path):
    df = convertir_smiles_a_smarts(df)
    guardar_csv(df, 'data/SMARTS.csv')

    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
VERIFICACIÓN DE CALCULO DE TODOS LOS SMARTS
"""

df = leer_csv('data/Analisis_univariado_dataset.csv')

#print(df['SMARTS'].isnull().value_counts())

"""
CALCULO DE LOS DESCRIPTORES MOLECULARES
"""
#Calculo del LogP

from modules.procesamiento.calculo_descriptores_moleculares import calcular_logp_crippen

# Aplicar la función a la columna "SMILES" y crear una nueva columna "LogP"
df["LogP"] = df["SMILES"].apply(calcular_logp_crippen)

df_estadistico = df

# Mostrar el DataFrame resultante
print(df["LogP"].isnull().value_counts())

"""
SE REALIZA EL ANÁLISIS UNIVARIADO DE LA VARIABLE
"""
logP_univariado = df['LogP'].describe()
print(logP_univariado)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['LogP'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
logp_min = fila_minima['LogP']

print(f"El valor más bajo de LogP es {logp_min}, correspondiente a la sustancia: {smiles_min}")

"""
Encuentra a qué valor corresponde el valor máximo de LogP
"""

# Encuentra la fila con el valor más alto de LogP
fila_maxima = df.loc[df['LogP'].idxmax()]

# Extrae el SMILES y el valor más alto de LogP
smiles_max = fila_maxima['INPUT']
logp_max = fila_maxima['LogP']

print(f"El valor más alto de LogP es {logp_max}, correspondiente a la sustancia: {smiles_max}")

"""
Graficamos esta variable
"""

from modules.procesamiento.graficas import graficar_y_guardar_variable_continua

output_path = 'data/graficas/univariado_logP.png'

if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'LogP', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
DETECTAR LOS OUTLIERS
"""
df = df_estadistico

from modules.procesamiento.analisis_estadistico import detectar_outliers

outliers = detectar_outliers(df, 'LogP')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
ANALISIS DE LOGP Y CLASIFICACIÓN ATS
"""

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['LogP'].describe()
#guardar_csv(estadisticas, 'data/estadisticas.csv')
print(estadisticas)

"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de LogP entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_LogP_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'LogP', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""

from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_LogP_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'LogP', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Grafico de histograma
"""
from modules.procesamiento.graficas import graficar_y_guardar_histograma

output_path = 'data/graficas/histogramaplot_LogP_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'LogP', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
CALCULO DE PESO MOLECULAR
"""
from modules.procesamiento.calculo_descriptores_moleculares import calcular_pesos_moleculares
# Calcular los pesos moleculares y actualizar el DataFrame
df = calcular_pesos_moleculares(df, 'SMILES')

"""
Análisis descriptivo
"""
describe = df['Peso_Molecular'].describe()
print(describe)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['Peso_Molecular'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
Peso_Molecular_min = fila_minima['Peso_Molecular']

print(f"El valor más bajo de Peso_Molecular es {Peso_Molecular_min}, correspondiente a la sustancia: {smiles_min}")

"""
Encuentra a qué valor corresponde el valor máximo de Peso_Molecular
"""

# Encuentra la fila con el valor más alto de LogP
fila_maxima = df.loc[df['Peso_Molecular'].idxmax()]

# Extrae el SMILES y el valor más alto de LogP
smiles_max = fila_maxima['INPUT']
Peso_Molecular_max = fila_maxima['Peso_Molecular']

print(f"El valor más alto de Peso_Molecular es {Peso_Molecular_max}, correspondiente a la sustancia: {smiles_max}")

"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_peso_molecular.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'Peso_Molecular', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
DETECTANDO LOS OUTLIERS
"""

outliers = detectar_outliers(df, 'Peso_Molecular')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_Peso_Molecular.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
ANALISIS DE PESO MOLECULAR Y CLASIFICACIÓN ATS
"""

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Peso_Molecular'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Peso_Molecular.csv')
print(estadisticas)

"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de LogP entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_peso_molecular_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'Peso_Molecular', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""

from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_Peso_Molecular_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'Peso_Molecular', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Grafico de histograma
"""
output_path = 'data/graficas/histogramaplot_Peso_Molecular_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'Peso_Molecular', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
CALCULO DE NUMERO DE ACEPTORES DE HIDRÍGENO
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_numero_aceptores

calcular_numero_aceptores(df, 'SMILES')
"""
Análisis descriptivo
"""
describe = df['NumHAcceptors'].describe()
print(describe)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['NumHAcceptors'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
logp_min = fila_minima['NumHAcceptors']

#guardar_csv(df, 'data/NumHAcceptors.csv')

print(f"El valor más bajo de NumHAcceptors es {logp_min}, correspondiente a la sustancia: {smiles_min}")

"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_NumHAcceptors.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'NumHAcceptors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


"""
DETECTANDO LOS OUTLIERS
"""
outliers = detectar_outliers(df, 'NumHAcceptors')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_NumHAcceptors.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/NumHAcceptors_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'NumHAcceptors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['NumHAcceptors'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_NumHAcceptors.csv')
print(estadisticas)

"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de NumHAcceptors entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_NumHAcceptors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'NumHAcceptors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""
from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_NumHAcceptors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'NumHAcceptors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Grafico de histograma
"""
output_path = 'data/graficas/histogramaplot_NumHAcceptors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'NumHAcceptors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
calculo donadores de hidrógeno
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_numero_donadores

calcular_numero_donadores(df, 'SMILES')
#guardar_csv(df, 'data/donors.csv')

"""
Análisis descriptivo
"""
describe = df['NumHDonors'].describe()
print(describe)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['NumHDonors'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
logp_min = fila_minima['NumHDonors']

#guardar_csv(df, 'data/NumHAcceptors.csv')

print(f"El valor más bajo de NumHDonors es {logp_min}, correspondiente a la sustancia: {smiles_min}")


"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_NumHDonors.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'NumHDonors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/NumHDonors_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'NumHDonors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


"""
DETECTANDO LOS OUTLIERS
"""
outliers = detectar_outliers(df, 'NumHDonors')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_NumHDonors.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['NumHDonors'].describe()
guardar_csv(estadisticas, 'data/estadisticas_NumHDonors.csv')
print(estadisticas)


"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de NumHDonors entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_NumHDonors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'NumHDonors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""
from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_NumHDonors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'NumHDonors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""  
Grafico de histograma
"""
output_path = 'data/graficas/histogramaplot_NumHDonors_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'NumHDonors', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
CALCULO DE ENLACES DOBLES
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_enlaces_dobles

calcular_enlaces_dobles(df, 'SMILES')
print(df)

"""
Análisis descriptivo
"""
describe = df['Dobles_Enlaces'].describe()
print(describe)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['Dobles_Enlaces'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
Dobles_Enlaces = fila_minima['Dobles_Enlaces']

guardar_csv(df, 'data/Dobles_Enlaces.csv')

print(f"El valor más bajo de Dobles_Enlaces es {Dobles_Enlaces}, correspondiente a la sustancia: {smiles_min}")


"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_Dobles_Enlaces.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'Dobles_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/Dobles_Enlaces_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'Dobles_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Dobles_Enlaces'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Dobles_Enlaces.csv')
print(estadisticas)


"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de NumHDonors entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_Dobles_Enlaces_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'Dobles_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""
from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_Dobles_Enlaces_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'Dobles_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""  
Grafico de histograma
"""
output_path = 'data/graficas/histogramaplot_Dobles_Enlaces_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'Dobles_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
NUMERO DE ENLACES TRIPLES
"""


from modules.procesamiento.calculo_descriptores_moleculares import calcular_enlaces_triples

calcular_enlaces_triples(df, 'SMILES')
print(df)

"""
Análisis descriptivo
"""
describe = df['Triples_Enlaces'].describe()
print(describe)

"""
Encontrando a qué SMILES corresponde el valor minimo
"""
fila_minima = df.loc[df['Triples_Enlaces'].idxmin()]

# Extrae el SMILES y el valor más bajo de LogP
smiles_min = fila_minima['INPUT']
Triples_Enlaces = fila_minima['Triples_Enlaces']

#guardar_csv(df, 'data/Triples_Enlaces.csv')

print(f"El valor más bajo de Triples_Enlaces es {Triples_Enlaces}, correspondiente a la sustancia: {smiles_min}")


"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_triples_Enlaces.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'Triples_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/Triples_Enlaces_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'Triples_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Triples_Enlaces'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Triples_Enlaces.csv')
print(estadisticas)


"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de Triples_Enlaces entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_Triples_Enlaces_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'Triples_Enlaces', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
HIDROXILOS
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_Numero_hidroxilos

calcular_Numero_hidroxilos(df, 'SMILES')
print(df)

"""
Análisis descriptivo
"""
describe = df['Numero_hidroxilos'].describe()
print(describe)
#guardar_csv(df, 'data/Numero_hidroxilos.csv')

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/Numero_hidroxilos_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'Numero_hidroxilos', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_Numero_hidroxilos.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'Numero_hidroxilos', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Numero_hidroxilos'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Numero_hidroxilos.csv')
print(estadisticas)

"""
numero de carboxilos
"""
from modules.procesamiento.calculo_descriptores_moleculares import calcular_Numero_carboxilos
calcular_Numero_carboxilos(df, 'SMILES')

"""
Análisis descriptivo
"""
describe = df['Numero_carboxilos'].describe()
print(describe)
#guardar_csv(df, 'data/Numero_carboxilos.csv')

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/Numero_carboxilos_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'Numero_carboxilos', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_Numero_carboxilos.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'Numero_carboxilos', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Numero_carboxilos'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Numero_carboxilos.csv')
print(estadisticas)


"""Calcula el fingerprint de MACCS para una molécula en formato SMILES."""
from modules.procesamiento.calculo_fingerprints import calcular_fingerprints_maccs

calcular_fingerprints_maccs(df, 'SMILES')
#guardar_csv(df, 'data/MACCS_KEYS.csv')
#print(df)

"""
Convertir MACCS a numpy 
"""

from modules.procesamiento.calculo_fingerprints import convertir_MACCS_a_numpy

df = convertir_MACCS_a_numpy(df, 'MACCS')
#guardar_csv(df, 'data/MACCS_KEYS.csv')
#print(df)

"""
CALCULO DE MORGAN FINGERPRINTS
"""
from modules.procesamiento.calculo_fingerprints import calcular_ecfp

calcular_ecfp(df, 'SMILES')
#guardar_csv(df, 'data/morgan_fingerprints.csv')
print(df)

"""
Convertir ECFP a numpy 
"""
from modules.procesamiento.calculo_fingerprints import convertir_ECFP_a_numpy

df = convertir_ECFP_a_numpy(df, 'ECFP')
#guardar_csv(df, 'data/ECFP_array.csv')
#print(df['ECFP'][0])

"""
visualizar una molecula 
"""
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Draw



from rdkit.Chem import Draw

from rdkit.Chem import Draw
from rdkit import Chem
mol = AllChem.MolFromSmiles(df['SMILES'][0])
#fingerprint = df['ECFP'][0]
# Generar bitInfo para almacenar la información de los bits activados
bit_info = {}
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, bitInfo=bit_info)
list_bits = [(mol, bit, bit_info) for bit in fingerprint.GetOnBits()]
legends = [str(bit) for bit in fingerprint.GetOnBits()]
#img = Draw.DrawMorganBits(list_bits, molsPerRow=4, legends=legends)
#img.show()

img = Draw.DrawMorganBits(list_bits, molsPerRow=4,legends=legends)

"""
CODIGO PARA VISUALIZAR LOS FINGERPRINTS
"""
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM
from PIL import Image

# Suponiendo que 'img' es el objeto SVG que generaste
svg_file_path = 'output.svg'  # La ruta donde guardas el SVG
png_file_path = 'output.png'  # La ruta donde guardas el PNG

# Guarda la imagen SVG
#with open(svg_file_path, 'w') as f:
#    f.write(str(img))

# Cargar el archivo SVG y convertirlo a un gráfico que pueda manejar ReportLab
#drawing = svg2rlg(svg_file_path)

from modules.procesamiento.calculo_descriptores_moleculares import calcular_tpsa

# Calcular TPSA
df = calcular_tpsa(df, 'SMILES')

"""
Análisis descriptivo
"""
describe = df['TPSA'].describe()
print(describe)
#guardar_csv(df, 'data/TPSA.csv')


"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_TPSA.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'TPSA', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['TPSA'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_TPSA.csv')
print(estadisticas)

"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de NumHDonors entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_TPSA.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'TPSA', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""
from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_TPSA.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'TPSA', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Grafico de histograma
"""
from modules.procesamiento.graficas import graficar_y_guardar_histograma

output_path = 'data/graficas/histogramaplot_TPSA_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'TPSA', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Calculo de enlaces rotables
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_enlaces_rotables

calcular_enlaces_rotables(df, 'SMILES')

"""
Análisis descriptivo
"""
describe = df['NumRotatableBonds'].describe()
print(describe)
guardar_csv(df, 'data/NumRotatableBonds.csv')

"""
grafico de barras para una variable discreta
"""
from modules.procesamiento.graficas import graficar_y_guardar_barras

output_path = 'data/graficas/NumRotatableBonds_bar.png'
if not os.path.exists(output_path):
    graficar_y_guardar_barras(df,'NumRotatableBonds', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Graficamos esta variable
"""

output_path = 'data/graficas/univariado_NumRotatableBonds.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df, 'NumRotatableBonds', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['NumRotatableBonds'].describe()
guardar_csv(estadisticas, 'data/estadisticas_NumRotatableBonds.csv')
print(estadisticas)

"""
Boxplot: Una gráfica de cajas te permite comparar visualmente la distribución de NumHDonors entre los dos grupos.
"""
import seaborn as sns
import matplotlib.pyplot as plt

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_NumRotatableBonds.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'NumRotatableBonds', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Gráfica de violín
"""
from modules.procesamiento.graficas import graficar_y_guardar_violinplot
output_path = 'data/graficas/violinplot_NumRotatableBonds.png'
if not os.path.exists(output_path):
    graficar_y_guardar_violinplot(df, 'Clasificacion_ATS', 'NumRotatableBonds', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Grafico de histograma
"""
from modules.procesamiento.graficas import graficar_y_guardar_histograma

output_path = 'data/graficas/histogramaplot_NumRotatableBonds_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_histograma(df, 'Clasificacion_ATS', 'NumRotatableBonds', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


