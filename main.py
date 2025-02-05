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

"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'Dobles_Enlaces')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_Dobles_Enlaces.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
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


"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'Triples_Enlaces')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_Triples_Enlaces.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
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

"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'Numero_hidroxilos')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_Numero_hidroxilos.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
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

"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'Numero_carboxilos')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_Numero_carboxilos.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['Numero_carboxilos'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_Numero_carboxilos.csv')
print(estadisticas)

from modules.procesamiento.graficas import graficar_y_guardar_boxplot
output_path = 'data/graficas/boxplot_Numero_carboxilos_actividad.png'
if not os.path.exists(output_path):
    graficar_y_guardar_boxplot(df, 'Clasificacion_ATS', 'Numero_carboxilos', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")




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
print(df)

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

"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'TPSA')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_TPSA.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
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
#guardar_csv(df, 'data/NumRotatableBonds.csv')

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

"""
Detección de outliers 
"""
outliers = detectar_outliers(df, 'NumRotatableBonds')
outliers = pd.DataFrame(outliers)
output_path = 'data/outliers_NumRotatableBonds.csv'
if not os.path.exists(output_path):
    guardar_csv(outliers, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Estadísticas descriptivas agrupadas
estadisticas = df.groupby('Clasificacion_ATS')['NumRotatableBonds'].describe()
#guardar_csv(estadisticas, 'data/estadisticas_NumRotatableBonds.csv')
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

"""
ANALIZAMOS LA NORMALIDAD DE LOS DATOS
"""
from modules.procesamiento.analisis_estadistico import calcular_sesgo_curtosis

# *** Define las columnas que deseas analizar ***
columnas_a_analizar = ['LogP', 'Peso_Molecular', 'NumHAcceptors', 'NumHDonors',
                       'Dobles_Enlaces', 'Triples_Enlaces', 'Numero_hidroxilos',
                       'Numero_carboxilos', 'TPSA', 'NumRotatableBonds']  # Reemplaza con tus columnas

# Calcular sesgo y curtosis
resultados_sesgo_curtosis = calcular_sesgo_curtosis(df, columnas_a_analizar)
print("Resultados de Sesgo y Curtosis:")
print(resultados_sesgo_curtosis)

"""
Graficar la normalidad de la variable LogP
"""
from modules.procesamiento.graficas import graficar_normalidad

output_path = 'data/graficas/histogramaplot_normalidad_LogP.png'
if not os.path.exists(output_path):
    graficar_normalidad(df, 'LogP', output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


from modules.procesamiento.graficas import graficar_qq_multivariable


# Variables a graficar
variables = [
    'Peso_Molecular', 'NumHAcceptors', 'NumHDonors',
    'Dobles_Enlaces', 'Triples_Enlaces']

# Llama a la función para graficar los Q-Q Plots
output_path = 'data/graficas/histogramaplot_normalidad_variables no normales.png'
if not os.path.exists(output_path):
    graficar_qq_multivariable(df, variables, output_path, titulo="Q-Q Plots de Variables No Normales")
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Variables a graficar
variables = [
    'Numero_hidroxilos',
    'Numero_carboxilos', 'TPSA', 'NumRotatableBonds']

# Llama a la función para graficar los Q-Q Plots
output_path = 'data/graficas/histogramaplot_normalidad_variables no normales2.png'
if not os.path.exists(output_path):
    graficar_qq_multivariable(df, variables, output_path, titulo="Q-Q Plots de Variables No Normales")
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
PRUEBA DE NORMALIDAD DE  Kolmogorov-Smirnov 
"""
from scipy import stats
print('Analisis de la normalidad de columna LogP')
# *** Define la columna que deseas analizar ***
columna_a_analizar = 'LogP'  # Reemplaza con el nombre real de tu columna

# Realizar la prueba de Kolmogorov-Smirnov
try:
    ks_statistic, p_value = stats.kstest(df[columna_a_analizar].dropna(), 'norm') #Manejo de valores nulos
except KeyError:
    print(f"Error: La columna '{columna_a_analizar}' no existe en el DataFrame.")
    exit()
except Exception as e:
    print(f"Error al ejecutar la prueba KS: {e}")
    exit()

print(f"Estadístico K-S: {ks_statistic:.4f}")
print(f"Valor p: {p_value:.4f}")

# Interpretar los resultados
alpha = 0.05  # Nivel de significancia común

if p_value > alpha:
    print("La columna parece seguir una distribución normal (no se rechaza la hipótesis nula).")
else:
    print("La columna no parece seguir una distribución normal (se rechaza la hipótesis nula).")

"""
CALCULAMOS LA MEDIANA DE LOS DATOS
"""

from modules.procesamiento.analisis_estadistico import obtener_mediana

# Suponiendo que ya tienes el DataFrame 'df' cargado con los datos
# Especifica las columnas de interés
columnas_interes = ['LogP', 'Peso_Molecular', 'NumHAcceptors', 'NumHDonors',
                    'Dobles_Enlaces', 'Triples_Enlaces', 'Numero_hidroxilos',
                    'Numero_carboxilos', 'TPSA', 'NumRotatableBonds']

# Llamar a la función para obtener la mediana de las columnas especificadas
mediana_resultados = obtener_mediana(df, columnas_interes)

# Mostrar el resultado
print("Mediana de cada columna:")
print(mediana_resultados)

"""
OBTENEMOS LA MEDIA
"""
from modules.procesamiento.analisis_estadistico import obtener_media

media_resultados = obtener_media(df, columnas_interes)
# Mostrar el resultado
print("Media de cada columna:")
print(media_resultados)

"""
calculo del rango intercuartilico
"""
from modules.procesamiento.analisis_estadistico import calcular_iqr

iqr_resultados  = calcular_iqr(df, columnas_interes)

print("Rango Intercuartílico (IQR) de cada columna:")
print(iqr_resultados)

"""
ESTADISTICOS DE CADA VARIABLE
"""
from modules.procesamiento.analisis_estadistico import obtener_estadisticos

output_path = 'data/estadisticos.csv'
if not os.path.exists(output_path):
    estadisticos = obtener_estadisticos(df, columnas_interes)
    print("estadisticos de cada variable:")
    guardar_csv(estadisticos, output_path)
    print(estadisticos)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


"""
ANALISIS DE LA VARIABLE CATEGORICA ATS
"""
from modules.procesamiento.analisis_estadistico import generar_tabla_frecuencias

print(generar_tabla_frecuencias(df, 'Clasificacion_ATS'))

"""
ANALISIS MULTIVARIADO DE LOS DATOS
"""
from modules.procesamiento.analisis_estadistico import obtener_estadisticas_por_grupo
columnas_numericas = ['LogP', 'Peso_Molecular', 'NumHAcceptors', 'NumHDonors',
                    'Dobles_Enlaces', 'Triples_Enlaces', 'Numero_hidroxilos',
                    'Numero_carboxilos', 'TPSA', 'NumRotatableBonds']

output_path = 'data/estadisticas_grupo.csv'
if not os.path.exists(output_path):
    estadisticas_grupo = obtener_estadisticas_por_grupo(df, 'Clasificacion_ATS', columnas_numericas)
    print(estadisticas_grupo.reset_index())
    estadisticas_grupo= estadisticas_grupo.reset_index()
    guardar_csv(estadisticas_grupo, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Escalado de variables con z-score
"""

from sklearn.preprocessing import StandardScaler

# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
df_scaled = df.copy()
df_scaled[['LogP_scaled', 'Peso_Molecular_scaled','TPSA_scaled']] = scaler.fit_transform(df[['LogP', 'Peso_Molecular', 'TPSA']])

# Ver los resultados
print(df_scaled.describe())

"""
Escalado max-min
"""
from sklearn.preprocessing import MinMaxScaler

# Crear el escalador
scaler = MinMaxScaler()

# Aplicar escalado solo a la columna NumRotatableBonds
df_scaled = df_scaled.copy()
df_scaled['NumRotatableBonds_scaled'] = scaler.fit_transform(df[['NumRotatableBonds']])

# Ver los resultados
print(df_scaled['NumRotatableBonds_scaled'].describe())

"""
scatterplot
"""
from modules.procesamiento.graficas import generar_pairplot

columnas = ['LogP_scaled', 'Peso_Molecular_scaled', 'NumRotatableBonds_scaled']

output_path = 'data/graficas/scater_escaladas.png'
if not os.path.exists(output_path):
    generar_pairplot(df_scaled, columnas, output_path) #genera scatterplot
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

output_path = 'data/escalado_somevariables.csv'
if not os.path.exists(output_path):
    guardar_csv(df_scaled, 'data/escalado_somevariables.csv') #permite veriricar como se observan la variable escalada
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

output_path = 'data/graficas/NumRotatableBonds_scaled.png'
if not os.path.exists(output_path):
    graficar_y_guardar_variable_continua(df_scaled, 'NumRotatableBonds_scaled', output_path) #grafica la variable numero de enlaces rotables
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Prueba de correlación de Pearson
"""
from scipy.stats import pearsonr
#entre Peso molecular y Numero de enlaces rotables

print('Prueba de correlación entre Peso molecular y Numero de enlaces rotables ')
r, p_value = pearsonr(df_scaled['Peso_Molecular_scaled'], df_scaled['NumRotatableBonds_scaled'])
print(f'Correlación: {r:.2f}, Valor p: {p_value:.5f}')

#entre Peso molecular y LogP

print('Prueba de correlación entre Peso molecular y LogP')
r, p_value = pearsonr(df_scaled['Peso_Molecular_scaled'], df_scaled['LogP_scaled'])
print(f'Correlación: {r:.2f}, Valor p: {p_value:.5f}')

#entre LogP y Numero de enlaces rotables

print('Prueba de correlación entre LogP y Numero de enlaces rotables')
r, p_value = pearsonr(df_scaled['LogP_scaled'], df_scaled['NumRotatableBonds_scaled'])
print(f'Correlación: {r:.2f}, Valor p: {p_value:.5f}')

# Aplicar escalado solo las demás columnas
df_scaled = df_scaled.copy()
df_scaled['Dobles_Enlaces_scaled'] = scaler.fit_transform(df[['Dobles_Enlaces']])
df_scaled['NumHAcceptors_scaled'] = scaler.fit_transform(df[['NumHAcceptors']])
df_scaled['NumHDonors_scaled'] = scaler.fit_transform(df[['NumHDonors']])

"""
scatterplot con las demás variables
"""
from modules.procesamiento.graficas import generar_pairplot


output_path = 'data/graficas/scater_escaladas_variables no tan relacionadas.png'
columnas = ['LogP_scaled', 'TPSA_scaled', 'Dobles_Enlaces_scaled', 'NumHAcceptors_scaled', 'NumHDonors_scaled']
if not os.path.exists(output_path):
    generar_pairplot(df_scaled, columnas, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Prueba de correlación de Pearson
"""
import pandas as pd
from modules.procesamiento.analisis_estadistico  import calcular_correlacion

# Suponiendo que df_scaled ya esté definido y contiene los datos escalados
# Lista de columnas para comparar en pares
columnas_relacionadas = ['LogP_scaled', 'TPSA_scaled', 'Dobles_Enlaces_scaled', 'NumHAcceptors_scaled', 'NumHDonors_scaled',
                         'NumRotatableBonds_scaled', 'Peso_Molecular_scaled']

# Llamar a la función que calcula las correlaciones
output_path = 'data/pearson_pvalue_variablespococorre.csv'
if not os.path.exists(output_path):
    resultados = calcular_correlacion(df_scaled, columnas_relacionadas)
    print(resultados)
# Imprimir los resultados
    print(resultados)
    guardar_csv(resultados, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
scatterplot con las demás variables y Peso moelcular
"""
from modules.procesamiento.graficas import generar_pairplot

columnas = ['Peso_Molecular_scaled', 'TPSA_scaled', 'Dobles_Enlaces_scaled', 'NumHAcceptors_scaled', 'NumHDonors_scaled']
output_path = 'data/graficas/scater_escaladas_variables_peso_molecular.png'
if not os.path.exists(output_path):
    generar_pairplot(df_scaled, columnas, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Scaterplot vs caegoria
"""
from modules.procesamiento.graficas import generar_pairplot_CATEOGRIAS
columnas_interes = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']
categoria = "Clasificacion_ATS"
output_path = 'data/graficas/pairplot_clasificacion.png'
if not os.path.exists(output_path):
    generar_pairplot_CATEOGRIAS(df_scaled, columnas_interes, categoria, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
Prueba de prueba_mann_whitney para evaluar la mediana de dos grupos de datos
"""

print('Prueba de prueba_mann_whitney para evaluar la mediana de dos grupos de datos')
from modules.procesamiento.analisis_estadistico import prueba_mann_whitney_df
columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
columna_categorica = 'Clasificacion_ATS'
resultado_mann_whitney = prueba_mann_whitney_df(df_scaled, columna_categorica, columnas)
print(resultado_mann_whitney)

"""
DIVIDIR DATOS PARA FEATURE SELECION
"""
print('______________________________ ')
print('DIVIDIR 20-80% DATOS PARA FEATURE SELECION')
print('______________________________ ')
from modules.procesamiento.modelos import dividir_datos
output_path = 'data/train_data.csv'
if not os.path.exists(output_path):
    dividir_datos(df_scaled, columna_etiqueta='Clasificacion_ATS')
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")
"""
Cargamos los datos del train data para realizar la feature selection
"""
train_data = leer_csv('data/train_data.csv')
#convertimos los valores de la variable categorica en factores:
train_data['Clasificacion_ATS'] = train_data['Clasificacion_ATS'].map({'Activo': 1, 'Inactivo': 0})

"""
Realizamos una regresión lógistica con todas las variables
"""
print('______________________________ ')
print('REGRESION LOGISTICA TODAS LAS VARIABLES')
print('______________________________ ')

#from modules.procesamiento.modelos import regresion_logistica
X_columns = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
target = 'Clasificacion_ATS'

#modeloRL, accuracy, cm, report,y_test, y_pred_proba = regresion_logistica(train_data, X_columns, target)


"""
Coeficientes del modelo y Calculo ODDS coeficientes
"""
# Calcular Odds Ratios
#from modules.procesamiento.modelos import obtener_coeficientes
#coeficientes = obtener_coeficientes(X_columns, modeloRL)
#print('Coeficientes del modelo y Odds Ratios de Coeficientes del modelo')
#print('______________________________ ')
#import numpy as np
#coeficientes['odds_ratios'] = np.exp(coeficientes['Coeficiente'])

# Imprimir resultados
#print(coeficientes)

"""
summary del modelo
"""
print('SUMMARY MODELO REGLOG')
print('______________________________ ')
from modules.procesamiento.modelos import regresion_logistica_sm
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data, X_columns, target)

"""
CURVA ROC 
"""
from modules.procesamiento.graficas import graficar_curva_roc

output_path = 'data/graficas/Curva_ROC_REGLOG_todaslasvariables.png'
if not os.path.exists(output_path):
    graficar_curva_roc(y_test,y_pred_proba, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")


"""
Coeficientes del modelo y Calculo ODDS coeficientes
"""
# Calcular Odds Ratios
from modules.procesamiento.modelos import obtener_coeficientes
coeficientes = obtener_coeficientes(X_columns, modelo)
print('Coeficientes del modelo y Odds Ratios de Coeficientes del modelo')
print('______________________________ ')
import numpy as np
coeficientes['odds_ratios'] = np.exp(coeficientes['Coeficiente'])

# Imprimir resultados
print(coeficientes)

"""
EVALUACION DE LOS SUPUESTOS DE LA REGRESION LOGISTICA
"""
print('______________________________ ')
print('EVALUACION DE LOS SUPUESTOS DE LA REGRESION LOGISTICA')
print('______________________________ ')
print('Evaluación de la linealidad')
print('______________________________ ')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Generar términos cuadráticos
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(train_data[X_columns])  # X es tu conjunto de datos con las variables predictoras
y = train_data['Clasificacion_ATS']
model = LogisticRegression()
model.fit(X_poly, y)  # Ajustar el modelo con las nuevas características

import matplotlib.pyplot as plt
import numpy as np

import numpy as np

# Predecir las probabilidades para cada clase
probas = model.predict_proba(X_poly)[:, 1]  # Obtener las probabilidades de la clase positiva (1)

# Calcular los log-odds (logaritmo de las probabilidades)
log_odds = np.log(probas / (1 - probas))  # Log-odds = log(p/(1-p))

import matplotlib.pyplot as plt

# Supongamos que quieres graficar una variable 'LogP_scaled' frente a los log-odds
plt.scatter(X_poly[:, 0], log_odds)
plt.xlabel('LogP_scaled')
plt.ylabel('Log-Odds')
plt.title('Log-Odds vs. LogP_scaled')
plt.show()

print('Independencia en las observaciones')
print('______________________________ ')

# Extraer las probabilidades ajustadas (fitted values)
fitted = modelo.fittedvalues
# Extraer los valores observados (la variable dependiente)
# Esto depende de cómo guardaste tus datos, pero una opción es:
y_observed = modelo.model.endog
# Calcular los residuos crudos
raw_resid = y_observed - fitted

import statsmodels.stats.stattools as st

# Usando los residuos crudos
dw_raw = st.durbin_watson(raw_resid)
print(f'Estadístico Durbin-Watson (residuos crudos): {dw_raw:.3f}')

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.scatter(fitted, raw_resid, alpha=0.7, label='Residuos crudos')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Valores ajustados (fitted values)')
plt.ylabel('Residuos crudos')
plt.title('Residuos crudos vs. Valores ajustados')
plt.legend()
plt.show()

print('______________________________ ')
print('REGRESION LOGISTICA SIN LA VARIABLE NumHAcceptors_scaled ')
print('______________________________ ')

#from modules.procesamiento.modelos import regresion_logistica
X_columns = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHDonors_scaled']
target = 'Clasificacion_ATS'

"""
summary del modelo
"""
print('SUMMARY MODELO REGLOG')
print('______________________________ ')
from modules.procesamiento.modelos import regresion_logistica_sm
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data, X_columns, target)

print('______________________________ ')
print('REGRESION LOGISTICA SIN LA VARIABLE NumHAcceptors_scaled Y NumHDonors_scaled')
print('______________________________ ')

#from modules.procesamiento.modelos import regresion_logistica
X_columns = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']
target = 'Clasificacion_ATS'

"""
summary del modelo
"""
print('SUMMARY MODELO REGLOG')
print('______________________________ ')
from modules.procesamiento.modelos import regresion_logistica_sm
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data, X_columns, target)

print('______________________________ ')
print('REGRESION LOGISTICA SIN LA VARIABLE NumHAcceptors_scaled , NumHDonors_scaled Y Dobles_Enlaces_scaled')
print('______________________________ ')

#from modules.procesamiento.modelos import regresion_logistica
X_columns = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled']
target = 'Clasificacion_ATS'

"""
summary del modelo
"""
print('SUMMARY MODELO REGLOG')
print('______________________________ ')
from modules.procesamiento.modelos import regresion_logistica_sm
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data, X_columns, target)

"""
Calculo VIF del modelo MULTICOLINEALIDAD
"""
print('______________________________ ')
print('Calculo VIF del modelo con las variables LogP_scaled, TPSA_scaled,  NumRotatableBonds_scaled, Peso_Molecular_scaled, Dobles_Enlaces_scaled,  NumHAcceptors_scaled, NumHDonors_scaled')
print('______________________________ ')
from modules.procesamiento.analisis_estadistico import calcular_vif

columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']

resultados_VIF = calcular_vif (df_scaled, columnas)
print(resultados_VIF)


"""
Calculo VIF
"""
print('______________________________ ')
print('Calculo VIF del modelo con las variables LogP_scaled, TPSA_scaled,  NumRotatableBonds_scaled, Peso_Molecular_scaled')
print('______________________________ ')
columnas = ['LogP_scaled', 'TPSA_scaled',  'NumRotatableBonds_scaled', 'Peso_Molecular_scaled']
resultados_VIF = calcular_vif (df_scaled, columnas)
print(resultados_VIF)

"""
ARBOLES DE DECISION
"""
from modules.procesamiento.modelos import arbol_decision
print('______________________________  ')
print('MODELO ARBOL DE DECISION')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(train_data, columnas_predictoras, target)
# Mostrar resultados
print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(cm)
print("Reporte de clasificación:")
print(report)

print('Extraer la importancia de las características')
print('______________________________  ')
importancias = modelo.feature_importances_

# Crear DataFrame para visualizar
df_importancias = pd.DataFrame({
    "Variable": columnas_predictoras,
    "Importancia": importancias
})

# Ordenar por importancia
df_importancias = df_importancias.sort_values(by="Importancia", ascending=False)

# Mostrar las 10 variables más importantes
print(df_importancias.head(10))


"""
Probar el modelo
"""
"""
Realizamos nuevamente el modelo de arboles con el df_seleccionados dejando aparte el df_restantes
"""
""
from modules.procesamiento.modelos import arbol_decision

print('modelo de arboles de decision CONJUNTO PRUEBA')
test_data = leer_csv('data/test_data.csv')
test_data['Clasificacion_ATS'] = test_data['Clasificacion_ATS'].map({'Activo': 1, 'Inactivo': 0})
columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
predict_data = test_data[columnas]

nuevas_predicciones = modelo.predict(predict_data)
test_data['nuevas_predicciones'] = nuevas_predicciones
guardar_csv(test_data, 'data/predicion_arbol.csv')


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data['Clasificacion_ATS']
y_pred = test_data['nuevas_predicciones']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)

## Mostrar la matriz de confusión gráficamente
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#a = disp.plot()
#a.show()

#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")


print('______________________________  ')
print('MODELO ARBOL DE DECISION sin las variables NumHAcceptors_scaled, NumHDonors_scaled')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(train_data, columnas_predictoras, target)
# Mostrar resultados
print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(cm)
print("Reporte de clasificación:")
print(report)

print('Extraer la importancia de las características')
print('______________________________  ')
importancias = modelo.feature_importances_

# Crear DataFrame para visualizar
df_importancias = pd.DataFrame({
    "Variable": columnas_predictoras,
    "Importancia": importancias
})

# Ordenar por importancia
df_importancias = df_importancias.sort_values(by="Importancia", ascending=False)

# Mostrar las 10 variables más importantes
print(df_importancias.head(10))


"""
Probar el modelo
"""
"""
Realizamos nuevamente el modelo de arboles con el df_seleccionados dejando aparte el df_restantes
"""
""
from modules.procesamiento.modelos import arbol_decision

print('modelo de arboles de decision CONJUNTO PRUEBA')
print('______________________________  ')
test_data = leer_csv('data/test_data.csv')
test_data['Clasificacion_ATS'] = test_data['Clasificacion_ATS'].map({'Activo': 1, 'Inactivo': 0})
columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']
predict_data = test_data[columnas]

nuevas_predicciones = modelo.predict(predict_data)
test_data['nuevas_predicciones'] = nuevas_predicciones
guardar_csv(test_data, 'data/predicion_arbol.csv')


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data['Clasificacion_ATS']
y_pred = test_data['nuevas_predicciones']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)

## Mostrar la matriz de confusión gráficamente
#disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#a = disp.plot()
#a.show()

#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")

"""
INCLUSION DE LOS FINGERPRINTS
"""
print('______________________________  ')
print('INCLUSION DE LOS FINGERPRINTS')
print('______________________________  ')
print('INCLUSION DE LOS MACCS')

train_data_MACCS_array =train_data[['Clasificacion_ATS','LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']]

MACCS = train_data['MACCS_array']
# Eliminar los corchetes y convertir cada fila en una lista de enteros
MACCS = train_data['MACCS_array'].str.replace(r"[\[\]]", "", regex=True).str.strip()
#MACCS = train_data['MACCS_array'].str.replace(" ", "")
# Dividir en columnas y convertir a enteros
MACCS_df = MACCS.str.split(expand=True).apply(pd.to_numeric, errors='coerce').astype(int)
#print(MACCS_df)

# Concatenamos ambos DataFrames (asegúrate de que ambos tengan el mismo índice)
train_data_MACCS_array = pd.concat([train_data_MACCS_array.reset_index(drop=True), MACCS_df.reset_index(drop=True)], axis=1)

# Mostramos el DataFrame resultante
#print(train_data_MACCS_array.head())

"""
MODELO DE ARBOLES DE DECISION USANDO MACCS
"""
print('______________________________  ')
print('MODELO DE ARBOLES DE DECISION USANDO MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = [col for col in train_data_MACCS_array.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
#print(columnas_predictoras)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(train_data_MACCS_array, columnas_predictoras, target)
# Mostrar resultados
print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(cm)
print("Reporte de clasificación:")
print(report)

"""
Inclusión de los fingerprints en el conjunto de prueba
"""
"""
INCLUSION DE LOS FINGERPRINTS
"""


test_data_MACCS_array =test_data[['Clasificacion_ATS','LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']]

MACCS = test_data['MACCS_array']
# Eliminar los corchetes y convertir cada fila en una lista de enteros
MACCS = test_data['MACCS_array'].str.replace(r"[\[\]]", "", regex=True).str.strip()
#MACCS = train_data['MACCS_array'].str.replace(" ", "")
# Dividir en columnas y convertir a enteros
MACCS_df = MACCS.str.split(expand=True).apply(pd.to_numeric, errors='coerce').astype(int)
#print(MACCS_df)

# Concatenamos ambos DataFrames (asegúrate de que ambos tengan el mismo índice)
test_data_MACCS_array = pd.concat([test_data_MACCS_array.reset_index(drop=True), MACCS_df.reset_index(drop=True)], axis=1)

# Mostramos el DataFrame resultante
#print(test_data_MACCS_array.head())

from modules.procesamiento.modelos import arbol_decision

print('modelo de arboles de decision CONJUNTO PRUEBA')
print('______________________________  ')
columnas = [col for col in test_data_MACCS_array.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
predict_data = test_data_MACCS_array[columnas]

nuevas_predicciones = modelo.predict(predict_data)
test_data_MACCS_array['nuevas_predicciones'] = nuevas_predicciones
#guardar_csv(test_data, 'data/predicion_arbol.csv')


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_MACCS_array['Clasificacion_ATS']
y_pred = test_data_MACCS_array['nuevas_predicciones']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)

#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")

"""
CROSS VALIDATION
"""
print('______________________________  ')
print('CROSS VALIDATION MODELO MACCS')
print('______________________________  ')

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# Supongamos que tienes tus datos en X (features) y y (target)
modelo = DecisionTreeClassifier()  # Puedes cambiarlo por otro modelo

# Aplicar validación cruzada con 5 folds
y = train_data_MACCS_array['Clasificacion_ATS']
X = train_data_MACCS_array[columnas_predictoras]

scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')

# Mostrar resultados
print("Precisión en cada fold:", scores)
print("Precisión media:", np.mean(scores))
print("Desviación estándar:", np.std(scores))

"""
INCLUSION DE LOS FINGERPRINTS DE MORGAN
"""
print('______________________________  ')
print('INCLUSION DE LOS FINGERPRINTS DE MORGAN')
print('______________________________  ')
print('INCLUSION DE LOS MORGAN')

calcular_ecfp(train_data, 'SMILES')
from modules.procesamiento.calculo_fingerprints import convertir_ECFP_a_numpy
convertir_ECFP_a_numpy(train_data, 'ECFP')
#print(train_data['ECFP_array'])
ECFP = train_data['ECFP_array']
ECFP = pd.DataFrame(ECFP.tolist())
#print(ECFP)
train_data_MACCS_array = pd.concat([train_data_MACCS_array.reset_index(drop=True), ECFP.reset_index(drop=True)], axis=1)
#print(train_data_MACCS_array)

print('______________________________  ')
print('MODELO DE ARBOLES DE DECISION USANDO MACCS y MORGAN')
print('______________________________  ')

# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = [col for col in train_data_MACCS_array.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
#print(columnas_predictoras)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(train_data_MACCS_array, columnas_predictoras, target)
# Mostrar resultados
print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(cm)
print("Reporte de clasificación:")
print(report)

#guardar_csv(train_data_MACCS_array, 'data/train_data_MACCS_array.csv')

print('modelo de arboles MACCS y MORGAN de decision CONJUNTO PRUEBA')
print('______________________________  ')

calcular_ecfp(test_data, 'SMILES')
from modules.procesamiento.calculo_fingerprints import convertir_ECFP_a_numpy
convertir_ECFP_a_numpy(test_data, 'ECFP')
#print(train_data['ECFP_array'])
ECFP = test_data['ECFP_array']
ECFP = pd.DataFrame(ECFP.tolist())
#print(ECFP)
test_data_MACCS_array = pd.concat([test_data_MACCS_array.reset_index(drop=True), ECFP.reset_index(drop=True)], axis=1)
#print(test_data_MACCS_array)

columnas = [col for col in test_data_MACCS_array.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
predict_data = test_data_MACCS_array[columnas]

nuevas_predicciones = modelo.predict(predict_data)
test_data_MACCS_array['nuevas_predicciones'] = nuevas_predicciones
#guardar_csv(test_data, 'data/predicion_arbol.csv')


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_MACCS_array['Clasificacion_ATS']
y_pred = test_data_MACCS_array['nuevas_predicciones']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)

#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")

"""
CROSS VALIDATION
"""
print('______________________________  ')
print('CROSS VALIDATION MODELO MACCS Y EFPC')
print('______________________________  ')

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# Supongamos que tienes tus datos en X (features) y y (target)
modelo = DecisionTreeClassifier()  # Puedes cambiarlo por otro modelo

# Aplicar validación cruzada con 5 folds
y = train_data_MACCS_array['Clasificacion_ATS']
X = train_data_MACCS_array[columnas_predictoras]

scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')

# Mostrar resultados
print("Precisión en cada fold:", scores)
print("Precisión media:", np.mean(scores))
print("Desviación estándar:", np.std(scores))

"""
MODELO ARBOLES SOLO CON EFPC
"""
df_ecfp = train_data_MACCS_array.iloc[:, 173:]  # Selecciona desde la columna 172 hasta el final
df_ecfp['Clasificacion_ATS'] = train_data_MACCS_array['Clasificacion_ATS']
#print(df_ecfp)

print('______________________________  ')
print('MODELO DE ARBOLES DE DECISION USANDO MORGAN')
print('______________________________  ')

# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = [col for col in df_ecfp.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
#print(columnas_predictoras)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(df_ecfp, columnas_predictoras, target)
# Mostrar resultados
print(f"Precisión del modelo: {accuracy:.4f}")
print("Matriz de confusión:")
print(cm)
print("Reporte de clasificación:")
print(report)

print('modelo de arboles MORGAN de decision CONJUNTO PRUEBA')
print('______________________________  ')


columnas = [col for col in df_ecfp.columns if col != 'Clasificación_ATS' and isinstance(col, str) == False]
predict_data = df_ecfp[columnas]

nuevas_predicciones = modelo.predict(predict_data)
df_ecfp['nuevas_predicciones'] = nuevas_predicciones
#guardar_csv(test_data, 'data/predicion_arbol.csv')


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = df_ecfp['Clasificacion_ATS']
y_pred = df_ecfp['nuevas_predicciones']

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
print("Matriz de Confusión:")
print(cm)

#Cálculo de métricas
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# Mostrar las métricas
print(f"Precisión: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print(f"Exactitud: {accuracy}")

"""
CROSS VALIDATION
"""
print('______________________________  ')
print('CROSS VALIDATION MODELO MORGAN')
print('______________________________  ')

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
import numpy as np

# Supongamos que tienes tus datos en X (features) y y (target)
modelo = DecisionTreeClassifier()  # Puedes cambiarlo por otro modelo

# Aplicar validación cruzada con 5 folds
y = df_ecfp['Clasificacion_ATS']
X = df_ecfp[columnas_predictoras]

scores = cross_val_score(modelo, X, y, cv=5, scoring='accuracy')

# Mostrar resultados
print("Precisión en cada fold:", scores)
print("Precisión media:", np.mean(scores))
print("Desviación estándar:", np.std(scores))


