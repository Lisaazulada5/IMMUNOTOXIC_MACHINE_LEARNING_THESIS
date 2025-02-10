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

print(df) #Los datos están sin escalar nuevamente, ya que se aplicara el escalamiento al conjunto de entrenamiento unicamente.
from modules.procesamiento.modelos import dividir_datos
output_path = 'data/train_data.csv'
if not os.path.exists(output_path):
    dividir_datos(df, columna_etiqueta='Clasificacion_ATS')
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
Escalamiento de los datos de entrenamiento
"""
"""
Escalado de variables con z-score
"""
print('Escalado de los datos de entrenamiento')
print('______________________________ ')
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
#train_data = leer_csv('data/train_data.csv')
output_path = 'data/train_data_scaled.csv'
if not os.path.exists(output_path):
    train_data_scaled = train_data.copy()
    train_data_scaled[['LogP_scaled', 'Peso_Molecular_scaled','TPSA_scaled']] = scaler.fit_transform(train_data[['LogP',
                                                                                                                 'Peso_Molecular', 'TPSA']])
    """
    Escalado max-min
    """
    # Crear el escalador
    scaler = MinMaxScaler()

# Aplicar escalado solo a la columna NumRotatableBonds
    train_data_scaled = train_data_scaled.copy()
    train_data_scaled['NumRotatableBonds_scaled'] = scaler.fit_transform(train_data_scaled[['NumRotatableBonds']])
    # Aplicar escalado solo las demás columnas
    train_data_scaled = train_data_scaled.copy()
    train_data_scaled['Dobles_Enlaces_scaled'] = scaler.fit_transform(train_data_scaled[['Dobles_Enlaces']])
    train_data_scaled['NumHAcceptors_scaled'] = scaler.fit_transform(train_data_scaled[['NumHAcceptors']])
    train_data_scaled['NumHDonors_scaled'] = scaler.fit_transform(train_data_scaled[['NumHDonors']])
    guardar_csv(train_data_scaled, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

train_data_scaled = leer_csv('data/train_data_scaled.csv' )

print('Escalado de los datos de prueba')
print('______________________________ ')

test_data = leer_csv('data/test_data.csv')
#convertimos los valores de la variable categorica en factores:
test_data['Clasificacion_ATS'] = test_data['Clasificacion_ATS'].map({'Activo': 1, 'Inactivo': 0})
# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias

output_path = 'data/test_data_scaled.csv'
if not os.path.exists(output_path):
    test_data_scaled = test_data.copy()
    test_data_scaled[['LogP_scaled', 'Peso_Molecular_scaled','TPSA_scaled']] = scaler.fit_transform(test_data[['LogP',
                                                                                                               'Peso_Molecular', 'TPSA']])
    """
    Escalado max-min
    """
    # Crear el escalador
    scaler = MinMaxScaler()

# Aplicar escalado solo a la columna NumRotatableBonds
    test_data_scaled = test_data_scaled.copy()
    test_data_scaled['NumRotatableBonds_scaled'] = scaler.fit_transform(test_data_scaled[['NumRotatableBonds']])
    test_data_scaled = test_data_scaled.copy()
    test_data_scaled['Dobles_Enlaces_scaled'] = scaler.fit_transform(test_data_scaled[['Dobles_Enlaces']])
    test_data_scaled['NumHAcceptors_scaled'] = scaler.fit_transform(test_data_scaled[['NumHAcceptors']])
    test_data_scaled['NumHDonors_scaled'] = scaler.fit_transform(test_data_scaled[['NumHDonors']])
    guardar_csv(test_data_scaled, output_path)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
FEATURE SELECTION
"""

"""
Realizamos una regresión lógistica con todas las variables
"""
print('______________________________ ')
print('REGRESION LOGISTICA TODAS LAS VARIABLES')
print('______________________________ ')

X_columns = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
target = 'Clasificacion_ATS'

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
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data_scaled, X_columns, target)

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
ANALISIS DE LOS PUNTOS INFLUYENTES USANDO LA DISTANCIA DE COOK
"""
print('PUNTOS INFLUYENTES USANDO LA DISTANCIA DE COOK')
print('______________________________ ')
from modules.procesamiento.analisis_estadistico import Puntos_influentes_Cook

Puntos_influentes_Cook(modelo, train_data_scaled)

from modules.procesamiento.graficas import graficar_cook
output_path = 'data/graficas/puntos_influyentes.png'
if not os.path.exists(output_path):
    graficar_cook(modelo, train_data_scaled,output_path)
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
#print(coeficientes)

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

from modules.procesamiento.analisis_estadistico import calcular_linealidad
log_odds, model, X_poly = calcular_linealidad(train_data_scaled, X_columns, target)

from modules.procesamiento.graficas import graficar_supeustolinealidad
X = X_poly[:,0]

output_path = 'data/graficas/supuestolinealidad_LogP_scaled.png'
if not os.path.exists(output_path):
    graficar_supeustolinealidad('LogP_scaled', X,  log_odds,  output_path) #variable debe usarse por posicion de variable y no su nombre
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
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
#summary del modelo
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
#summary del modelo
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
#summary del modelo
"""
print('SUMMARY MODELO REGLOG')
print('______________________________ ')
from modules.procesamiento.modelos import regresion_logistica_sm
modelo, summary, accuracy, cm, report, y_test, y_pred_proba = regresion_logistica_sm(train_data, X_columns, target)
"""
"""
Calculo VIF del modelo MULTICOLINEALIDAD
"""
print('______________________________ ')
print('Calculo VIF del modelo con las variables LogP_scaled, TPSA_scaled,  NumRotatableBonds_scaled, Peso_Molecular_scaled, Dobles_Enlaces_scaled,  NumHAcceptors_scaled, NumHDonors_scaled')
print('______________________________ ')
from modules.procesamiento.analisis_estadistico import calcular_vif

columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']

resultados_VIF = calcular_vif (train_data_scaled, columnas)
print(resultados_VIF)

"""
Calculo VIF
"""
print('______________________________ ')
print('Calculo VIF del modelo con las variables LogP_scaled, TPSA_scaled,  NumRotatableBonds_scaled, Peso_Molecular_scaled')
print('______________________________ ')
columnas = ['LogP_scaled', 'TPSA_scaled',  'NumRotatableBonds_scaled', 'Peso_Molecular_scaled']
resultados_VIF = calcular_vif (train_data_scaled, columnas)
print(resultados_VIF)

"""
#ARBOLES DE DECISION
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
modelo, accuracy, cm, report = arbol_decision(train_data_scaled, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_todaslasvariables.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Mostrar resultados
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

# Mostrar las variables más importantes
print(df_importancias)

print('______________________________  ')
print('MODELO ARBOL DE DECISION sin las variables NumHAcceptors_scaled NumHDonors_scaled')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
columnas_predictoras = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']
target = "Clasificacion_ATS"
# Llamar al modelo
modelo, accuracy, cm, report = arbol_decision(train_data_scaled, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_sinNumHAcceptors_scaledNumHDonors_scaled.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
BOSQUES DE DECISIÓN
"""
print('______________________________  ')
print('MODELO BOSQUES DE DECISION')
print('______________________________  ')
from modules.procesamiento.modelos import entrenar_random_forest

X = train_data_scaled[['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']]
y= train_data_scaled["Clasificacion_ATS"]

random, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

# Mostrar resultados
print('Extraer la importancia de las características')
print('______________________________  ')
columnas = ['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled',  'NumHAcceptors_scaled', 'NumHDonors_scaled']
importancias = random.feature_importances_

# Crear DataFrame para visualizar
df_importancias = pd.DataFrame({
    "Variable": columnas,
    "Importancia": importancias
})

# Ordenar por importancia
df_importancias = df_importancias.sort_values(by="Importancia", ascending=False)

# Mostrar las 10 variables más importantes
print(df_importancias)

print('______________________________  ')
print('MODELO BOSQUES DE DECISION sin las variables NumHAcceptors_scaled NumHDonors_scaled ')
from modules.procesamiento.modelos import entrenar_random_forest

X = train_data_scaled[['LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled']]
y= train_data_scaled["Clasificacion_ATS"]

random, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_sin_NumRotatableBonds_scaledPeso_Molecular_scaled.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#INCLUSION DEL LOS FINGERPRINTS DENTRO DE LOS DATOS DE ENTRENAMIENTO.
"""
print('______________________________  ')
print('INCLUSION DEL LOS FINGERPRINTS DENTRO DE LOS DATOS DE ENTRENAMIENTO ')
train_data_fingerprints = train_data_scaled[['Clasificacion_ATS', 'LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled','SMILES']].copy()
#calculamos los ECFP  a partir de los SMILES
from modules.procesamiento.calculo_fingerprints import calcular_ecfp
calcular_ecfp(train_data_fingerprints, 'SMILES')
#print(train_data_fingerprints['ECFP'][0])
from rdkit.Chem import DataStructs
ECFP_array = []
for i in train_data_fingerprints['ECFP']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    ECFP_array.append(arr)

train_data_fingerprints = pd.concat([train_data_fingerprints, pd.DataFrame(ECFP_array)], axis  =1)
#print(train_data_fingerprints)
#calculamos los MACCS a partir de los SMILES
calcular_fingerprints_maccs(train_data_fingerprints, 'SMILES')
from rdkit.Chem import DataStructs
MACCS_array = []
for i in train_data_fingerprints['MACCS']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    MACCS_array.append(arr)
train_data_fingerprints = pd.concat([train_data_fingerprints, pd.DataFrame(MACCS_array)], axis  =1)
#guardar_csv(train_data_fingerprints, 'data/train_data_fingerprints.csv')

print('______________________________  ')
print('INCLUSION DEL LOS FINGERPRINTS DENTRO DE LOS DATOS DE PRUEBA ')
test_data_scaled = leer_csv('data/test_data_scaled.csv')
#print(test_data_scaled.columns)
test_data_fingerprints = test_data_scaled[['Clasificacion_ATS', 'LogP_scaled', 'TPSA_scaled', 'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
                    'Dobles_Enlaces_scaled','SMILES']].copy()
#print(test_data_fingerprints.columns)
#calcular los ECFP para el conjunto de prueba
calcular_ecfp(test_data_fingerprints, 'SMILES')
#print(train_data_fingerprints['ECFP'][0])
from rdkit.Chem import DataStructs
ECFP_array_test = []
for i in test_data_fingerprints['ECFP']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    ECFP_array_test.append(arr)

#print(ECFP_array_test)
test_data_fingerprints = pd.concat([test_data_fingerprints, pd.DataFrame(ECFP_array_test)], axis  =1)
#print(test_data_fingerprints)

#calculamos los MACCS a partir de los SMILES
calcular_fingerprints_maccs(test_data_fingerprints, 'SMILES')
from rdkit.Chem import DataStructs
MACCS_array_test = []
for i in test_data_fingerprints['MACCS']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    MACCS_array_test.append(arr)
test_data_fingerprints = pd.concat([test_data_fingerprints, pd.DataFrame(MACCS_array_test)], axis  =1)
#test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str) #convertimos las columnas a strings
#print(test_data_fingerprints)


"""
"""
#MODELADO CON FINGERPRINTS
"""
print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo

train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])

MACCS = train_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras = MACCS.columns
target = "Clasificacion_ATS"
# Llamar al modelo
modeloMACCS, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_MACCS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras_test = MACCS_test.columns #extrae los nombres de las columnas del df
predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modeloMACCS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_MACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_MACCS']
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


print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS y FISICOQUIMICOS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo

train_data_fingerprints.columns = train_data_fingerprints.columns.astype(str)

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)
columnas_predictoras = columns_subset.columns.astype(str)

target = "Clasificacion_ATS"
# Llamar al modelo
modeloMACCS_FISICOQUIMICOS, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_MACCSFISICOQUIMICOS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str)
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columnas_predictoras_test = columns_subset_test.columns.astype(str)

predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modeloMACCS_FISICOQUIMICOS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_MACCSFISICOQUIMICOS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_MACCSFISICOQUIMICOS']
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

print('______________________________  ')
print('MODELO ARBOL DE DECISION ECFP ')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo

ECFP = train_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras = ECFP.columns

#columnas_predictoras = columnas_predictoras.astype(int)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo_ECFP, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_ECFP.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test.columns

predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modelo_ECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFP'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFP']
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

print('______________________________  ')
print('MODELO ARBOL DE DECISION ECFP y FISICOQUIMICAS ')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
# Seleccionar las columnas que no son la posición 0 ni de la 7 a la 2054

columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columnas_predictoras = columns_subset.columns

#columnas_predictoras = columnas_predictoras.astype(int)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo_ECFP_FISICOQUIMICAS, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_ECFPFISICOQUIMICAS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
columnas_predictoras_test = columns_subset_test.columns

predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modelo_ECFP_FISICOQUIMICAS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPFISICOQUIMICAS']
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

print('______________________________  ')
print('MODELO ARBOL DE DECISION ECFP y MACCS ')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
# Seleccionar las columnas que no son la posición 0 ni de la 7 a la 2054
#MACCS = MACCS.add_prefix("MACCS_")
#train_data_fingerprints['MACCS'] = train_data_fingerprints['MACCS'].add_prefix("MACCS_")
columns_subset = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras = columns_subset.columns.astype(str)

#columnas_predictoras = columnas_predictoras.astype(int)
target = "Clasificacion_ATS"
# Llamar al modelo
modelo_ECFP_MACCS, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_ECFPMACCS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test], axis=1)
columnas_predictoras_test = columns_subset_test.columns.astype(str)

predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modelo_ECFP_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPFMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPFMACCS']
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

print('______________________________  ')
print('MODELO ARBOL DE DECISION ECFP,  MACCS y FISICOQUIMICAS ')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
# Seleccionar las columnas que no son la posición 0 ni de la 7 a la 2054

columns_subset = pd.concat([MACCS, ECFP, Fisicoquimicas], axis=1)
columnas_predictoras = columns_subset.columns.astype(str)

#columnas_predictoras = columnas_predictoras.astype(int)
target = "Clasificacion_ATS"
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy, cm, report = arbol_decision(train_data_fingerprints, columnas_predictoras, target)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_arboles_ECFPMACCSFISICOQUIMICAS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(cm, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP MACCS FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test], axis=1)
#print(columns_subset_test)
columnas_predictoras_test = columns_subset_test.columns.astype(str)
#print(columnas_predictoras_test)
#guardar_csv(columns_subset_test, 'data/columas_preductoras_test.csv')
"""
"""
predict_data = test_data_fingerprints[columnas_predictoras_test]
nuevas_predicciones = modeloECFP_MACCS_FISICOQUIMICAS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPFMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPFMACCSFISICOQUIMICAS']
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



"""
"""
#MODELADO DE BOSQUES ALEATORIOS
"""
print('***************************')
print('MODELO BOSQUES DE DECISION MACCS')
print('______________________________  ')
from modules.procesamiento.modelos import entrenar_random_forest
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_MACCS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
#columnas_predictoras_test = MACCS_test #extrae los nombres de las columnas del df
predict_data = MACCS_test
nuevas_predicciones = randomMACCS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_MACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_MACCS']
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

print('***************************')
print('MODELO BOSQUES DE DECISION MACCS y FISICOQUIMICOS')
print('______________________________  ')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columnas_predictoras = pd.concat([Fisicoquimicas, MACCS], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS_FISICOQUIMICOS, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_MACCS_FISICOQUIMICOS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')

Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)


predict_data = columns_subset_test
nuevas_predicciones = randomMACCS_FISICOQUIMICOS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_MACCS_FISICOQUIMICOS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_MACCS_FISICOQUIMICOS']
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

print('***************************')
print('MODELO BOSQUES DE DECISION ECFP')
print('______________________________  ')
ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
#columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_ECFP.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')

ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
predict_data = ECFP_test
nuevas_predicciones = randomECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_MACCS_FISICOQUIMICOS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_MACCS_FISICOQUIMICOS']
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

print('***************************')
print('MODELO BOSQUES DE DECISION ECFP Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([Fisicoquimicas, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_FISICOQUIMICOS, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_ECFP_FISICOQUIMICOS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP Y FISICOQUIMICOS')
print('--------------------------------')

ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
columnas_predictoras.columns = columnas_predictoras.columns.astype(str)
predict_data = columnas_predictoras
nuevas_predicciones = randomECFP_FISICOQUIMICOS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFP_FISICOQUIMICOS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFP_FISICOQUIMICOS']
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

print('***************************')
print('MODELO BOSQUES DE DECISION ECFP Y MACCS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_ECFP Y MACCS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP Y MACCS')
print('--------------------------------')

columnas_predictoras = pd.concat([MACCS_test, ECFP_test], axis=1)
columnas_predictoras.columns = columnas_predictoras.columns.astype(str)
predict_data = columnas_predictoras
nuevas_predicciones = randomECFP_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFP Y MACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFP Y MACCS']
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

print('***************************')
print('MODELO BOSQUES DE DECISION ECFP, MACCS Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP, Fisicoquimicas], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS_FISICOQUIMICOS, confusion_matrix = entrenar_random_forest(X, y)

from modules.procesamiento.graficas import plot_confusion_matrix
output_path = 'data/graficas/matriz_confusion_bosques_ECFP_MACCS_FISICOQUIMICOS.png'
if not os.path.exists(output_path):
    class_names = ["Inactivo", "Activo"]
    plot_confusion_matrix(confusion_matrix, output_path, class_names)
    print(f"Archivo generado: {output_path}")
else:
    print(f"El archivo {output_path} ya existe. No se ha procesado de nuevo.")

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP, MACCS Y FISICOQUIMICOS')
print('--------------------------------')

columnas_predictoras = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test], axis=1)
columnas_predictoras.columns = columnas_predictoras.columns.astype(str)
predict_data = columnas_predictoras
nuevas_predicciones = randomECFP_MACCS_FISICOQUIMICOS.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFP_MACCS_FISICOQUIMICOS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFP_MACCS_FISICOQUIMICOS']
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




"""
"""
#Se explora que el dataset sea diverso SIMILARIDAD DE TANIMOTO
""" 
print('____________________')
print('\n Diversidad datos')
print('\n ____________________')
df_diversidad = df_scaled.copy()
df_diversidad = df_diversidad[['INPUT', 'SMILES']]
print('\n Calculo de fingerprints ECFP')
print('____________________')

from modules.procesamiento.calculo_fingerprints import calcular_ecfp
calcular_ecfp(df_diversidad, 'SMILES')
print(df_diversidad.columns)
#print(train_data_fingerprints['ECFP'][0])
#from rdkit.Chem import DataStructs
#ECFP_array = []
#for i in train_data_fingerprints['ECFP']:
#    arr = np.zeros((0,), dtype=np.int8)
#    DataStructs.ConvertToNumpyArray(i, arr)
#    ECFP_array.append(arr)

#train_data_fingerprints = pd.concat([train_data_fingerprints, pd.DataFrame(ECFP_array)], axis  =1)
#print(train_data_fingerprints)

print(type(df_diversidad["ECFP"].iloc[0]))  # Tipo de dato del primer fingerprint
import time
from rdkit.SimDivFilters import MaxMinPicker

# Inicializar el selector
mmp = MaxMinPicker()

# Extraer los fingerprints
fps = df_diversidad["ECFP"].tolist()

# Medir tiempo de ejecución
t1 = time.time()
bv_ids = mmp.LazyBitVectorPick(fps, len(fps), 50)
t2 = time.time()

print("Selección completada en %.2f segundos" % (t2 - t1))

# Filtrar df_diversidad con los índices seleccionados
df_seleccionados = df_diversidad.iloc[bv_ids]

# Mostrar los primeros resultados
print(df_seleccionados)

import matplotlib.pyplot as plt
from rdkit import DataStructs

# Lista para almacenar las similitudes de Tanimoto
dist_hist = []

# Comparar la similitud entre todas las moléculas seleccionadas
for i in range(len(bv_ids)):
    for j in range(i + 1, len(bv_ids)):
        sim = DataStructs.TanimotoSimilarity(fps[bv_ids[i]], fps[bv_ids[j]])
        dist_hist.append(sim)

# Graficar el histograma de similitudes
plt.hist(dist_hist, bins=20, edgecolor="black")
plt.title("Distribución de Similitudes (MaxMin Picks)")
plt.xlabel("Similaridad de Tanimoto")
plt.ylabel("Frecuencia")
plt.show()
"""




"""
"""
#MODELO xgboost
"""
"""
#MODELO USANDO MACCS
"""
print('\n ****************')
print('MODELOS xgboost')
print('****************')
from modules.procesamiento.modelos import entrenar_xgboost
print('\n----------------')
print('MODELOS MACCS')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS = entrenar_xgboost(X,y)


print('\n----------------')
print('MODELOS MACCS y Fisicoquimicas')
print('----------------')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_Fisicoquimicas = entrenar_xgboost(X,y)

print('\n----------------')
print('MODELOS ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP = entrenar_xgboost(X,y)

print('\n----------------')
print('MODELOS ECFP y Fisicoquimicos')
print('----------------')

columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP_Fisicoquimicos = entrenar_xgboost(X,y)

print('\n----------------')
print('MODELOS ECFP Y MACCS ')
print('----------------')
MACCS = MACCS.add_prefix("MACCS_")
#print(MACCS)
columns_subset = pd.concat([ECFP, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
#print(target)
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP = entrenar_xgboost(X,y)

print('\n----------------')
print('MODELOS ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas = entrenar_xgboost(X,y)
"""


"""
"""
#MÁQUINAS DE SOPORTE VECTORIAL
"""
print('\n ****************')
print('MODELOS MSV')
print('****************')
from modules.procesamiento.modelos import entrenar_svm

print('\n----------------')
print('MODELOS MSV MACCS')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_MACCS = entrenar_svm(X, y)

print('\n----------------')
print('MODELOS MSV MACCS fisicoqumicas')
print('----------------')
Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_MACCS_Fisicoquimicas = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_ECFP = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP Y FISICOQUIMICOS')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_ECFP = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP Y MACCS ')
print('----------------')
MACCS = MACCS.add_prefix("MACCS_")
#print(MACCS)
columns_subset = pd.concat([ECFP, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
#print(target)
X = columnas_predictoras
y= target
MSV_MACCS_ECFP = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_MACCS_ECFP_fisicoquimicas = entrenar_svm(X,y)
"""


"""
"""
#REDES NEURONALES
"""
print('\n ****************')
print('MODELOS REDES NEURONALES')
print('****************')
from modules.procesamiento.modelos import entrenar_red_neuronal

print('\n----------------')
print('MODELOS REDES NEURONALES MACCS')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)

columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
RN_MACCS = entrenar_red_neuronal(X, y)


print('\n----------------')
print('MODELOS REDES NEURONALES MACCS fisicoqumicas')
print('----------------')
Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)

columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
RN_MACCS_Fisicoquimicas = entrenar_red_neuronal(X,y)


print('\n----------------')
print('MODELOS REDES NEURONALES ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]

columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
RN_ECFP = entrenar_red_neuronal(X,y)

print('\n----------------')
print('REDES NEURONALES ECFP Y FISICOQUIMICOS')
print('----------------')

#ECFP = train_data_fingerprints.iloc[:, 6:2054]
columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
RN_ECFP = entrenar_red_neuronal(X,y)



print('\n----------------')
print('REDES NEURONALES ECFP Y MACCS')
print('----------------')
MACCS = MACCS.add_prefix("MACCS_")
#print(MACCS)
columns_subset = pd.concat([ECFP, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

RN_ECFP = entrenar_red_neuronal(X,y)



print('\n----------------')
print('MODELOS REDES NEURONALES ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
RN_MACCS_ECFP_fisicoquimicas = entrenar_red_neuronal(X,y)
"""



"""
#DADO QUE LAS VARIABLES PROPUESTAS PARECEN NO SER SUCIFIENTES SE HACE NECESARIO EL CALCULO DE OTRAS VARIABLES

#Para esto se usa el dataframe que ya tiene las variables escaladas y que contiene los fingerprints
"""
df_nuevasvariables = train_data_fingerprints.copy()
from modules.procesamiento.calculo_descriptores_moleculares import agregar_cargas_moleculares
agregar_cargas_moleculares(df_nuevasvariables, 'SMILES')
#guardar_csv(df_nuevasvariables, 'data/df_nuevasvariables.csv')
#graficar_y_guardar_variable_continua(df_nuevasvariables,'Carga_Gasteiger', 'data/graficas/Carga_Gasteiger')
#graficar_y_guardar_boxplot(df_nuevasvariables, 'Clasificacion_ATS', 'Carga_Gasteiger', 'data/graficas/cajas_bigotes')
df_nuevasvariables = leer_csv('data/df_nuevasvariables.csv')
from sklearn.preprocessing import StandardScaler

# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
#df_nuevasvariables = df.copy()
df_nuevasvariables['Carga_Gasteiger_scaled'] = scaler.fit_transform(df_nuevasvariables[['Carga_Gasteiger']])
print(df_nuevasvariables.columns)


print('\n----------------')
print('MODELOS ECFP Y FISICOQUIMICOS con gaiser')
print('----------------')
from modules.procesamiento.modelos import entrenar_svm
df_nuevasvariables.columns = df_nuevasvariables.columns.astype(str)
df_nuevasvariables = df_nuevasvariables.drop(columns=['MACCS', 'SMILES', 'ECFP'])
Fisicoquimicas = df_nuevasvariables[['LogP_scaled', 'TPSA_scaled',
       'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
       'Dobles_Enlaces_scaled', 'Carga_Gasteiger_scaled']]
ECFP = df_nuevasvariables.iloc[:, 6:2054]
MACCS = df_nuevasvariables.iloc[:, 2054:2222]
MACCS = MACCS.add_prefix("MACCS_")
#print(ECFP)
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = df_nuevasvariables["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_ECFP = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP Y FISICOQUIMICOS sin gaiser')
print('----------------')
from modules.procesamiento.modelos import entrenar_svm
#df_nuevasvariables.columns = df_nuevasvariables.columns.astype(str)
#df_nuevasvariables = df_nuevasvariables.drop(columns=['MACCS', 'SMILES', 'ECFP'])
Fisicoquimicas = df_nuevasvariables[['LogP_scaled', 'TPSA_scaled',
       'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
       'Dobles_Enlaces_scaled']]
ECFP = df_nuevasvariables.iloc[:, 6:2054]
#print(ECFP)
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = df_nuevasvariables["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_ECFP = entrenar_svm(X,y)


