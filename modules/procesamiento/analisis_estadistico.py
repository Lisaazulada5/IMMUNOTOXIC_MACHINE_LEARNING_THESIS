import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# modules/procesamiento/analisis_estadistico.py

def detectar_outliers(df, columna):
    print("DataFrame recibido en detectar_outliers:")
    print(df.head())  # Verifica que llega correctamente
    Q1 = df[columna].quantile(0.25)
    Q3 = df[columna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[columna] < lower_bound) | (df[columna] > upper_bound)]
    return outliers

import pandas as pd
from scipy import stats

"""
CALCULO DE SESGO Y CURTOSIS PARA NORMALIDAD
"""
def calcular_sesgo_curtosis(df, columnas):
    """
    Calcula el sesgo y la curtosis para las columnas especificadas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        columnas (list): Una lista con los nombres de las columnas a analizar.

    Returns:
        pd.DataFrame: Un DataFrame con el sesgo y la curtosis para cada columna.
                     Devuelve un DataFrame vacío si hay errores o columnas inexistentes
    """
    try:
        resultados = []
        for columna in columnas:
            if columna not in df.columns:
                print(f"Advertencia: La columna '{columna}' no existe en el DataFrame.")
                return pd.DataFrame()
            sesgo = stats.skew(df[columna].dropna()) #Manejo de valores nulos
            curtosis = stats.kurtosis(df[columna].dropna()) #Manejo de valores nulos
            resultados.append({'Columna': columna, 'Sesgo': sesgo, 'Curtosis': curtosis})
        return pd.DataFrame(resultados)
    except Exception as e:
        print(f"Error al calcular sesgo y curtosis: {e}")
        return pd.DataFrame()

"""
CODIGO PARA OBTENER LA MEDIANA DE LOS DATOS
"""

import pandas as pd

import pandas as pd

import pandas as pd


def obtener_mediana(df, columnas):
    """
    Calcula la mediana de las columnas especificadas en un DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame con los datos.
    columnas (list): Lista de columnas sobre las que calcular la mediana.

    Returns:
    pd.Series: Mediana de cada columna.
    """
    # Filtrar el DataFrame solo con las columnas especificadas
    df_filtrado = df[columnas]

    # Calcular la mediana de cada columna
    mediana = df_filtrado.median()
    return mediana


def obtener_media(df, columnas):
    """
    Calcula la media de las columnas especificadas en un DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame con los datos.
    columnas (list): Lista de columnas sobre las que calcular la media.

    Returns:
    pd.Series: Media de cada columna.
    """
    # Filtrar el DataFrame solo con las columnas especificadas
    df_filtrado = df[columnas]

    # Calcular la mediana de cada columna
    media = df_filtrado.mean()
    return media

"""
CALCULO DEL IQR
"""

import pandas as pd


def calcular_iqr(df, columnas):
    """
    Calcula el Rango Intercuartílico (IQR) de las columnas especificadas en un DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame con los datos.
    columnas (list): Lista de columnas sobre las que calcular el IQR.

    Returns:
    pd.Series: IQR de cada columna.
    """
    # Filtrar el DataFrame solo con las columnas especificadas
    df_filtrado = df[columnas]

    # Calcular el primer cuartil (Q1) y el tercer cuartil (Q3)
    Q1 = df_filtrado.quantile(0.25)
    Q3 = df_filtrado.quantile(0.75)

    # Calcular el IQR como la diferencia entre Q3 y Q1
    IQR = Q3 - Q1
    return IQR

"""
DESCRIPCION ESTADISTICA DE LAS VARIABLES
"""

def obtener_estadisticos(df, columnas):
    """
    Calcula estadisticos de las columnas especificadas en un DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame con los datos.
    columnas (list): Lista de columnas sobre las que calcular  estadisticos.

    Returns:
    pd.Series: estadisticos de cada columna.
    """
    # Filtrar el DataFrame solo con las columnas especificadas
    df_filtrado = df[columnas]

    # Calcular la mediana de cada columna
    estadisticos = df_filtrado.describe()
    return estadisticos


import pandas as pd

"""
GENERAR TABLAS DE FRECUENCIAS
"""

def generar_tabla_frecuencias(df, columna):
    """
    Genera una tabla de frecuencias para una variable categórica.

    Parameters:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        columna (str): Nombre de la columna categórica para la cual se generará la tabla de frecuencias.

    Returns:
        pandas.DataFrame: Tabla de frecuencias con columnas para la categoría y la frecuencia.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Crear la tabla de frecuencias
    tabla_frecuencias = df[columna].value_counts().reset_index()
    tabla_frecuencias.columns = [columna, 'Frecuencia']

    return tabla_frecuencias

"""
FRECUENCIAS ABSOLUTAS Y RELATIVAS
"""
import pandas as pd


def generar_tabla_frecuencias(df, columna):
    """
    Genera una tabla de frecuencias absolutas y relativas para una variable categórica.

    Parameters:
        df (pandas.DataFrame): DataFrame que contiene los datos.
        columna (str): Nombre de la columna categórica para la cual se generará la tabla de frecuencias.

    Returns:
        pandas.DataFrame: Tabla de frecuencias con columnas para la categoría, frecuencia absoluta y relativa.
    """
    if columna not in df.columns:
        raise ValueError(f"La columna '{columna}' no existe en el DataFrame.")

    # Crear la tabla de frecuencias absolutas
    frecuencias_absolutas = df[columna].value_counts()

    # Calcular frecuencias relativas
    frecuencias_relativas = df[columna].value_counts(normalize=True)

    # Combinar las frecuencias en un solo DataFrame
    tabla_frecuencias = pd.DataFrame({
        columna: frecuencias_absolutas.index,
        'Frecuencia Absoluta': frecuencias_absolutas.values,
        'Frecuencia Relativa': frecuencias_relativas.values
    })

    return tabla_frecuencias

"""
ANALISIS MULTIVARIADO
"""
#MATRIZ DE CORRELACIÓN

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calcular_matriz_correlacion(df, columnas, metodo='pearson', guardar_figura=False,
                                nombre_figura="heatmap_correlacion.png"):
    """
    Calcula y visualiza la matriz de correlación de columnas específicas de un DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        columnas (list): Lista de columnas específicas para incluir en la matriz de correlación.
        metodo (str): Método de correlación ('pearson', 'spearman', 'kendall'). Por defecto 'pearson'.
        guardar_figura (bool): Si True, guarda la figura como archivo PNG.
        nombre_figura (str): Nombre del archivo si se guarda el heatmap.

    Returns:
        pd.DataFrame: Matriz de correlación calculada.
    """
    # Seleccionar solo las columnas especificadas
    df_seleccionado = df[columnas]

    # Calcular la matriz de correlación
    matriz_correlacion = df_seleccionado.corr(method=metodo)
    print(matriz_correlacion)

    # Visualizar la matriz de correlación con un heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, square=True)
    plt.title(f"Matriz de Correlación ({metodo.capitalize()})", fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Guardar la figura si es necesario
    if guardar_figura:
        plt.savefig(nombre_figura)
        print(f"Figura guardada como: {nombre_figura}")

    # Mostrar el heatmap
    plt.show()

    return matriz_correlacion

"""
Código para describe con Mediana e IQR
"""
import pandas as pd

def obtener_estadisticas_por_grupo(df, columna_grupo, columnas_numericas):
    """
    Genera estadísticas descriptivas por grupo, incluyendo mediana e IQR.

    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        columna_grupo (str): La columna que define los grupos (ej. 'activo' o 'inactivo').
        columnas_numericas (list): Lista de columnas numéricas a incluir en el análisis.

    Returns:
        pd.DataFrame: DataFrame con estadísticas descriptivas para cada grupo.
    """
    resultados = []

    # Iterar sobre cada grupo en la columna de agrupación
    for grupo, datos in df.groupby(columna_grupo):
        estadisticas = datos[columnas_numericas].describe().transpose()
        estadisticas['median'] = datos[columnas_numericas].median()
        estadisticas['IQR'] = datos[columnas_numericas].quantile(0.75) - datos[columnas_numericas].quantile(0.25)
        estadisticas['group'] = grupo  # Añadir el nombre del grupo para referencia
        resultados.append(estadisticas)

    # Combinar los resultados en un único DataFrame
    return pd.concat(resultados)

# Ejemplo de uso
if __name__ == "__main__":
    # Supongamos que tienes un DataFrame llamado 'df'
    # con una columna 'etiqueta' y varias columnas numéricas
    columnas_numericas = ['columna1', 'columna2', 'columna3']  # Reemplaza con tus columnas numéricas
    columna_grupo = 'etiqueta'  # Columna que contiene "activos" e "inactivos"

    estadisticas_por_grupo = obtener_estadisticas_por_grupo(df, columna_grupo, columnas_numericas)

    # Mostrar las estadísticas por grupo
    print(estadisticas_por_grupo)

"""
Calcular correlación
"""
from scipy.stats import pearsonr
import pandas as pd

from scipy.stats import pearsonr
import pandas as pd
import itertools

def calcular_correlacion(df, columnas):
    """
    Calcula la correlación de Pearson y el valor p entre todas las combinaciones de columnas proporcionadas
    de un DataFrame y devuelve una tabla con los resultados.

    Args:
        df (pd.DataFrame): DataFrame que contiene los datos.
        columnas (list): Lista de nombres de columnas a comparar.

    Returns:
        pd.DataFrame: DataFrame con la relación entre las columnas, el coeficiente de Pearson y el valor p.
    """
    # Crear lista para almacenar los resultados
    resultados = []

    # Obtener todas las combinaciones posibles de columnas
    combinaciones = itertools.combinations(columnas, 2)

    # Calcular la correlación de Pearson para cada par de columnas
    for col1, col2 in combinaciones:
        r, p_value = pearsonr(df[col1], df[col2])
        resultados.append([col1, col2, r, p_value])

    # Convertir los resultados en un DataFrame
    df_resultados = pd.DataFrame(resultados, columns=['Variable 1', 'Variable 2', 'Coeficiente de Pearson', 'Valor P'])

    return df_resultados


"""
Pueba de Mann Withney para evaluar la mediana de dos grupos de datos.
"""
# analisis_estadistico.py

import numpy as np
from scipy import stats
# analisis_estadistico.py

import pandas as pd
from scipy import stats

def prueba_mann_whitney_df(df, columna_grupo, columnas_numericas, alternativa='two-sided'):
    """
    Realiza la prueba U de Mann-Whitney para cada columna numérica en el DataFrame,
    comparando dos grupos definidos por la columna categórica.

    Parámetros:
    df (pd.DataFrame): DataFrame que contiene los datos.
    columna_grupo (str): Nombre de la columna categórica que define los grupos.
    columnas_numericas (list): Lista de nombres de columnas numéricas a analizar.
    alternativa (str): Define la hipótesis alternativa ('two-sided', 'less' o 'greater').

    Retorna:
    pd.DataFrame: DataFrame con los resultados de la prueba para cada columna numérica.
    """
    resultados = []

    # Obtener los nombres de los grupos únicos
    grupos = df[columna_grupo].dropna().unique()
    if len(grupos) != 2:
        raise ValueError("La columna de grupo debe contener exactamente dos grupos distintos.")

    grupo1, grupo2 = grupos

    for columna in columnas_numericas:
        # Filtrar los datos para cada grupo, omitiendo valores nulos
        datos_grupo1 = df[df[columna_grupo] == grupo1][columna].dropna()
        datos_grupo2 = df[df[columna_grupo] == grupo2][columna].dropna()

        # Verificar que ambos grupos tengan datos
        if datos_grupo1.empty or datos_grupo2.empty:
            raise ValueError(f"Una de las muestras para la columna '{columna}' está vacía.")

        # Realizar la prueba U de Mann-Whitney
        estadistico, p_valor = stats.mannwhitneyu(datos_grupo1, datos_grupo2, alternative=alternativa)

        # Almacenar los resultados
        resultados.append({
            'Columna': columna,
            'Grupo 1': grupo1,
            'Grupo 2': grupo2,
            'Estadístico U': estadistico,
            'Valor p': p_valor
        })

    # Convertir la lista de resultados en un DataFrame
    resultados_df = pd.DataFrame(resultados)
    return resultados_df


"""
ANALISIS DE MULTICOLINEALIDAD
"""
# analisis_estadistico.py
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


def calcular_vif(df, columnas):
    """
    Calcula el Factor de Inflación de Varianza (VIF) para evaluar la multicolinealidad.
    :param df: DataFrame con los datos.
    :param columnas: Lista de nombres de columnas a analizar.
    :return: DataFrame con los valores de VIF.
    """
    X = df[columnas]
    X = sm.add_constant(X)  # Agregar constante para la regresión

    vif_data = pd.DataFrame()
    vif_data["Variable"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return vif_data[vif_data["Variable"] != "const"]  # Excluir la constante

"""
Analisis de componentes principales
"""
import pandas as pd

def calcular_matriz_covarianza(df, columnas_numericas):
    """
    Calcula la matriz de covarianza de un DataFrame para las columnas seleccionadas.

    :param df: DataFrame de pandas con los datos.
    :param columnas_numericas: Lista de nombres de columnas numéricas.
    :return: DataFrame con la matriz de covarianza.
    """
    return df[columnas_numericas].cov()


import numpy as np


def Puntos_influentes_Cook(modelo, train_data_scaled):
    """
    Calcula e imprime los puntos influyentes basados en la distancia de Cook.

    Parámetros:
    - modelo: modelo ya ajustado que implementa el método get_influence().
    - train_data_scaled: conjunto de datos de entrenamiento escalados (se utiliza para obtener el tamaño muestral).

    Retorna:
    - puntos_influyentes: array con los índices de las observaciones influyentes.
    """
    # Obtener la influencia del modelo
    influencia = modelo.get_influence()

    # Extraer la distancia de Cook (se asume que cooks_distance es una tupla o arreglo, donde el primero contiene los valores)
    cooks_d = influencia.cooks_distance[0]

    # Calcular umbral típico para la distancia de Cook
    umbral_cook = 4 / len(train_data_scaled)

    # Identificar observaciones influyentes
    puntos_influyentes = np.where(cooks_d > umbral_cook)[0]

    print("Puntos influyentes según Cook:", puntos_influyentes)

    return puntos_influyentes

"""
Analisis de supuesto de linealidad de LOGIT
"""
# analisis_estadistico.py

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt


def calcular_linealidad(df, X_columns, target, degree=2, plot=False):
    """
    Calcula la linealidad en un modelo de regresión logística a partir de términos polinomiales.

    Se generan términos cuadráticos (u otros de grado especificado) para las variables predictoras,
    se ajusta un modelo de regresión logística y se calculan los log-odds de la probabilidad predicha
    para la clase positiva.

    Parámetros:
        - train_data_scaled: Conjunto de datos escalados (array o DataFrame) que contiene las variables predictoras.
        - train_data: DataFrame original que contiene la variable respuesta.
        - X_columns: Lista con los nombres de las columnas predictoras.
        - degree: Grado de la transformación polinómica (por defecto 2, para términos cuadráticos).
        - plot: Booleano que indica si se debe generar una gráfica de los log-odds (por defecto False).

    Retorna:
        - log_odds: Array de log-odds calculados para cada observación.
        - model: Modelo de regresión logística ajustado.
        - X_poly: Datos transformados que incluyen los términos polinomiales.
    """
    # Generar términos polinomiales (por defecto cuadráticos)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(df[X_columns])

    # Variable de respuesta
    y = df[target]

    # Ajustar el modelo de regresión logística
    model = LogisticRegression()
    model.fit(X_poly, y)

    # Predecir las probabilidades para la clase positiva (1)
    probas = model.predict_proba(X_poly)[:, 1]

    # Calcular los log-odds: log(p/(1-p))
    log_odds = np.log(probas / (1 - probas))
    #print('Log Odds')
    #print('________________')
    #print(log_odds)
    """
    # Si se solicita, generar la gráfica de log-odds
    if plot:
        plt.figure(figsize=(8, 5))
        plt.plot(log_odds, 'o', alpha=0.7)
        plt.xlabel("Índice de observación")
        plt.ylabel("Log-odds")
        plt.title("Log-odds de la clase positiva")
        plt.show()
    """
    return log_odds, model, X_poly



