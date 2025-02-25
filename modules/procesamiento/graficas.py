import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
import seaborn as sns


def graficar_y_guardar_variable_continua(df, columna, ruta_guardado="grafica.jpg"):
    """
    Genera un histograma y un boxplot para analizar una variable continua y guarda la figura como archivo JPG.

    Parámetros:
    - df: DataFrame que contiene los datos.
    - columna: Nombre de la columna que se desea analizar (str).
    - ruta_guardado: Ruta donde se guardará el archivo JPG (str).
    """
    if columna not in df.columns:
        print(f"La columna '{columna}' no se encuentra en el DataFrame.")
        return

    # Crear figura con dos subgráficos
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [3, 1]})

    # Histograma
    sns.histplot(df[columna], kde=True, ax=axes[0], color="skyblue", bins=30)
    axes[0].set_title(f"Histograma de {columna}", fontsize=14)
    axes[0].set_xlabel(columna)
    axes[0].set_ylabel("Frecuencia")

    # Boxplot
    sns.boxplot(x=df[columna], ax=axes[1], color="lightgreen")
    axes[1].set_title(f"Boxplot de {columna}", fontsize=14)
    axes[1].set_xlabel(columna)

    # Ajustar la gráfica
    plt.tight_layout(pad=6.0)

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

    # Mostrar la gráfica
    plt.show()
    print(f"Gráfica guardada en {ruta_guardado}")

import seaborn as sns
import matplotlib.pyplot as plt

def graficar_y_guardar_boxplot(df, columna_x, columna_y, ruta_guardado="grafica.jpg"):
    """Genera     un     boxplot     para analizar una variable continua y una categórica
    y     guarda     la    figura    como     archivo    png.
    Parámetros:     - df: DataFrame     que     contiene     los     datos.
    - columna: Nombre     de     la     columna    que     se    desea     analizar(str).
    - ruta_guardado: Ruta     donde    se    guardará    el    archivo    JPG(str).
    """
    if columna_x not in df.columns:
        print(f"La columna '{columna}' no se encuentra en el DataFrame.")
        return

    fig, ax = plt.subplots(figsize=(10, 8)) # Crear figura con un solo gráfico
    #Boxplot

    sns.boxplot(data=df, x=columna_x, y=columna_y, palette='pastel')
    plt.title(f"Distribución de '{columna_y}' por Clasificación ATS")
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

"""
Violin plot: Similar al boxplot, pero mostrando la densidad de los datos.
"""
def graficar_y_guardar_violinplot(df, columna_x, columna_y, ruta_guardado="grafica.jpg"):

    fig, ax = plt.subplots(figsize=(10, 10))  # Crear figura con un solo gráfico

    sns.violinplot(data=df, x=columna_x, y=columna_y, palette='pastel', alpha=1)
    plt.title(f"Distribución de {columna_y} por Clasificación ATS (Violin Plot)")
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

"""
Histogramas: Graficar histogramas separados para cada grupo.
"""
def graficar_y_guardar_histograma(df, columna_x, columna_y, ruta_guardado="grafica.jpg"):
    fig, ax = plt.subplots(figsize=(10, 8))  # Crear figura con un solo gráfico
    sns.histplot(data=df, x=columna_y, hue=columna_x, kde=True, bins=20, palette='Set2')
    plt.title(f"Histograma de {columna_y} por Clasificación ATS")
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

"""
Graficas de frecuencia
"""
def graficar_y_guardar_barras(df, columna_x, ruta_guardado="grafica.jpg"):
    # Suponiendo que el DataFrame se llama df y la columna es "NumHAcceptors"
    frecuencias = df[columna_x].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 8))  # Crear figura con un solo gráfico
    frecuencias.plot(kind='bar', color='blue', alpha=0.7, edgecolor='black')
    plt.title(f"Gragico de barras de {columna_x}")
    plt.xlabel(f"{columna_x}")
    plt.ylabel("Frecuencia")
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

"""
GRAFICA DE NORMALIDAD
"""

import matplotlib.pyplot as plt
import scipy.stats as stats
import numpy as np

def graficar_normalidad(data, variable, ruta_guardado="grafica.jpg", bins=30):
    """
    Grafica un histograma y un gráfico Q-Q para evaluar la normalidad de una variable.

    Parámetros:
        data (pd.DataFrame): DataFrame que contiene la variable a analizar.
        variable (str): Nombre de la columna que contiene los datos.
        bins (int): Número de bins para el histograma (opcional, por defecto 30).

    Salida:
        Muestra un histograma y un gráfico Q-Q en una figura.
    """
    if variable not in data.columns:
        raise ValueError(f"La variable '{variable}' no está en el DataFrame.")

    valores = data[variable].dropna()

    # Crear la figura
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Histograma con la curva de densidad normal
    axs[0].hist(valores, bins=bins, density=True, alpha=0.6, color='skyblue', edgecolor='black')
    xmin, xmax = axs[0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, valores.mean(), valores.std())
    axs[0].plot(x, p, 'k', linewidth=2, label='Densidad Normal')
    axs[0].set_title('Histograma y Densidad Normal')
    axs[0].set_xlabel(variable)
    axs[0].set_ylabel('Densidad')
    axs[0].legend()

    # Gráfico Q-Q
    stats.probplot(valores, dist="norm", plot=axs[1])
    axs[1].set_title('Gráfico Q-Q')

    # Ajustar diseño
    plt.tight_layout()
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)


import matplotlib.pyplot as plt
import scipy.stats as stats

"""
GRAFICAS Q-Q PLOT PARA DIFERENTES VARIABLES
"""

def graficar_qq_multivariable(df, columnas, ruta_guardado="grafica.jpg", titulo="Q-Q Plots"):
    """
    Genera Q-Q Plots para múltiples variables en un solo gráfico.

    Parámetros:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de las columnas a graficar.
        titulo (str): Título del gráfico general.
    """
    num_vars = len(columnas)
    filas = (num_vars // 3) + (1 if num_vars % 3 != 0 else 0)  # Calcula las filas necesarias (3 gráficos por fila)
    fig, axes = plt.subplots(filas, 3, figsize=(15, 5 * filas))
    fig.suptitle(titulo, fontsize=16)

    # Asegurarse de que `axes` sea una matriz, incluso si es solo una fila
    if filas == 1:
        axes = [axes]

    for i, columna in enumerate(columnas):
        fila, col = divmod(i, 3)  # Posición en la cuadrícula
        ax = axes[fila][col] if filas > 1 else axes[col]
        stats.probplot(df[columna], dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {columna}", fontsize=12, pad=10)
        ax.grid(True)

    # Ocultar gráficos vacíos si hay menos de 3 gráficos en la última fila
    for j in range(i + 1, filas * 3):
        fila, col = divmod(j, 3)
        ax = axes[fila][col] if filas > 1 else axes[col]
        ax.axis("off")

    plt.subplots_adjust(hspace=10, wspace=1)
    plt.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    plt.show()

    # Guardar la figura como archivo JPG
    fig.savefig(ruta_guardado, format="png", dpi=300)

"""
ANALISIS MULTIVARIADO
"""
#Código para Scatter Plots

import matplotlib.pyplot as plt
def generar_scatterplots(df, columnas_x, columnas_y, guardar_figuras=False, carpeta_figuras="scatterplots"):
    """
    Genera scatter plots entre pares de columnas especificadas del DataFrame.

    Args:
        df (pd.DataFrame): El DataFrame con los datos.
        columnas_x (list): Lista de columnas para el eje X.
        columnas_y (list): Lista de columnas para el eje Y.
        guardar_figuras (bool): Si True, guarda las figuras como archivos PNG.
        carpeta_figuras (str): Carpeta donde se guardarán las figuras.

    Returns:
        None
    """
    import os

    # Crear carpeta para guardar las figuras, si no existe
    if guardar_figuras:
        os.makedirs(carpeta_figuras, exist_ok=True)

    # Generar scatter plots para cada combinación de columnas
    for x in columnas_x:
        for y in columnas_y:
            if x != y:  # Evitar scatter plot de la misma variable
                plt.figure(figsize=(8, 6))
                sns.scatterplot(data=df, x=x, y=y, alpha=0.7, edgecolor=None)

                # Configurar el título y etiquetas
                plt.title(f"Scatter Plot: {x} vs {y}", fontsize=14, pad=15)
                plt.xlabel(x, fontsize=12)
                plt.ylabel(y, fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.7)

                # Guardar la figura si es necesario
                if guardar_figuras:
                    figura_nombre = f"{carpeta_figuras}/scatter_{x}_vs_{y}.png"
                    plt.savefig(figura_nombre)
                    print(f"Figura guardada como: {figura_nombre}")

                # Mostrar el scatter plot
                plt.show()

#MATRIZ DE SCATTERPLOT

import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt


def generar_pairplot(df, columnas, ruta_guardado = "grafica.png", hue=None):
    """
    Genera una matriz de scatterplots (pairplot) para columnas específicas del DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con los datos.
        columnas (list): Lista de nombres de columnas a incluir en el pairplot.
        hue (str, optional): Columna categórica para colorear los puntos.

    Returns:
        sns.PairGrid: Objeto PairGrid de seaborn que contiene la matriz de scatterplots.
    """
    # Filtrar las columnas seleccionadas
    df_seleccion = df[columnas]

    # Agregar la columna hue si está definida
    if hue and hue in df.columns:
        df_seleccion[hue] = df[hue]

    # Crear el pairplot
    sns.set_theme(style="ticks")
    pairplot = sns.pairplot(df_seleccion, hue=hue, height = 2)

    for ax in pairplot.axes.flat:
        # Rotar etiquetas del eje X
        ax.tick_params(axis='x', labelsize=8, rotation=0)

        # Obtener el título del eje Y y rotarlo
        ax.set_ylabel(ax.get_ylabel(), rotation=45, fontsize=8, labelpad=20)  # Rotar y ajustar padding
        # Obtener el título del eje X y rotarlo
        ax.set_xlabel(ax.get_xlabel(), rotation=45, fontsize=8, labelpad=20)  # Rotar y ajustar padding

    # Agregar valores de correlación
    for i, j in zip(*np.triu_indices_from(pairplot.axes, 1)):  # Solo parte superior de la matriz
        if i < len(columnas) and j < len(columnas):  # Evitar índices fuera de rango
            x_col = columnas[j]
            y_col = columnas[i]
            # Calcular correlación de Pearson
            corr_coef = df[[x_col, y_col]].corr().iloc[0, 1]
            # Agregar la correlación al gráfico
            pairplot.axes[i, j].annotate(
                f"r = {corr_coef:.2f}",
                xy=(0.1, 0.9),
                xycoords="axes fraction",
                ha="left",
                va="center",
                fontsize=10,
                color="red",
            )

    # Mostrar el plot
    plt.show()

    # Guardar la figura como archivo JPG
    pairplot.savefig(ruta_guardado, format="png", dpi=300)


def calcular_matriz_correlacion(df, columnas,ruta_guardado= "grafica.png", metodo='pearson', guardar_figura=False,
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


    # Guardar la figura como archivo JPG
    plt.savefig(ruta_guardado, format="png", dpi=300)
    # Mostrar el heatmap
    plt.show()

"""
GRAFICO DE DISPERSION TENIENDO EN CUENTA LA CATEGORIA
"""

import seaborn as sns
import matplotlib.pyplot as plt


def generar_pairplot_CATEOGRIAS(df, columnas, categoria, ruta_guardado):
    """
    Genera un pairplot con las columnas seleccionadas y una variable categórica para diferenciar los datos.

    Parámetros:
    df (DataFrame): DataFrame con los datos.
    columnas (list): Lista de columnas numéricas a graficar.
    categoria (str): Nombre de la columna categórica para diferenciar los datos.
    """
    sns.set_theme(style="ticks")

    # Filtrar el DataFrame con las columnas seleccionadas
    df_filtrado = df[columnas + [categoria]]

    # Crear el pairplot
    pairplot = sns.pairplot(df_filtrado, hue=categoria)

    for ax in pairplot.axes.flat:
        # Rotar etiquetas del eje X
        ax.tick_params(axis='x', labelsize=8, rotation=0)
        # Obtener el título del eje Y y rotarlo
        ax.set_ylabel(ax.get_ylabel(), rotation=45, fontsize=8, labelpad=20)  # Rotar y ajustar padding
        # Obtener el título del eje X y rotarlo
        ax.set_xlabel(ax.get_xlabel(), rotation=0, fontsize=8, labelpad=20)  # Rotar y ajustar padding

    # Guardar la figura como archivo JPG
    plt.savefig(ruta_guardado, format="png", dpi=300)
    # Mostrar la gráfica
    plt.show()

"""
Matriz de covarianza
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graficar_matriz_covarianza(df, columnas_numericas):
    """
    Calcula y grafica la matriz de covarianza de un DataFrame para las columnas seleccionadas.

    :param df: DataFrame de pandas con los datos.
    :param columnas_numericas: Lista de nombres de columnas numéricas.
    """
    cov_matrix = df[columnas_numericas].cov()  # Calcula la matriz de covarianza

    plt.figure(figsize=(8, 6))  # Tamaño de la figura
    sns.heatmap(cov_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)

    plt.title("Matriz de Covarianza")
    plt.show()

# modules/procesamiento/graficas.py
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

def graficar_curva_roc(y_test, y_pred_proba, ruta_guardado, titulo="Curva ROC para Regresión Logística"):
    """
    Grafica la Curva ROC y calcula el AUC.

    :param y_test: Etiquetas verdaderas del conjunto de prueba.
    :param y_pred_proba: Probabilidades predichas para la clase positiva.
    :param titulo: Título de la gráfica (opcional).
    """
    # Calcular la curva ROC y el AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)

    # Graficar la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Línea diagonal para referencia
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title(titulo)
    plt.legend(loc='lower right')


    # Guardar la figura como archivo JPG
    plt.savefig(ruta_guardado, format="png", dpi=300)
    plt.show()

    # Imprimir el valor del AUC
    print(f'AUC: {auc:.2f}')


# graficas.py

import matplotlib.pyplot as plt


def graficar_cook(modelo, train_data_scaled, ruta_guardado):
    """
    Genera y muestra una gráfica de la distancia de Cook para cada observación.

    Parámetros:
    - modelo: modelo ya ajustado que implementa el método get_influence().
    - train_data_scaled: conjunto de datos de entrenamiento escalados (para determinar el umbral).
    """
    # Obtener la influencia del modelo
    influencia = modelo.get_influence()

    # Extraer la distancia de Cook (se asume que cooks_distance es una tupla y que el primer elemento contiene los valores)
    cooks_d = influencia.cooks_distance[0]

    # Calcular umbral típico para la distancia de Cook
    umbral_cook = 4 / len(train_data_scaled)

    # Crear la gráfica
    plt.figure(figsize=(8, 5))
    plt.stem(range(len(cooks_d)), cooks_d, markerfmt="ro")
    plt.axhline(umbral_cook, color="r", linestyle="dashed", label="Umbral 4/n")
    plt.xlabel("Índice de observación")
    plt.ylabel("Distancia de Cook")
    plt.title("Distancia de Cook por observación")
    plt.legend()
    # Guardar la figura como archivo JPG
    plt.savefig(ruta_guardado, format="png", dpi=300)
    plt.show()

import matplotlib.pyplot as plt
def graficar_supeustolinealidad(variable,X,log_odds, ruta_guardado):
    """
    Genera y muestra una gráfica para verificar el supuesto de linealidad de cada variable

    Parámetros:
    - modelo: modelo ya ajustado que implementa el método get_influence().
    - variable: variable a graficar
    -Log_odds: calculado anteriormente
    """
    # Supongamos que quieres graficar una variable frente a los log-odds
    plt.scatter(X, log_odds)
    plt.xlabel(variable)
    plt.ylabel('log_odds')
    plt.title(f"Log-Odds vs. {variable}")
    # Guardar la figura como archivo JPG
    plt.savefig(ruta_guardado, format="png", dpi=300)
    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix(cm, ruta_guardado, class_names=None, cmap="Blues"):
    """
    Grafica una matriz de confusión con Seaborn y Matplotlib.

    Parámetros:
    cm (array-like): Matriz de confusión generada por sklearn.metrics.confusion_matrix().
    class_names (list, opcional): Lista con los nombres de las clases. Si es None, usa índices numéricos.
    cmap (str, opcional): Colormap para el heatmap. Ejemplo: "Blues", "Reds", "Greens".

    Retorna:
    None (muestra la gráfica).
    """
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap=cmap, cbar=False, linewidths=1,
                linecolor="black", xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicción")
    plt.ylabel("Valor Real")
    plt.title("Matriz de Confusión")
    plt.savefig(ruta_guardado, format="png", dpi=300)
    plt.show()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos):
    """
    Función para graficar las curvas ROC de varios modelos con diferentes datos de entrada y salidas.

    Parámetros:
    - modelos: lista de modelos entrenados.
    - X_test_list: lista de conjuntos de características de prueba (uno para cada modelo).
    - y_test_list: lista de conjuntos de etiquetas de prueba (uno para cada modelo).
    - etiquetas_modelos: lista de etiquetas para los modelos.
    """
    # Crear la figura del gráfico
    plt.figure(figsize=(10, 8))

    # Graficar la curva ROC para cada modelo
    for modelo, X_test, y_test, etiqueta in zip(modelos, X_test_list, y_test_list, etiquetas_modelos):
        # Obtener las probabilidades para la clase positiva (1)
        y_prob = X_test

        # Calcular las tasas de falsos positivos (FPR) y verdaderos positivos (TPR)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Calcular el AUC
        roc_auc = auc(fpr, tpr)

        # Graficar la curva ROC
        plt.plot(fpr, tpr, label=f'{etiqueta} (AUC = {roc_auc:.2f})')

    # Graficar la línea de aleatoriedad (diagonal)
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    # Añadir título, etiquetas y leyenda
    plt.title('Curvas ROC de Modelos')
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.legend(loc='lower right')
    plt.grid(True)

    # Mostrar el gráfico
    plt.show()



