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
