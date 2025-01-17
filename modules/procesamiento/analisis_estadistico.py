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

