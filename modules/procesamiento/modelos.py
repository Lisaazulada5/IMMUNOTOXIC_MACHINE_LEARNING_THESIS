import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np
import statsmodels.api as sm
"""
Regresión  multiple
"""


def modelo_regresion(df, X_columns, Y_column, significance_level=0.05): #Esta función aplica la eliminación hacia atrás para seleccionar las variables más significativas en un modelo de regresión.
    X = df[X_columns].values #Extrae los valores de las columnas independientes (variables predictoras) del DataFrame.
    Y = df[Y_column].values #Extrae los valores de la columna dependiente (variable objetivo) del DataFrame.

    X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1) #Agrega una columna de unos al inicio de la matriz X para incluir el término de intercepto en la regresión.
    X_opt = X.copy()
    regressor_OLS = sm.OLS(Y, X_opt).fit() #Crea y ajusta un modelo de regresión lineal usando mínimos cuadrados ordinarios (OLS)

    return regressor_OLS #Devuelve el modelo final después de eliminar las variables no significativas

"""
Regresión logistica
"""

# modelos.py
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


def regresion_logistica(df, columnas, target):
    """
    Función para realizar regresión logística.

    :param df: DataFrame con los datos.
    :param columnas: Lista de nombres de columnas para las variables predictoras.
    :param target: Nombre de la columna de la variable dependiente.
    :return: Modelo ajustado y resultados de predicción.
    """
    X = df[columnas]
    y = df[target]

    # Dividir los datos en conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Agregar la constante para el intercepto en el modelo
    X_train = sm.add_constant(X_train)
    X_test = sm.add_constant(X_test)

    # Crear el modelo de regresión logística
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    # Predicciones
    y_pred = log_reg.predict(X_test)

    # Evaluación del modelo
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Mostrar los resultados
    print(f"Precisión del modelo: {accuracy:.4f}")
    print("Matriz de confusión:")
    print(cm)
    print("Reporte de clasificación:")
    print(report)

    return log_reg, accuracy, cm, report

"""
MODELO DE ÁRBOLES DE DECISION
"""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np


def arbol_decision(df, columnas_predictoras, target, criterio='gini', max_depth=None, min_samples_split=2,
                   min_samples_leaf=1):
    """
    Entrena un modelo de Árbol de Decisión y devuelve su rendimiento.

    Parámetros:
    - df: DataFrame con los datos.
    - columnas_predictoras: Lista con los nombres de las columnas predictoras.
    - target: Nombre de la columna objetivo.
    - criterio: 'gini' o 'entropy' para la función de evaluación del árbol.
    - max_depth: Profundidad máxima del árbol (None para sin restricción).
    - min_samples_split: Mínimo de muestras para dividir un nodo.
    - min_samples_leaf: Mínimo de muestras en una hoja.

    Retorna:
    - modelo: El modelo entrenado.
    - accuracy: Precisión del modelo.
    - cm: Matriz de confusión.
    - report: Reporte de clasificación.
    """
    X = df[columnas_predictoras]
    y = df[target]

    modelo = DecisionTreeClassifier(criterion=criterio, max_depth=max_depth,
                                    min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf,
                                    random_state=42)

    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    accuracy = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred)

    return modelo, accuracy, cm, report
