import os
import pandas as pd
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.modelos import entrenar_random_forest
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

"""
# Definir las columnas predictoras y la variable objetivo
train_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_fingerprints.csv')

#MODELADO DE BOSQUES ALEATORIOS

#Escoger los mejores hiperparámetros para los modelos


print('***************************')
print('MODELO BOSQUES ALEATORIOS MACCS')
print('______________________________  ')

from modules.procesamiento.modelos import entrenar_random_forest
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS, confusion_matrix, y_pred_prob_MACCS, y_test_MACCS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
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
print('MODELO BOSQUES ALEATORIOS MACCS y FISICOQUIMICOS')
print('______________________________  ')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columnas_predictoras = pd.concat([Fisicoquimicas, MACCS], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_MACCS_FISICOQUIMICAS, y_test_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP')
print('______________________________  ')
ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
#columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP, confusion_matrix, y_pred_prob_ECFP, y_test_ECFP = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([Fisicoquimicas, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_FISICOQUIMICAS, y_test_ECFP_FISICOQUIMICAS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y MACCS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS, confusion_matrix,  y_pred_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP, MACCS Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP, Fisicoquimicas], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS, y_test_ECFP_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [randomMACCS, randomMACCS_FISICOQUIMICOS, randomECFP, randomECFP_FISICOQUIMICOS, randomECFP_MACCS, randomECFP_MACCS_FISICOQUIMICOS]
X_test_list = [y_pred_prob_MACCS, y_pred_prob_MACCS_FISICOQUIMICAS, y_pred_prob_ECFP, y_pred_prob_ECFP_FISICOQUIMICAS, y_pred_prob_ECFP_MACCS, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICAS, y_test_ECFP, y_test_ECFP_FISICOQUIMICAS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICAS]
etiquetas_modelos = ["Modelo MACCS_Bosques",	" Modelo MACCS + FISICOQUIMICAS_Bosques",	" Modelo ECFP_Bosques",	" Modelo ECFP + FISICOQUIMICAS_Bosques",	" Modelo ECFP + MACCS_Bosques",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Bosques"]
graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)
"""



"""
#ARBOLES CON DATASET BALANCEADO
"""
print('*******************************')
print('ARBOLES CON DATASET BALANCEADO')
print('*******************************')

df_balanceado = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/df_balanceado_escalado_nuevo.csv')
train_data_fingerprints = df_balanceado.copy()

train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS, confusion_matrix, y_pred_prob_MACCS, y_test_MACCS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
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
print('MODELO BOSQUES ALEATORIOS MACCS y FISICOQUIMICOS')
print('______________________________  ')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columnas_predictoras = pd.concat([Fisicoquimicas, MACCS], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_MACCS_FISICOQUIMICAS, y_test_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP')
print('______________________________  ')
ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
#columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP, confusion_matrix, y_pred_prob_ECFP, y_test_ECFP = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([Fisicoquimicas, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_FISICOQUIMICAS, y_test_ECFP_FISICOQUIMICAS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y MACCS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS, confusion_matrix,  y_pred_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP, MACCS Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP, Fisicoquimicas], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS, y_test_ECFP_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

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



from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from modules.procesamiento.modelos import entrenar_modelo
param_grid = {
    'n_estimators': [100, 200, 300],  # Número de árboles
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0, 0.01, 0.1],
    'class_weight': ['balanced', None]
}
#modelo = RandomForestClassifier(random_state=42)  # ✅ Crear instancia
#entrenar_modelo(modelo, param_grid, X, y)










"""
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [randomMACCS, randomMACCS_FISICOQUIMICOS, randomECFP, randomECFP_FISICOQUIMICOS, randomECFP_MACCS, randomECFP_MACCS_FISICOQUIMICOS]
X_test_list = [y_pred_prob_MACCS, y_pred_prob_MACCS_FISICOQUIMICAS, y_pred_prob_ECFP, y_pred_prob_ECFP_FISICOQUIMICAS, y_pred_prob_ECFP_MACCS, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICAS, y_test_ECFP, y_test_ECFP_FISICOQUIMICAS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICAS]
etiquetas_modelos = ["Modelo MACCS_Bosques",	" Modelo MACCS + FISICOQUIMICAS_Bosques",	" Modelo ECFP_Bosques",	" Modelo ECFP + FISICOQUIMICAS_Bosques",	" Modelo ECFP + MACCS_Bosques",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Bosques"]
graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)



"""
#PROPIEDADES ELECTRONICAS DATASET DESBALANCEADO
"""

Propiedades_electrónicas = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/Propiedades_electrónicas.csv')
Propiedades_electrónicas_test = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/Propiedades_electrónicas_test.csv')
propiedades_electronicas_a_usar = ["PEOE_VSA2",	"SMR_VSA7",	"SMR_VSA9"]

from sklearn.preprocessing import StandardScaler

# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
Propiedades_electrónicas_scaled = Propiedades_electrónicas.copy()
Propiedades_electrónicas_test_scaled = Propiedades_electrónicas_test.copy()
# Aplicar el escalado a todas las columnas
Propiedades_electrónicas_scaled[:] = scaler.fit_transform(Propiedades_electrónicas_scaled)
Propiedades_electrónicas_test_scaled[:] = scaler.fit_transform(Propiedades_electrónicas_test_scaled)


train_data_scaled_electronic = pd.concat([train_data_fingerprints, Propiedades_electrónicas_scaled], axis=1)
train_data_scaled_electronic_test = pd.concat([test_data_fingerprints, Propiedades_electrónicas_test_scaled], axis=1)
"""
"""
#INCLUSION DE LAS PROPIEDADES ELECTRONICAS EN EL DATASET DESBALANCEADO
"""


print('***************************')
print('MODELO BOSQUES ALEATORIOS MACCS')
print('______________________________  ')

from modules.procesamiento.modelos import entrenar_random_forest

columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS, confusion_matrix, y_pred_prob_MACCS, y_test_MACCS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
#test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
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
print('MODELO BOSQUES ALEATORIOS MACCS y FISICOQUIMICOS')
print('______________________________  ')
Electronicas = train_data_scaled_electronic[propiedades_electronicas_a_usar]
columnas_predictoras = pd.concat([Fisicoquimicas,Electronicas,  MACCS], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_MACCS_FISICOQUIMICAS, y_test_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
Electronicas_TEST = train_data_scaled_electronic_test[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([Fisicoquimicas_test,Electronicas_TEST,  MACCS_test], axis=1)
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
print('MODELO BOSQUES ALEATORIOS ECFP')
print('______________________________  ')
ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
#columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP, confusion_matrix, y_pred_prob_ECFP, y_test_ECFP = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_FISICOQUIMICAS, y_test_ECFP_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP Y FISICOQUIMICOS')
print('--------------------------------')

columnas_predictoras = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
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
print('MODELO BOSQUES ALEATORIOS ECFP Y MACCS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS, confusion_matrix,  y_pred_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP, MACCS Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP, Fisicoquimicas, Electronicas], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS, y_test_ECFP_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP, MACCS Y FISICOQUIMICOS')
print('--------------------------------')

columnas_predictoras = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [randomMACCS, randomMACCS_FISICOQUIMICOS, randomECFP, randomECFP_FISICOQUIMICOS, randomECFP_MACCS, randomECFP_MACCS_FISICOQUIMICOS]
X_test_list = [y_pred_prob_MACCS, y_pred_prob_MACCS_FISICOQUIMICAS, y_pred_prob_ECFP, y_pred_prob_ECFP_FISICOQUIMICAS, y_pred_prob_ECFP_MACCS, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICAS, y_test_ECFP, y_test_ECFP_FISICOQUIMICAS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICAS]
etiquetas_modelos = ["Modelo MACCS_Electronicas_Bosques",	" Modelo MACCS + FISICOQUIMICAS_Electronicas_Bosques",
                     " Modelo ECFP_Electronicas_Bosques",	" Modelo ECFP + FISICOQUIMICAS_Electronicas_Bosques",	" Modelo ECFP + MACCS_Electronicas_Bosques",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Electronicas_Bosques"]
graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)
"""


"""
#DATOS BALANCEADOS CON PROPIEDADES ELECTRONICAS
"""

propiedades_electronicas_balanced = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_balanced_electronic.csv')
propiedades_electronicas_a_usar = ["PEOE_VSA2",	"SMR_VSA7",	"SMR_VSA9"]
"""
#ARBOLES CON DATASET BALANCEADO
"""
print('*******************************')
print('BOSQUES ALEATORIOS CON DATASET BALANCEADO Y VARIABLES ELECTRONICAS')
print('*******************************')

#train_data_fingerprints = propiedades_electronicas_balanced.copy()

#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS, confusion_matrix, y_pred_prob_MACCS, y_test_MACCS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = train_data_scaled_electronic_test
#test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2221]
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
print('MODELO BOSQUES ALEATORIOS MACCS y FISICOQUIMICOS')
print('______________________________  ')
Electronicas = propiedades_electronicas_balanced[propiedades_electronicas_a_usar]

columnas_predictoras = pd.concat([Fisicoquimicas, Electronicas, MACCS], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomMACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_MACCS_FISICOQUIMICAS, y_test_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
Electronicas_TEST = test_data_fingerprints[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, MACCS_test], axis=1)
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
print('MODELO BOSQUES ALEATORIOS ECFP')
print('______________________________  ')
ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
#columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP, confusion_matrix, y_pred_prob_ECFP, y_test_ECFP = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_FISICOQUIMICAS, y_test_ECFP_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP Y FISICOQUIMICOS')
print('--------------------------------')

columnas_predictoras = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
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
print('MODELO BOSQUES ALEATORIOS ECFP Y MACCS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS, confusion_matrix,  y_pred_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_random_forest(X, y)

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
print('MODELO BOSQUES ALEATORIOS ECFP, MACCS Y FISICOQUIMICOS')
print('______________________________  ')
columnas_predictoras = pd.concat([MACCS, ECFP, Fisicoquimicas, Electronicas], axis=1)
columnas_predictoras.columns =columnas_predictoras.columns.astype(str)
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target

randomECFP_MACCS_FISICOQUIMICOS, confusion_matrix, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS, y_test_ECFP_MACCS_FISICOQUIMICAS = entrenar_random_forest(X, y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba ECFP, MACCS Y FISICOQUIMICOS')
print('--------------------------------')

columnas_predictoras = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [randomMACCS, randomMACCS_FISICOQUIMICOS, randomECFP, randomECFP_FISICOQUIMICOS, randomECFP_MACCS, randomECFP_MACCS_FISICOQUIMICOS]
X_test_list = [y_pred_prob_MACCS, y_pred_prob_MACCS_FISICOQUIMICAS, y_pred_prob_ECFP, y_pred_prob_ECFP_FISICOQUIMICAS, y_pred_prob_ECFP_MACCS, y_pred_prob_ECFP_MACCS_FISICOQUIMICAS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICAS, y_test_ECFP, y_test_ECFP_FISICOQUIMICAS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICAS]
etiquetas_modelos = ["Modelo MACCS_Electronicas_Bosques_Balanceado",	" Modelo MACCS + FISICOQUIMICAS_Electronicas_Bosques_Balanceado",
                     " Modelo ECFP_Electronicas_Bosques_Balanceado",	" Modelo ECFP + FISICOQUIMICAS_Electronicas_Bosques_Balanceado",
                     " Modelo ECFP + MACCS_Electronicas_Bosques_Balanceado",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Electronicas_Bosques_Balanceado"]
graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)