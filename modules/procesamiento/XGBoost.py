import os
import pandas as pd
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.modelos import entrenar_xgboost
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


"""
# Definir las columnas predictoras y la variable objetivo
train_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_fingerprints.csv')

#MODELADO DE BOSQUES ALEATORIOS

#Escoger los mejores hiperparámetros para los modelos



#MODELO xgboost
"""

"""
"""
#MODELO USANDO MACCS
"""
print('\n ****************')
print('MODELOS xgboost')
print('****************')
from modules.procesamiento.modelos import entrenar_xgboost
print('\n----------------')
print('MODELOS xgboost MACCS')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS, cm, y_prob_MACCS, y_test_MACCS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba  MACCS ')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS_test)
predict_data = MACCS_test
nuevas_predicciones = xgboost_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_prediccionesMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_prediccionesMACCS']
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


print('\n----------------')
print('MODELOS xgboost MACCS y Fisicoquimicas')
print('----------------')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_Fisicoquimicas, cm, y_prob_MACCS_FISICOQUIMICOS, y_test_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')

Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_Fisicoquimicas.predict(predict_data)
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



print('\n----------------')
print('MODELOS xgboost ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP, cm, y_prob_ECFP, y_test_ECFP = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
nuevas_predicciones = xgboost_ECFP.predict(predict_data)
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


print('\n----------------')
print('MODELOS xgboost ECFP y Fisicoquimicos')
print('----------------')

columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP_Fisicoquimicos, cm, y_prob_ECFP_FISICOQUIMICOS, y_test_ECFP_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
nuevas_predicciones = xgboost_ECFP_Fisicoquimicos.predict(predict_data)
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS ')
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
xgboost_MACCS_ECFP, cm, y_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
MACCS_test = MACCS_test.add_prefix("MACCS_")
columns_subset_test = pd.concat([ECFP_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCS']
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
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
modelos = [xgboost_MACCS, xgboost_MACCS_Fisicoquimicas, xgboost_ECFP, xgboost_ECFP_Fisicoquimicos, xgboost_MACCS_ECFP, xgboost_MACCS_ECFP_fisicoquimicas]
X_test_list = [y_prob_MACCS, y_prob_MACCS_FISICOQUIMICOS, y_prob_ECFP, y_prob_ECFP_FISICOQUIMICOS, y_prob_ECFP_MACCS, y_prob_ECFP_MACCS_FISICOQUIMICOS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICOS, y_test_ECFP, y_test_ECFP_FISICOQUIMICOS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo_MACCS_XGBOOST",	"Modelo_MACCS_FISICOQUIMICOS_XGBOOST",	"Modelo_ECFP_XGBOOST",
                     "Modelo_ECFP_FISICOQUIMICOS_XGBOOST",	"Modelo_ECFP_MACCS_XGBOOST",	"Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST"]

graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)
"""



"""
#XGBOOST CON DATASET BALANCEADO
"""
print('*******************************')
print('XGBOOST CON DATASET BALANCEADO MACCS')
print('*******************************')

df_balanceado = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/df_balanceado_escalado_nuevo.csv')
train_data_fingerprints = df_balanceado.copy()
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS, cm, y_prob_MACCS, y_test_MACCS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba  MACCS ')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS_test)
predict_data = MACCS_test
nuevas_predicciones = xgboost_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_prediccionesMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_prediccionesMACCS']
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


print('\n----------------')
print('MODELOS xgboost MACCS y Fisicoquimicas')
print('----------------')

Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_Fisicoquimicas, cm, y_prob_MACCS_FISICOQUIMICOS, y_test_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')

Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_Fisicoquimicas.predict(predict_data)
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



print('\n----------------')
print('MODELOS xgboost ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP, cm, y_prob_ECFP, y_test_ECFP = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
nuevas_predicciones = xgboost_ECFP.predict(predict_data)
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


print('\n----------------')
print('MODELOS xgboost ECFP y Fisicoquimicos')
print('----------------')

columns_subset = pd.concat([Fisicoquimicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP_Fisicoquimicos, cm, y_prob_ECFP_FISICOQUIMICOS, y_test_ECFP_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
nuevas_predicciones = xgboost_ECFP_Fisicoquimicos.predict(predict_data)
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS ')
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
xgboost_MACCS_ECFP, cm, y_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
MACCS_test = MACCS_test.add_prefix("MACCS_")
columns_subset_test = pd.concat([ECFP_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCS']
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
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



import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from modules.procesamiento.modelos import entrenar_modelo
# Definir la grilla de hiperparámetros para XGBoost
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2],  # ✅ Tasa de aprendizaje
    'n_estimators': [50, 100, 200],  # ✅ Número de árboles
    'subsample': [0.7, 1.0],  # ✅ Submuestreo
    'colsample_bytree': [0.7, 1.0],  # ✅ Características por árbol
    'gamma': [0, 0.1, 0.2],  # ✅ Regularización
    #'scale_pos_weight': [1, 1.48]  # ✅ Para manejar clases desbalanceadas
}
# Crear instancia del modelo XGBoost
#modelo = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')

# Entrenar el modelo usando la función personalizada
#entrenar_modelo(modelo, param_grid, columnas_predictoras, target)












"""
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [xgboost_MACCS, xgboost_MACCS_Fisicoquimicas, xgboost_ECFP, xgboost_ECFP_Fisicoquimicos, xgboost_MACCS_ECFP, xgboost_MACCS_ECFP_fisicoquimicas]
X_test_list = [y_prob_MACCS, y_prob_MACCS_FISICOQUIMICOS, y_prob_ECFP, y_prob_ECFP_FISICOQUIMICOS, y_prob_ECFP_MACCS, y_prob_ECFP_MACCS_FISICOQUIMICOS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICOS, y_test_ECFP, y_test_ECFP_FISICOQUIMICOS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo_MACCS_XGBOOST_BALANCED",	"Modelo_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED",	"Modelo_ECFP_XGBOOST_BALANCED",
                     "Modelo_ECFP_FISICOQUIMICOS_XGBOOST_BALANCED",	"Modelo_ECFP_MACCS_XGBOOST_BALANCED",
                     "Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED"]

graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)



"""
#PROPIEDADES ELECTRONICAS DATASET DESBALANCEADO
"""

#Propiedades_electrónicas = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/Propiedades_electrónicas.csv')
Propiedades_electrónicas_test = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/Propiedades_electrónicas_test.csv')
propiedades_electronicas_a_usar = ["PEOE_VSA2",	"SMR_VSA7",	"SMR_VSA9"]

from sklearn.preprocessing import StandardScaler

# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
#Propiedades_electrónicas_scaled = Propiedades_electrónicas.copy()
Propiedades_electrónicas_test_scaled = Propiedades_electrónicas_test.copy()
# Aplicar el escalado a todas las columnas
#Propiedades_electrónicas_scaled[:] = scaler.fit_transform(Propiedades_electrónicas_scaled)
Propiedades_electrónicas_test_scaled[:] = scaler.fit_transform(Propiedades_electrónicas_test_scaled)


#train_data_scaled_electronic = pd.concat([train_data_fingerprints, Propiedades_electrónicas_scaled], axis=1)
train_data_scaled_electronic_test = pd.concat([test_data_fingerprints, Propiedades_electrónicas_test_scaled], axis=1)
#guardar_csv(train_data_scaled_electronic, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic.csv')
#guardar_csv(train_data_scaled_electronic_test, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic_test.csv')

"""
"""
#MODELO USANDO MACCS
"""
print('\n ****************')
print('MODELOS xgboost ELECTRONICAS')
print('****************')
from modules.procesamiento.modelos import entrenar_xgboost
print('\n----------------')
print('MODELOS xgboost MACCS')
print('----------------')

columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS, cm, y_prob_MACCS, y_test_MACCS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba  MACCS ')
print('--------------------------------')
#test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
#test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
#MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
#print(MACCS_test)
predict_data = MACCS_test
nuevas_predicciones = xgboost_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_prediccionesMACCS'] = nuevas_predicciones


from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_prediccionesMACCS']
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


print('\n----------------')
print('MODELOS xgboost MACCS y Fisicoquimicas')
print('----------------')
Electronicas = train_data_scaled_electronic[propiedades_electronicas_a_usar]
Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
columns_subset = pd.concat([Fisicoquimicas, MACCS, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_Fisicoquimicas, cm, y_prob_MACCS_FISICOQUIMICOS, y_test_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
Electronicas_TEST = train_data_scaled_electronic_test[propiedades_electronicas_a_usar]
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_Fisicoquimicas.predict(predict_data)
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



print('\n----------------')
print('MODELOS xgboost ECFP')
print('----------------')

ECFP = train_data_fingerprints.iloc[:, 6:2054]
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP, cm, y_prob_ECFP, y_test_ECFP = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
nuevas_predicciones = xgboost_ECFP.predict(predict_data)
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


print('\n----------------')
print('MODELOS xgboost ECFP y Fisicoquimicos')
print('----------------')

columns_subset = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP_Fisicoquimicos, cm, y_prob_ECFP_FISICOQUIMICOS, y_test_ECFP_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
nuevas_predicciones = xgboost_ECFP_Fisicoquimicos.predict(predict_data)
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS ')
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
xgboost_MACCS_ECFP, cm, y_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
MACCS_test = MACCS_test.add_prefix("MACCS_")
columns_subset_test = pd.concat([ECFP_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCS']
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
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
modelos = [xgboost_MACCS, xgboost_MACCS_Fisicoquimicas, xgboost_ECFP, xgboost_ECFP_Fisicoquimicos, xgboost_MACCS_ECFP, xgboost_MACCS_ECFP_fisicoquimicas]
X_test_list = [y_prob_MACCS, y_prob_MACCS_FISICOQUIMICOS, y_prob_ECFP, y_prob_ECFP_FISICOQUIMICOS, y_prob_ECFP_MACCS, y_prob_ECFP_MACCS_FISICOQUIMICOS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICOS, y_test_ECFP, y_test_ECFP_FISICOQUIMICOS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo_MACCS_XGBOOST_Electronicas",	"Modelo_MACCS_FISICOQUIMICOS_XGBOOST_Electronicas",
                     "Modelo_ECFP_XGBOOST_Electronicas",	"Modelo_ECFP_FISICOQUIMICOS_XGBOOST_Electronicas",
                     "Modelo_ECFP_MACCS_XGBOOST_Electronicas",	"Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_Electronicas"]

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
print('XGBoost CON DATASET BALANCEADO Y VARIABLES ELECTRONICAS')
print('*******************************')

train_data_fingerprints = propiedades_electronicas_balanced.copy()

columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS, cm, y_prob_MACCS, y_test_MACCS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba  MACCS ')
print('--------------------------------')
test_data_fingerprints = train_data_scaled_electronic_test
#test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
#MACCS_test = test_data_fingerprints.iloc[:, 2054:2221]
#print(MACCS_test)
predict_data = MACCS_test
nuevas_predicciones = xgboost_MACCS.predict(predict_data)
test_data_fingerprints['nuevas_prediccionesMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_prediccionesMACCS']
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


print('\n----------------')
print('MODELOS xgboost MACCS y Fisicoquimicas')
print('----------------')
Electronicas = propiedades_electronicas_balanced[propiedades_electronicas_a_usar]
columns_subset = pd.concat([Fisicoquimicas, Electronicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_Fisicoquimicas, cm, y_prob_MACCS_FISICOQUIMICOS, y_test_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

"""
#Prueba modelo con dataset prueba
"""
print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
Electronicas_TEST = test_data_fingerprints[propiedades_electronicas_a_usar]
#Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_Fisicoquimicas.predict(predict_data)
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



print('\n----------------')
print('MODELOS xgboost ECFP')
print('----------------')
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP, cm, y_prob_ECFP, y_test_ECFP = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
#ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
nuevas_predicciones = xgboost_ECFP.predict(predict_data)
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


print('\n----------------')
print('MODELOS xgboost ECFP y Fisicoquimicos')
print('----------------')
columns_subset = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_ECFP_Fisicoquimicos, cm, y_prob_ECFP_FISICOQUIMICOS, y_test_ECFP_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
#Electronicas_TEST = test_data_fingerprints[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
nuevas_predicciones = xgboost_ECFP_Fisicoquimicos.predict(predict_data)
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS ')
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
xgboost_MACCS_ECFP, cm, y_prob_ECFP_MACCS, y_test_ECFP_MACCS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
MACCS_test = MACCS_test.add_prefix("MACCS_")
columns_subset_test = pd.concat([ECFP_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCS']
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

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')

columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas.predict(predict_data)
test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS'] = nuevas_predicciones

#guardar_csv(test_data, 'data/predicion_arbol.csv')
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay
# Ya tienes las predicciones en la columna 'nuevas_predicciones' y las etiquetas reales en 'Clasificacion_ATS'
y_true = test_data_fingerprints['Clasificacion_ATS']
y_pred = test_data_fingerprints['nuevas_predicciones_ECFPMACCSFISICOQUIMICAS']
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
modelos = [xgboost_MACCS, xgboost_MACCS_Fisicoquimicas, xgboost_ECFP, xgboost_ECFP_Fisicoquimicos, xgboost_MACCS_ECFP, xgboost_MACCS_ECFP_fisicoquimicas]
X_test_list = [y_prob_MACCS, y_prob_MACCS_FISICOQUIMICOS, y_prob_ECFP, y_prob_ECFP_FISICOQUIMICOS, y_prob_ECFP_MACCS, y_prob_ECFP_MACCS_FISICOQUIMICOS]
y_test_list = [y_test_MACCS, y_test_MACCS_FISICOQUIMICOS, y_test_ECFP, y_test_ECFP_FISICOQUIMICOS, y_test_ECFP_MACCS, y_test_ECFP_MACCS_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo_MACCS_XGBOOST_Electronicas_balanced",	"Modelo_MACCS_FISICOQUIMICOS_XGBOOST_Electronicas_balanced",
                     "Modelo_ECFP_XGBOOST_Electronicas_balanced",	"Modelo_ECFP_FISICOQUIMICOS_XGBOOST_Electronicas_balanced",
                     "Modelo_ECFP_MACCS_XGBOOST_Electronicas_balanced",	"Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_Electronicas_balanced"]

graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)