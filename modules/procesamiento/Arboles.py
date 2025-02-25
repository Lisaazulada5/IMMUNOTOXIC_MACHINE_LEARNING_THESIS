import os
import pandas as pd
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.modelos import arbol_decision
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

"""
print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
train_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_fingerprints.csv')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])


MACCS = train_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS, y_testMACCS = arbol_decision(columnas_predictoras, target)


#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])


MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
predict_data = MACCS_test #extrae los nombres de las columnas del df
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
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS_FISICOQUIMICOS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_FISICOQUIMICOS, y_testMACCS_FISICOQUIMICOS= arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str)
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
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
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP, y_testECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
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
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP_FISICOQUIMICOS, y_testECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
predict_data = columns_subset_test
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
columns_subset = pd.concat([MACCS, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_MACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_ECFP, y_testMACCS_ECFP= arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
#columnas_predictoras = columnas_predictoras.astype(int)
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_testa, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS, y_testMACCS_ECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP MACCS FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [modeloMACCS, modeloMACCS_FISICOQUIMICOS, modelo_ECFP, modelo_ECFP_FISICOQUIMICAS, modelo_ECFP_MACCS, modeloECFP_MACCS_FISICOQUIMICAS]
X_test_list = [y_pred_prob_testMACCS, y_pred_prob_testMACCS_FISICOQUIMICOS, y_pred_prob_testECFP, y_pred_prob_testECFP_FISICOQUIMICOS, y_pred_prob_testMACCS_ECFP, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS]
y_test_list = [y_testMACCS, y_testMACCS_FISICOQUIMICOS, y_testECFP, y_testECFP_FISICOQUIMICOS, y_testMACCS_ECFP, y_testMACCS_ECFP_FISICOQUIMICOS]
etiquetas_modelos = ['Modelo MACCS', 'Modelo MACCS + FISICOQUIMICAS', 'Modelo ECFP', 'Modelo ECFP + FISICOQUIMICAS', 'Modelo ECFP + MACCS', 'Modelo ECFP, MACCS, FISICOQUIMICAS']
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

print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo

train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])

MACCS = train_data_fingerprints.iloc[:, 2054:2222]
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS, y_testMACCS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
predict_data = MACCS_test #extrae los nombres de las columnas del df
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
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS_FISICOQUIMICOS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_FISICOQUIMICOS, y_testMACCS_FISICOQUIMICOS= arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str)
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
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
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP, y_testECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
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
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP_FISICOQUIMICOS, y_testECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, ECFP_test], axis=1)
predict_data = columns_subset_test
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
columns_subset = pd.concat([MACCS, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_MACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_ECFP, y_testMACCS_ECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
#columnas_predictoras = columnas_predictoras.astype(int)
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_testa, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS, y_testMACCS_ECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP MACCS FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from modules.procesamiento.modelos import entrenar_modelo
param_grid = {
    'max_depth': [3, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy'],
    'ccp_alpha': [0, 0.01, 0.1 ],
    'class_weight': [None, 'balanced']
}
#modelo = DecisionTreeClassifier(random_state=42)  # ✅ Crear instancia
#entrenar_modelo(modelo, param_grid, columnas_predictoras, target)



"""
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [modeloMACCS, modeloMACCS_FISICOQUIMICOS, modelo_ECFP, modelo_ECFP_FISICOQUIMICAS, modelo_ECFP_MACCS, modeloECFP_MACCS_FISICOQUIMICAS]
X_test_list = [y_pred_prob_testMACCS, y_pred_prob_testMACCS_FISICOQUIMICOS, y_pred_prob_testECFP, y_pred_prob_testECFP_FISICOQUIMICOS, y_pred_prob_testMACCS_ECFP, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS]
y_test_list = [y_testMACCS, y_testMACCS_FISICOQUIMICOS, y_testECFP, y_testECFP_FISICOQUIMICOS, y_testMACCS_ECFP, y_testMACCS_ECFP_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo MACCS_Electronicos_Balanced",	" Modelo MACCS + FISICOQUIMICAS_Balanced",	" Modelo ECFP_Balanced",
                     " Modelo ECFP + FISICOQUIMICAS_Balanced",	" Modelo ECFP + MACCS_Balanced",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Balanced"]

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

"""
"""
#ARBOLES DE DECISION DESBALANCEADO CON PROPIEDADES ELECTRONICAS
"""
print('_************************************************************* ')
print('ARBOLES DE DECISION DESBALANCEADO CON PROPIEDADES ELECTRONICAS')
print('_************************************************************* ')

print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
#train_data_fingerprints = train_data_scaled_electronic
#train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#MACCS = train_data_fingerprints.iloc[:, 2054:2221]
#print(MACCS)
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS, y_testMACCS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = train_data_scaled_electronic_test
#test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2221]
predict_data = MACCS_test #extrae los nombres de las columnas del df
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
Electronicas = train_data_scaled_electronic[propiedades_electronicas_a_usar]
#Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
print(Fisicoquimicas.columns)
columns_subset = pd.concat([Fisicoquimicas,Electronicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS_FISICOQUIMICOS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_FISICOQUIMICOS, y_testMACCS_FISICOQUIMICOS= arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str)
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
Electronicas_TEST = test_data_fingerprints[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([Fisicoquimicas_test,Electronicas_TEST, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
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

#ECFP = train_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP, y_testECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
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

columns_subset = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP_FISICOQUIMICOS, y_testECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
predict_data = columns_subset_test
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
columns_subset = pd.concat([MACCS, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_MACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_ECFP, y_testMACCS_ECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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

columns_subset = pd.concat([MACCS, ECFP, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
#columnas_predictoras = columnas_predictoras.astype(int)
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_testa, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS, y_testMACCS_ECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP MACCS FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [modeloMACCS, modeloMACCS_FISICOQUIMICOS, modelo_ECFP, modelo_ECFP_FISICOQUIMICAS, modelo_ECFP_MACCS, modeloECFP_MACCS_FISICOQUIMICAS]
X_test_list = [y_pred_prob_testMACCS, y_pred_prob_testMACCS_FISICOQUIMICOS, y_pred_prob_testECFP, y_pred_prob_testECFP_FISICOQUIMICOS, y_pred_prob_testMACCS_ECFP, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS]
y_test_list = [y_testMACCS, y_testMACCS_FISICOQUIMICOS, y_testECFP, y_testECFP_FISICOQUIMICOS, y_testMACCS_ECFP, y_testMACCS_ECFP_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo MACCS_Electronicos_Electronicas",	" Modelo MACCS + FISICOQUIMICAS_Electronicas",	" Modelo ECFP_Electronicas",
                     " Modelo ECFP + FISICOQUIMICAS_Electronicas",	" Modelo ECFP + MACCS_Electronicas",
                     " Modelo ECFP_MACCS_Electronicas"]
graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)
"""


"""
print('______________________________  ')
print('MODELO ARBOL DE DECISION ELECTRONICAS Y FISICOQUIMICAS ')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo
# Seleccionar las columnas que no son la posición 0 ni de la 7 a la 2054

columns_subset = pd.concat([Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
#columnas_predictoras = columnas_predictoras.astype(int)
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_testa = arbol_decision(columnas_predictoras, target)

print('Extraer la importancia de las características modelo_todaslasvariables')
print('______________________________  ')
importancias = modeloECFP_MACCS_FISICOQUIMICAS.feature_importances_

# Crear DataFrame para visualizar
variables = columns_subset.columns
df_importancias = pd.DataFrame({
    "Variable": variables,
    "Importancia": importancias})

# Ordenar por importancia
df_importancias = df_importancias.sort_values(by="Importancia", ascending=False)

# Mostrar las variables más importantes
print(df_importancias)
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
print('ARBOLES CON DATASET BALANCEADO')
print('*******************************')

#train_data_fingerprints = propiedades_electronicas_balanced.copy()

print('______________________________  ')
print('MODELO ARBOL DE DECISION MACCS')
print('______________________________  ')
# Definir las columnas predictoras y la variable objetivo

#train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#MACCS = train_data_fingerprints.iloc[:, 2054:2221]
columnas_predictoras = MACCS
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS, y_testMACCS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS')
print('--------------------------------')
test_data_fingerprints = train_data_scaled_electronic_test
#test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS_test = test_data_fingerprints.iloc[:, 2054:2221]
predict_data = MACCS_test #extrae los nombres de las columnas del df
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
Electronicas = propiedades_electronicas_balanced[propiedades_electronicas_a_usar]
train_data_fingerprints.columns = train_data_fingerprints.columns.astype(str)
#Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
print(Fisicoquimicas)
columns_subset = pd.concat([Fisicoquimicas, Electronicas, MACCS], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloMACCS_FISICOQUIMICOS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_FISICOQUIMICOS, y_testMACCS_FISICOQUIMICOS= arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba MACCS y FISICOQUIMICOS')
print('--------------------------------')
Electronicas_TEST = test_data_fingerprints[propiedades_electronicas_a_usar]
test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str)
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
columns_subset_test = pd.concat([Fisicoquimicas_test,Electronicas_TEST, MACCS_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
predict_data = columns_subset_test
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

#ECFP = train_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras = ECFP
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP, y_testECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
#print(ECFP)
columnas_predictoras_test = ECFP_test
predict_data = columnas_predictoras_test
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

columns_subset = pd.concat([Fisicoquimicas, Electronicas, ECFP], axis=1)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testECFP_FISICOQUIMICOS, y_testECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([Fisicoquimicas_test, Electronicas_TEST, ECFP_test], axis=1)
predict_data = columns_subset_test
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
columns_subset = pd.concat([MACCS, ECFP], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modelo_ECFP_MACCS, accuracy_train, accuracy_test, cm_test, report_test, y_pred_prob_testMACCS_ECFP, y_testMACCS_ECFP = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([MACCS_test, ECFP_test], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
#columnas_predictoras = columnas_predictoras.astype(int)
target = train_data_fingerprints["Clasificacion_ATS"]
# Llamar al modelo
modeloECFP_MACCS_FISICOQUIMICAS, accuracy_train, accuracy_test, cm_test, report_testa, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS, y_testMACCS_ECFP_FISICOQUIMICOS = arbol_decision(columnas_predictoras, target)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP MACCS FISICOQUIMICAS')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
print(columns_subset_test)
columns_subset_test.columns = columns_subset_test.columns.astype(str)
columnas_predictoras_test = columns_subset_test
predict_data = columnas_predictoras_test
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
#CURVAS ROC PARA VER MEJORES MODELOS
"""
from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [modeloMACCS, modeloMACCS_FISICOQUIMICOS, modelo_ECFP, modelo_ECFP_FISICOQUIMICAS, modelo_ECFP_MACCS, modeloECFP_MACCS_FISICOQUIMICAS]
X_test_list = [y_pred_prob_testMACCS, y_pred_prob_testMACCS_FISICOQUIMICOS, y_pred_prob_testECFP, y_pred_prob_testECFP_FISICOQUIMICOS, y_pred_prob_testMACCS_ECFP, y_pred_prob_testMACCS_ECFP_FISICOQUIMICOS]
y_test_list = [y_testMACCS, y_testMACCS_FISICOQUIMICOS, y_testECFP, y_testECFP_FISICOQUIMICOS, y_testMACCS_ECFP, y_testMACCS_ECFP_FISICOQUIMICOS]
etiquetas_modelos = ["Modelo MACCS_Electronicos_Electronicas_Balanced",	" Modelo MACCS + FISICOQUIMICAS_Electronicas_Balanced",
                     " Modelo ECFP_Electronicas_Balanced",	" Modelo ECFP + FISICOQUIMICAS_Electronicas_Balanced",
                     " Modelo ECFP + MACCS_Electronicas_Balanced",	" Modelo ECFP + MACCS + FISICOQUIMICAS_Electronicas_Balanced",]

graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)
