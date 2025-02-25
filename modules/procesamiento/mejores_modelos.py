import pandas as pd
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.modelos import entrenar_xgboost
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# Definir las columnas predictoras y la variable objetivo
train_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_fingerprints.csv')

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')
train_data_fingerprints = train_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
MACCS = train_data_fingerprints.iloc[:, 2054:2222]
MACCS = MACCS.add_prefix("MACCS_")
Fisicoquimicas = train_data_fingerprints.iloc[:, 1:6]
ECFP = train_data_fingerprints.iloc[:, 6:2054]
ECFP = ECFP.add_prefix("ECFP_")
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS, X_train, X_test = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
test_data_fingerprints = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/test_data_fingerprints.csv')
test_data_fingerprints = test_data_fingerprints.drop(columns=['MACCS', 'SMILES', 'ECFP'])
#test_data_fingerprints.rename(columns=lambda col: f"MACCS_{col}" if col in test_data_fingerprints.iloc[:, 2054:2222] else col, inplace=True)
MACCS_test = test_data_fingerprints.iloc[:, 2054:2222]
MACCS_test = MACCS_test.add_prefix("MACCS_")
Fisicoquimicas_test = test_data_fingerprints.iloc[:, 1:6]
ECFP_test = test_data_fingerprints.iloc[:, 6:2054]
ECFP_test = ECFP_test.add_prefix("ECFP_")
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


import shap

"""
SHAP MODELO PREDICCIONES
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas, X_train)
shap_values = explainer(X_test)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
shap.summary_plot(shap_values, X_test, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Modelo dataset desbalanceado + fisicoquímicas", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
#plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo.png', dpi=300)
#plt.show()  # Mostrar la figura en pantalla






"""
PREDICCION DEL MODELO MOLECULAS COSING
"""
X_nuevo = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS.csv')
#X_nuevo = X_nuevo.drop(columns=["SMILES", "ECFP", "MACCS"])
#probs = xgboost_MACCS_ECFP_fisicoquimicas.predict_proba(X_nuevo)
#preds = xgboost_MACCS_ECFP_fisicoquimicas.predict(X_nuevo)

PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES =  X_nuevo.copy()
PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES["Predicción"] = xgboost_MACCS_ECFP_fisicoquimicas.predict(X_nuevo)
PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES[["Probabilidad_Inac", "Probabilidad_Act"]] = xgboost_MACCS_ECFP_fisicoquimicas.predict_proba(X_nuevo)
#guardar_csv(PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES.csv')
print("realizado")

import shap

"""
SHAP MODELO PREDICCIONES
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas)
shap_values = explainer(X_nuevo)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
shap.summary_plot(shap_values, X_nuevo, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Predicciones", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
#plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo_Predicciones.png', dpi=300)
#plt.show()  # Mostrar la figura en pantalla

"""
Dependence SHAP
"""
#shap_values_array = shap_values.values  # Extraer los valores SHAP
#shap.dependence_plot("LogP_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("Peso_Molecular_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("TPSA_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("NumRotatableBonds_scaled", shap_values_array, X_nuevo)
#shap.dependence_plot("Dobles_Enlaces_scaled", shap_values_array, X_nuevo)






"""
Segundo mejor modelo
"""

print('\n----------------')
print('MODELOS xgboost ECFP Y MACCS fisicoquimicas')
print('----------------')
train_data_scaled_electronic = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic.csv')
#print(train_data_scaled_electronic)
propiedades_electronicas_a_usar = ["PEOE_VSA2",	"SMR_VSA7",	"SMR_VSA9"]
Electronicas = train_data_scaled_electronic[propiedades_electronicas_a_usar]
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas, Electronicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = train_data_fingerprints["Clasificacion_ATS"]
X = columnas_predictoras
y= target
xgboost_MACCS_ECFP_fisicoquimicas_electronicas, cm, y_prob_ECFP_MACCS_FISICOQUIMICOS_electronicos, y_test_ECFP_MACCS_FISICOQUIMICOS_electronicos, X_train, X_test = entrenar_xgboost(X,y)

#Prueba modelo con dataset prueba

print('Prueba modelo con dataset prueba ECFP y MACCS +  Fisicoquimicas')
print('--------------------------------')
# Definir las columnas predictoras y la variable objetivo
train_data_scaled_electronic_test = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/train_data_scaled_electronic_test.csv')
Electronicas_TEST = train_data_scaled_electronic_test[propiedades_electronicas_a_usar]
columns_subset_test = pd.concat([ECFP_test, MACCS_test, Fisicoquimicas_test, Electronicas_TEST], axis=1)
columns_subset_test.columns = columns_subset_test.columns.astype(str)

predict_data = columns_subset_test
nuevas_predicciones = xgboost_MACCS_ECFP_fisicoquimicas_electronicas.predict(predict_data)
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

from modules.procesamiento.graficas import graficar_roc_multiple
modelos = [xgboost_MACCS_ECFP_fisicoquimicas , xgboost_MACCS_ECFP_fisicoquimicas_electronicas ]
X_test_list = [y_prob_ECFP_MACCS_FISICOQUIMICOS, y_prob_ECFP_MACCS_FISICOQUIMICOS_electronicos]
y_test_list = [y_test_ECFP_MACCS_FISICOQUIMICOS, y_test_ECFP_MACCS_FISICOQUIMICOS_electronicos]
etiquetas_modelos = ["Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED", "Modelo_ECFP_MACCS_FISICOQUIMICOS_XGBOOST_BALANCED_electronicas"]
#graficar_roc_multiple(modelos, X_test_list, y_test_list, etiquetas_modelos)


"""
SHAP MODELO DE ENTRENAMIENTO
"""
explainer = shap.TreeExplainer(xgboost_MACCS_ECFP_fisicoquimicas_electronicas, X_train)
shap_values = explainer(X_test)  # Sin la columna de predicción
# Crear el gráfico SHAP con título

# Crear la figura y agregar título
plt.figure()
#shap.summary_plot(shap_values, X_test, show=False)  # Desactiva el show automático de SHAP
plt.title("SHAP Modelo dataset desbalanceado + área superficial", fontsize=14, pad=30)  # Aumenta el pad
# Ajustar la posición del gráfico
plt.subplots_adjust(top=0.85)  # Reduce este valor para bajar más el gráfico
# Guardar la figura
#plt.savefig('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/shap_modelo_areasuperf.png', dpi=300)
#plt.show()  # Mostrar la figura en pantalla


"""
EXTRAER INCI NAME, IUPAC NAME, CAS DE LAS SUSTANCIAS PREDICHAS
"""
predicciones_COSING = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/predicciones_COSING.csv')
PFAS_cosing_SMILES = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFAS_cosing_SMILES.csv')
import pandas as pd

# Realizar el join asegurando que predicciones_COSING mantiene su tamaño
PFAS_COSING_INFORMATION = predicciones_COSING.merge(PFAS_cosing_SMILES, on="SMILES", how="left")
print(PFAS_COSING_INFORMATION. columns)
# Verificar el tamaño del resultado
#guardar_csv(PFAS_COSING_INFORMATION, 'C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFAS_COSING_INFORMATION.csv')


Analisis_resultados_Predicciones = leer_csv('C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES.csv')
"""
Histograma
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6, 4))
ax = sns.countplot(x=Analisis_resultados_Predicciones["Predicción"], palette=["#8da0cb", "#fc8d62"])

# Etiquetas con los valores exactos
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                textcoords='offset points')

plt.xlabel("Clase Predicha")
plt.ylabel("Frecuencia")
plt.title("Distribución de Predicciones")
plt.xticks([0, 1], ["Inactivo", "Activo"])  # Etiquetas personalizadas
plt.grid(axis='y', linestyle="--", alpha=0.7)

#plt.show()



import matplotlib.pyplot as plt
import seaborn as sns


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(6,4))
sns.boxplot(data=Analisis_resultados_Predicciones[["Probabilidad_Act", "Probabilidad_Inac"]])
plt.xlabel("Clase")
plt.ylabel("Probabilidad")
plt.title("Boxplot de Probabilidades por Clase")
plt.xticks([0, 1], ["Activo", "Inactivo"])
#plt.show()

"""
Hisotgramas para las probabilidades act e inac
"""
plt.figure(figsize=(10, 4))

# Histograma para Probabilidad_Act
plt.subplot(1, 2, 1)  # Primera figura (izquierda)
sns.histplot(Analisis_resultados_Predicciones["Probabilidad_Act"], bins=20, color="#4c72b0", kde=True)
plt.xlabel("Probabilidad de ser Activo")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidad Activo")

# Histograma para Probabilidad_Inac
plt.subplot(1, 2, 2)  # Segunda figura (derecha)
sns.histplot(Analisis_resultados_Predicciones["Probabilidad_Inac"], bins=20, color="#dd8452", kde=True)
plt.xlabel("Probabilidad de ser Inactivo")
plt.ylabel("Frecuencia")
plt.title("Distribución de Probabilidad Inactivo")

plt.tight_layout()  # Ajustar los gráficos para que no se solapen
#plt.show()

"""
VISUALIZAR TODOS LOS MACCS
"""

from rdkit import Chem
from rdkit.Chem import MACCSkeys

# Obtener todas las definiciones de los bits MACCS
maccs_definitions = MACCSkeys.smartsPatts
print("Claves MACCS")
print(maccs_definitions)

# Ver las primeras claves para entender la indexación
print(list(maccs_definitions.keys())[:10])  # Muestra los primeros 10 índices

# Revisar si el bit 112 está en índice 112
bit_112 = maccs_definitions[112] if 112 in maccs_definitions else "No definido"
print(f"El bit 112 en MACCS (índice 112 en Python) representa: {bit_112}")

# Revisar si los bits están en el diccionario de definiciones de MACCS
bits_interes = [154, 99, 49, 44, 98, 109, 72, 166]

for bit in bits_interes:
    bit_definicion = maccs_definitions[bit] if bit in maccs_definitions else "No definido"
    print(f"El bit {bit} en MACCS (índice {bit} en Python) representa: {bit_definicion}")

"""
visualizar las estructuras
"""

from rdkit import Chem
from rdkit.Chem import Draw

# Subestructura del bit 112 en MACCS
smarts_pattern = "[!#6;!#1]1~*~*~*~*~*~1"  # Representación SMARTS del bit 112
substructure_mol = Chem.MolFromSmarts(smarts_pattern)

# Verifica si la conversión fue exitosa
if substructure_mol:
    # Guardar la imagen en un archivo
    img_path = "C:/Users/licit/OneDrive/Documentos/Proyectos python/TESIS/data/graficas/MACCS_49.png"
    Draw.MolToFile(substructure_mol, img_path, size=(300, 300))
    print(f"Imagen guardada en {img_path}")
else:
    print("Error al generar la subestructura.")

