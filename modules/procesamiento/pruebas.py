import pandas as pd
from modules.procesamiento.validar_smiles import obtener_iupac_desde_pubchem
from data.manejo_archivos import leer_csv, guardar_csv
from modules.procesamiento.obtener_iupac_desde_smiles_rdkit import obtener_iupac_desde_smiles
import os
import pandas as pd
from modules.procesamiento.validar_smiles import obtener_iupac_desde_pubchem
from data.manejo_archivos import leer_csv, guardar_csv

"""
"""
#DADO QUE LAS VARIABLES PROPUESTAS PARECEN NO SER SUCIFIENTES SE HACE NECESARIO EL CALCULO DE OTRAS VARIABLES

#Para esto se usa el dataframe que ya tiene las variables escaladas y que contiene los fingerprints
"""
df_nuevasvariables = train_data_fingerprints.copy()
from modules.procesamiento.calculo_descriptores_moleculares import calcular_balaban, calcular_wiener
#df_nuevasvariables["BalabanJ"] = df_nuevasvariables["SMILES"].apply(calcular_balaban)
#df_nuevasvariables["wiener"] = df_nuevasvariables["SMILES"].apply(calcular_wiener)
#guardar_csv(df_nuevasvariables, 'data/df_nuevasvariables.csv')
#graficar_y_guardar_variable_continua(df_nuevasvariables,'BalabanJ', 'data/graficas/Carga_Gasteiger')
#graficar_y_guardar_boxplot(df_nuevasvariables, 'Clasificacion_ATS', 'BalabanJ', 'data/graficas/cajas_bigotes')
#df_nuevasvariables = leer_csv('data/df_nuevasvariables.csv')

from sklearn.preprocessing import StandardScaler

# Crear el escalador
#scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
#df_nuevasvariables = df_nuevasvariables.copy()
#df_nuevasvariables['BalabanJ_scaled'] = scaler.fit_transform(df_nuevasvariables[['BalabanJ']])

from modules.procesamiento.calculo_fingerprints import  calcular_fingerprints_atom

atom_pair = calcular_fingerprints_atom(df_nuevasvariables['SMILES'])
df_nuevasvariables['atom_pair'] = atom_pair

import numpy as np
import pandas as pd

atom_pair_array = []

for fp in df_nuevasvariables['atom_pair']:
    if fp is not None:
        on_bits = list(fp.GetOnBits())  # Obtiene los bits activados
        arr = np.zeros(2048, dtype=np.int8)  # Vector de 2048 bits

        for bit in on_bits:
            if bit < 2048:  # Asegurar que el bit está dentro del rango permitido
                arr[bit] = 1  # Marca el bit como activo

        atom_pair_array.append(arr)
    else:
        atom_pair_array.append(np.zeros(2048, dtype=np.int8))  # Si no hay FP, agrega un vector vacío

# Convertir a ame
df_atom_pair = pd.DataFrame(atom_pair_array)
df_atom_pair = df_atom_pair.add_prefix("atompair_")
print(df_atom_pair)

df_nuevasvariables = pd.concat([df_nuevasvariables, df_atom_pair ], axis=1)
#guardar_csv(df_nuevasvariables, 'data/df_nuevasvariables_atompair.csv')



"""

"""
#print(atom_pair)

#print(df_nuevasvariables.columns)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar tu DataFrame (reemplázalo con tu archivo si es necesario)

df_nuevasvariables = leer_csv('data/df_nuevasvariables_atompair.csv')



from modules.procesamiento.calculo_descriptores_moleculares import calcular_bcutw, calcular_bcutc
#calcular_minparcialcharge(df_nuevasvariables, 'SMILES')
#calcular_maxparcialcharge(df_nuevasvariables, 'SMILES')
# Aplicar la función a cada SMILES y almacenar los resultados en nuevas columnas
df_nuevasvariables[['BCUTw_1_l', 'BCUTw_1_h']] = df_nuevasvariables['SMILES'].apply(lambda x: pd.Series(calcular_bcutw(x)))
#df_nuevasvariables[['bcutc_l', 'bcutc_h']] = df_nuevasvariables['SMILES'].apply(lambda x: pd.Series(calcular_bcutc(x)))
#calcular_numero_aceptores(df_nuevasvariables, 'SMILES')
# Mostrar el resultado
#print(f" 'numero de valores nulos:' {df_nuevasvariables['bcutc_l'].isnull().sum()}")
#print(f" 'numero de valores nulos:' {df_nuevasvariables['bcutc_h'].isnull().sum()}")


#graficar_y_guardar_variable_continua(df_nuevasvariables,'BCUTw_1_l', 'data/BCUTw_1_l')
#graficar_y_guardar_variable_continua(df_nuevasvariables,'BCUTw_1_h', 'data/BCUTw_1_h')
#graficar_y_guardar_boxplot(df_nuevasvariables, 'Clasificacion_ATS', 'BCUTw_1_l', 'data/BCUTw_1_l')
#graficar_y_guardar_boxplot(df_nuevasvariables, 'Clasificacion_ATS', 'BCUTw_1_h', 'data/BCUTw_1_h')
#print(f" 'numero de valores nulos:' {df_nuevasvariables['Min_Charge'].isnull().sum()}")
#print(f" 'sin quitar los nan:' {df_nuevasvariables}")
#df_nuevasvariables = df_nuevasvariables.dropna()
#print(f" 'quitando los nan:' {df_nuevasvariables}")

from sklearn.preprocessing import StandardScaler

# Crear el escalador
scaler = StandardScaler()

# Aplicar escalado solo a las columnas necesarias
df_nuevasvariables = df_nuevasvariables.copy()
df_nuevasvariables['BCUTw_1_l_scaled'] = scaler.fit_transform(df_nuevasvariables[['BCUTw_1_l']])
df_nuevasvariables['BCUTw_1_h_scaled'] = scaler.fit_transform(df_nuevasvariables[['BCUTw_1_h']])
#df_nuevasvariables['NumHDonors_scaled'] = scaler.fit_transform(df_nuevasvariables[['NumHDonors']])


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Seleccionar las variables de interés
variables = ['LogP_scaled', 'TPSA_scaled',
             'NumRotatableBonds_scaled', 'Peso_Molecular_scaled', 'Dobles_Enlaces_scaled', 'BCUTw_1_l_scaled', 'BCUTw_1_h_scaled']

# Calcular la matriz de correlación
corr_matrix = df_nuevasvariables[variables].corr()

# Imprimir la matriz
print(corr_matrix)


#conteo de variables para cambiar el class weight dde las MSV
conteo = df_nuevasvariables["Clasificacion_ATS"].value_counts()
print(conteo)





from modules.procesamiento.modelos import entrenar_svm
df_nuevasvariables.columns = df_nuevasvariables.columns.astype(str)
df_nuevasvariables = df_nuevasvariables.drop(columns=['MACCS', 'SMILES', 'ECFP','atom_pair'])
Fisicoquimicas_ = df_nuevasvariables[['LogP_scaled', 'Peso_Molecular_scaled', 'BCUTw_1_l_scaled']]
ECFP = df_nuevasvariables.iloc[:, 6:2054]
ECFP = ECFP.add_prefix("ECFP_")
#print(ECFP)
MACCS = df_nuevasvariables.iloc[:, 2054:2221]
MACCS = MACCS.add_prefix("MACCS_")
#print(MACCS)
Atom_pair = df_nuevasvariables.iloc[:, 2221:4273]
#print(Atom_pair)

print('\n----------------')
print('MODELOS ECFP Y FISICOQUIMICOS sin gaiser')
print('----------------')
from modules.procesamiento.modelos import entrenar_svm
Fisicoquimicas = df_nuevasvariables[['LogP_scaled', 'TPSA_scaled',
       'NumRotatableBonds_scaled', 'Peso_Molecular_scaled',
       'Dobles_Enlaces_scaled']]
columns_subset = pd.concat([ECFP, MACCS, Fisicoquimicas], axis=1)
columns_subset.columns = columns_subset.columns.astype(str)
columnas_predictoras = columns_subset
target = df_nuevasvariables["Clasificacion_ATS"]
X = columnas_predictoras
y= target
MSV_ECFP_2 = entrenar_svm(X,y)

print('\n----------------')
print('MODELOS ECFP Y FISICOQUIMICOS con gaiser')
print('----------------')
predictores = pd.concat([MACCS, ECFP, Fisicoquimicas_], axis=1)
predictores.columns = predictores.columns.astype(str)
target = df_nuevasvariables["Clasificacion_ATS"]
X_ = predictores
y_ = target
MSV_ECFP = entrenar_svm(X_,y_)


pruebadf = test_data_fingerprints.copy()

pruebadf.columns = pruebadf.columns.astype(str)
pruebadf = pruebadf.drop(columns=['MACCS', 'SMILES', 'ECFP'])
Fisicoquimicas_prueba = pruebadf[['LogP_scaled', 'Peso_Molecular_scaled']]
ECFP_prueba= pruebadf.iloc[:, 6:2054]
ECFP_prueba = ECFP_prueba.add_prefix("ECFP_")
#print(ECFP)
MACCS_prueba = pruebadf.iloc[:, 2054:2221]
MACCS_prueba = MACCS_prueba.add_prefix("MACCS_")
prueba_data = pd.concat([ECFP_prueba, MACCS_prueba, Fisicoquimicas_prueba], axis=1)
prueba_data.columns = prueba_data.columns.astype(str)

y_test = df_nuevasvariables["Clasificacion_ATS"]

y_pred = MSV_ECFP_2.predict(prueba_data)
# Evaluar el modelo
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred) if y_test is not None else "No hay etiquetas reales")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def arbol_decision(X, y, test_size=0.2, random_state=42):
    # Separar datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size, random_state=random_state, stratify=y)

    # Definir modelo base
    modelo = DecisionTreeClassifier(random_state=random_state, class_weight='balanced')

    # GridSearch para encontrar los mejores hiperparámetros
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [3, 5, 10, None],  
        'min_samples_split': [2, 5, 10],  
        'min_samples_leaf': [1, 2, 5],  
        'ccp_alpha': [0.0, 0.001, 0.01]
    }

    grid_search = GridSearchCV(modelo, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    modelo = grid_search.best_estimator_

    # Evaluación en prueba
    y_pred_test = modelo.predict(X_test)
    y_pred_prob_test = modelo.predict_proba(X_test)[:, 1]
    accuracy_test = accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    report_test = classification_report(y_test, y_pred_test)

    # Validación cruzada
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
    auc_scores = cross_val_score(modelo, X_train, y_train, cv=cv, scoring="roc_auc")

    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Precisión en prueba: {accuracy_test:.4f}")
    print("Matriz de confusión (Prueba):")
    print(cm_test)
    print("Reporte de clasificación (Prueba):")
    print(report_test)
    print("\n______________________________")
    print("CROSS VALIDATION AUC-ROC")
    print("______________________________")
    print(f"AUC en cada fold: {auc_scores}")
    print(f"AUC promedio: {np.mean(auc_scores)}")
    print(f"Desviación estándar: {np.std(auc_scores)}")

    # Graficar curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_test)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_score(y_test, y_pred_prob_test):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()

    return modelo, accuracy_test, cm_test, report_test
