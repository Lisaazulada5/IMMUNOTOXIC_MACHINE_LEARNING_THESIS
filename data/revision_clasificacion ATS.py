import pandas as ps
from manejo_archivos import leer_csv, guardar_csv
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from modules.procesamiento.calculo_fingerprints import calcular_ecfp
import pandas as pd
import numpy as np
from modules.procesamiento.calculo_fingerprints import calcular_fingerprints_maccs

PFA_COSING_SMILES = leer_csv('PFAS_cosing_SMILES.csv')
print(PFA_COSING_SMILES)

#Calculo del LogP

from modules.procesamiento.calculo_descriptores_moleculares import calcular_logp_crippen

# Aplicar la función a la columna "SMILES" y crear una nueva columna "LogP"
PFA_COSING_SMILES["LogP"] = PFA_COSING_SMILES["SMILES"].apply(calcular_logp_crippen)

PFA_COSING_SMILES = PFA_COSING_SMILES.dropna(subset=["LogP"])
print(PFA_COSING_SMILES["LogP"].isnull().value_counts())

"""
CALCULO DE PESO MOLECULAR
"""
from modules.procesamiento.calculo_descriptores_moleculares import calcular_pesos_moleculares
# Calcular los pesos moleculares y actualizar el DataFrame
PFA_COSING_SMILES = calcular_pesos_moleculares(PFA_COSING_SMILES, 'SMILES')
PFA_COSING_SMILES = PFA_COSING_SMILES.dropna(subset=["Peso_Molecular"])
print(PFA_COSING_SMILES["Peso_Molecular"].isnull().value_counts())

"""
CALCULO DE ENLACES DOBLES
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_enlaces_dobles

calcular_enlaces_dobles(PFA_COSING_SMILES, 'SMILES')
PFA_COSING_SMILES = PFA_COSING_SMILES.dropna(subset=["Dobles_Enlaces"])
print(PFA_COSING_SMILES["Dobles_Enlaces"].isnull().value_counts())
print(PFA_COSING_SMILES)

"""
Enlaces TPSA
"""
from modules.procesamiento.calculo_descriptores_moleculares import calcular_tpsa

# Calcular TPSA
fPFA_COSING_SMILES = calcular_tpsa(PFA_COSING_SMILES, 'SMILES')
PFA_COSING_SMILES = PFA_COSING_SMILES.dropna(subset=["TPSA"])
print(PFA_COSING_SMILES["TPSA"].isnull().value_counts())

"""
Calculo de enlaces rotables
"""

from modules.procesamiento.calculo_descriptores_moleculares import calcular_enlaces_rotables

calcular_enlaces_rotables(PFA_COSING_SMILES, 'SMILES')
PFA_COSING_SMILES = PFA_COSING_SMILES.dropna(subset=["NumRotatableBonds"])
print(PFA_COSING_SMILES["NumRotatableBonds"].isnull().value_counts())
print(PFA_COSING_SMILES)

PFA_COSING_SMILES = PFA_COSING_SMILES.drop(columns=['INCI Name/Substance Name', 'Type', 'CAS No.','Annex/Ref'])
#guardar_csv(PFA_COSING_SMILES, 'PFA_COSING_SMILES_fisicoquimicas.csv')
print(PFA_COSING_SMILES.columns)

PFA_COSING_SMILES_fisicoquimicas = leer_csv('PFA_COSING_SMILES_fisicoquimicas.csv')
"""
   Escalado max-min
   """
# Crear el escalador
scaler = MinMaxScaler()

PFA_COSING_SMILES_fisicoquimicas['NumRotatableBonds_scaled'] = scaler.fit_transform(PFA_COSING_SMILES_fisicoquimicas[['NumRotatableBonds']])
PFA_COSING_SMILES_fisicoquimicas['Dobles_Enlaces_scaled'] = scaler.fit_transform(PFA_COSING_SMILES_fisicoquimicas[['Dobles_Enlaces']])

scaler = StandardScaler()

PFA_COSING_SMILES_fisicoquimicas[['LogP_scaled', 'Peso_Molecular_scaled', 'TPSA_scaled']] = scaler.fit_transform(PFA_COSING_SMILES_fisicoquimicas[['LogP',
                                                                                                            'Peso_Molecular', 'TPSA']])

print(PFA_COSING_SMILES_fisicoquimicas)

calcular_ecfp(PFA_COSING_SMILES_fisicoquimicas, 'SMILES')
#print(train_data_fingerprints['ECFP'][0])
from rdkit.Chem import DataStructs
ECFP_array_test = []
for i in PFA_COSING_SMILES_fisicoquimicas['ECFP']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    ECFP_array_test.append(arr)

#print(ECFP_array_test)
PFA_COSING_SMILES_fisicoquimicas = pd.concat([PFA_COSING_SMILES_fisicoquimicas, pd.DataFrame(ECFP_array_test)], axis  =1)
#print(test_data_fingerprints)

#calculamos los MACCS a partir de los SMILES
calcular_fingerprints_maccs(PFA_COSING_SMILES_fisicoquimicas, 'SMILES')
from rdkit.Chem import DataStructs
MACCS_array_test = []
for i in PFA_COSING_SMILES_fisicoquimicas['MACCS']:
    arr = np.zeros((0,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(i, arr)
    MACCS_array_test.append(arr)
PFA_COSING_SMILES_fisicoquimicas = pd.concat([PFA_COSING_SMILES_fisicoquimicas, pd.DataFrame(MACCS_array_test)], axis  =1)
PFA_COSING_SMILES_fisicoquimicas = PFA_COSING_SMILES_fisicoquimicas.drop(columns=["LogP", "Peso_Molecular", "Dobles_Enlaces", "TPSA",
                                                                                  "NumRotatableBonds"])
#guardar_csv(PFA_COSING_SMILES_fisicoquimicas, 'PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS.csv')

#test_data_fingerprints.columns = test_data_fingerprints.columns.astype(str) #convertimos las columnas a strings
#print(test_data_fingerprints)


"""iupac cosgin"""
predicciones_COSING = leer_csv('PFA_COSING_SMILES_fisicoquimicas_FINGERPRINTS_PREDICCIONES.csv')


from modules.procesamiento.validar_smiles import obtener_iupac_desde_pubchem
predicciones_COSING = obtener_iupac_desde_pubchem(predicciones_COSING, 'SMILES')
guardar_csv(predicciones_COSING, 'predicciones_COSING.csv')
