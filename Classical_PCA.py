import os
import numpy as np
from sklearn import datasets
from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt

#Cargamos los datos
#os.chdir(r"D:\DeptIA\congresos\cuantica")
#data = pd.read_csv("DAX_PERFORMANCE_INDEX.csv", sep=';')
url = "https://raw.githubusercontent.com/ibonreinoso/qiskit-hackathon-bilbao-19/master/DAX_PERFORMANCE_INDEX.csv"
data = pd.read_csv(url, sep=';')
#Creamos un  dataset sin label para el no supervisado
# El primer paso es conocer nuestros datos.
# Realizamos una primera visualizacion.
# EL objetivo es precedir la dureza del cemento.cd
print(data)

# Visualizamos los 10 primero datos, de una manera mas comoda.
data.head(n=10)
#Creamos una pca
#primero estandarizamos los datos
# Standardizing the features
DataX = data.drop(['wkn_500340'], axis = 1)
DataX = DataX.loc[:,['wkn_515100', 'wkn_575200']]
print(DataX)

from sklearn.preprocessing import StandardScaler
DataNorm = StandardScaler().fit_transform(DataX)

DataNorm = pd.DataFrame(DataNorm)
print(DataNorm)

#PCA Projection to 2D
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(DataNorm)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])

print(pca.explained_variance_ratio_)  

plt.plot('principal component 1','principal component 2', 'bo', data=principalDf)
