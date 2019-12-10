#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[ ]:


#Calculamos la regresion con computacion cuántica


# In[2]:


DataPred = DataNorm


# In[4]:


DataPred['Label'] = data.loc[:,['wkn_500340']]


# In[5]:


print(DataPred)


# In[ ]:


#Calculamos el LHH de la matriz


# In[256]:


Dmatrix = DataPred.iloc[:,0:2]


# In[257]:


b = np.zeros((1107, 1105))


# In[258]:


matrix = np.append(Dmatrix.to_numpy(), b, axis = 1)


# In[274]:


np.shape(matrix)


# In[238]:


#matrix = Dmatrix.to_numpy().reshape(1107,2)


# In[262]:


matrix = matrix.tolist()


# In[276]:


DVector = DataPred.iloc[:,-1]


# In[266]:


vector = DVector.to_numpy().reshape(1107,1)


# In[277]:


vector = DVector.tolist()


# In[278]:


np.shape(vector)


# In[269]:


from qiskit.aqua import run_algorithm
from qiskit.aqua.input import LinearSystemInput
from qiskit.quantum_info import state_fidelity
from qiskit.aqua.algorithms.classical import ExactLSsolver
import numpy as np


# In[270]:


params = {
    'problem': {
        'name': 'linear_system'
    },
    'algorithm': {
        'name': 'HHL'
    },
    'eigs': {
        'expansion_mode': 'suzuki',
        'expansion_order': 2,
        'name': 'EigsQPE',
        'num_ancillae': 3,
        'num_time_slices': 50
    },
    'reciprocal': {
        'name': 'Lookup'
    },
    'backend': {
        'provider': 'qiskit.BasicAer',
        'name': 'statevector_simulator'
    }
}

def fidelity(hhl, ref):
    solution_hhl_normed = hhl / np.linalg.norm(hhl)
    solution_ref_normed = ref / np.linalg.norm(ref)
    fidelity = state_fidelity(solution_hhl_normed, solution_ref_normed)
    print("fidelity %f" % fidelity)


# In[205]:


matrix = [[1.3, 0.5], [-0.3, -2.4]]
vector = [1.5, 4.4]


# In[279]:


params3 = params
params3['input'] = {
    'name': 'LinearSystemInput',
    'matrix': matrix,
    'vector': vector
}
params3['reciprocal'] = {
    'negative_evals': True
}
params3['eigs'] = {
    'negative_evals': True
}


# In[ ]:


result = run_algorithm(params3)
print("solution ", np.round(result['solution'], 5))

result_ref = ExactLSsolver(matrix, vector).run()
print("classical solution ", np.round(result_ref['solution'], 5))

print("probability %f" % result['probability_result'])
fidelity(result['solution'], result_ref['solution'])


# In[ ]:




