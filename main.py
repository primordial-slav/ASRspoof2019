import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans as MBKM
import os
import matplotlib.pyplot as  plt
#3
print("Loading files")
DIRECTORY= "C:\\Users\\papic\\Desktop\\boris\\data_train"
rawdata = pd.read_hdf('data.h5')
print("Loaded files.")
#print(rawdata.tail(10))
print(rawdata.shape)


x = rawdata.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
rawdata1 = pd.DataFrame(x_scaled)
print("Normalized data.")

model = MBKM(n_clusters=200, random_state=0,batch_size=10000)
kmeans=model.fit(rawdata1)
print("Finished making KMeans model.")

lista_fajl = []
lista_imena = []
iter  = 0

print("Predicting...")
for filename in os.listdir(DIRECTORY):
    iter+=1
    lista_klaster = np.zeros((200,), dtype=int)
    name = str(filename)
    lista_imena.append(name)
    data = pd.read_csv(f"C:\\Users\\papic\\Desktop\\boris\\data_train\\{name}")
    
    x = data.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    duzina = data.shape[0]
    for j in range(duzina):
        rez = kmeans.predict([df.iloc[j,:]])
        lista_klaster[rez] += 1
    lista_fajl.append(lista_klaster)
    
    if iter%500 == 0:
        print(f'Predicted {iter} files')
    

kk=pd.DataFrame(lista_fajl)
names = pd.DataFrame(lista_imena)
kk.sample(5)
print("Finished predicting.\n Saving...")
kk.to_hdf('final.h5',key='df',mode='w')
#names.to_hdf('names.h5',key='df',mode='w')
print("Saved.\n Saving names...")

names.to_csv('imena.csv',mode='w')
