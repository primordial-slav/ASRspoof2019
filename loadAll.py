import os
import pandas as pd

#2
LOADDIR = "C:\\Users\\papic\\Desktop\\boris\\data_train"
iter = 0
lista = []
print("Loading files...")
for filename in os.listdir(LOADDIR):
    iter +=1
    name = str(filename)
    df = pd.read_csv(f"C:\\Users\\papic\\Desktop\\boris\\data_train\\{name}")
    lista.append(df)
    if iter%500 ==0:
        left = 25380 - iter
        print(f"Finished {iter} files. Left: {left}")

rawdata = pd.concat(lista)

rawdata.to_hdf('data.h5',key='df',mode='w')
print("Uspesno")
