import pandas as pd
import matplotlib.pyplot as plt
import os
#5

rawdata = pd.read_hdf("final.h5")
q = pd.read_csv("imena.csv")
train_labels = pd.read_csv("train_labels.csv")


train_labels.columns = ['no','names','y']
names = q.iloc[:,1]
names.replace(to_replace=r'(LA_T_\d{7}).csv',value=r'\1',inplace = True,regex=True)
df = rawdata.rename(names,axis=0)

suma = df.sum(axis=1)   #Normalizacija
df = df.div(suma, axis=0)


train_labels.set_index('names',inplace=True)
train_labels.drop('no',axis=1,inplace=True)
#print(train_labels.sample(10))
#print(train_labels.shape)
result = pd.concat([train_labels,df],axis=1,sort=False)
result.iloc[:,0].replace(to_replace='spoof',value=0,inplace = True)
result.iloc[:,0].replace(to_replace='bonafide',value=1,inplace = True)
indBonafide = result.index[result['y'] == 1].tolist()
indSpoof = result.index[result['y'] == 0].tolist()
#print(indBonafide)
#print(len(indBonafide))
#print(len(indSpoof))
dfBonafide = df.loc[indBonafide,:]
dfSpoof = df.loc[indSpoof,:]

#print(dfBonafide.shape)
#print(dfSpoof.shape)
#print(dfBonafide.head)

for i in range(200):
    plt.figure()
    plt.boxplot([dfBonafide.iloc[:,i],dfSpoof.iloc[:,i]])
    plt.title(f"Histogram za {i} obelezje")
    plt.xlabel(["Bonafide","Spoof"])
    plt.ylabel("Broj pojavljivanja")
    #plt.show()
    plt.savefig(f"C:\\Users\\Boris\\Desktop\\ASRspoof\\histogrami\\histogram_{i}_obelezje")
