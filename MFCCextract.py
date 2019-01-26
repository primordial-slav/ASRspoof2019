from python_speech_features import mfcc
from python_speech_features.base import delta
import scipy.io.wavfile as wav
import pandas as pd
import os
#1
DIRECTORY = "C:\\Users\\Boris\\Desktop\\ASRspoof\\wav_train"
SAVEDIR = "C:\\Users\\Boris\\Desktop\\ASRspoof\\data_train"
iter =0
mfcc_column_names =[]
delta_column_names =[]
delta_delta_column_names=[]
for i in range(40):
    name_mfcc = f"mfcc_{i+1}"
    name_d = f"delta_{i+1}"
    name_dd = f"delta_delta_{i+1}"
    mfcc_column_names.append(name_mfcc)
    delta_column_names.append(name_d)
    delta_delta_column_names.append(name_dd)



for filename in os.listdir(DIRECTORY):
    name = str(filename)
    (rate,sig) = wav.read(f"C:\\Users\\Boris\\Desktop\\ASRspoof\\wav_train\\{name}")
    mfcc_feat = mfcc(sig,rate,winlen=0.025,winstep=0.005,numcep=40,nfilt=40)
    delta_1 = delta(mfcc_feat, N=1)
    delta_2 = delta(delta_1, N=1)
    frame1 = pd.DataFrame(mfcc_feat)
    frame2 = pd.DataFrame(delta_1)
    frame3 = pd.DataFrame(delta_2)

    df = pd.concat([frame1, frame2, frame3],axis=1)
    df.columns =  mfcc_column_names+delta_column_names+delta_delta_column_names
    #print(df.shape)
    iter += 1
    if iter%1000 == 0:
        print(f"Gotov {iter}")
    df.to_csv(f'{SAVEDIR}\\{os.path.splitext(filename)[0]}.csv', header=True, index=None, sep=',', mode='a',float_format='%.6f')
    '''if iter == 4:
        break'''
