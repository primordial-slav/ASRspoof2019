import re
import pandas as pd
#4
regex = re.compile(r"LA_\d{4}\s(LA_T_\d{7})\s-\s((\w{2}_\d)|(-))\s(spoof|bonafide)",re.VERBOSE)

textfile = open('C:\\Users\\Boris\\Desktop\\ASRspoof\\ASVspoof2019_LA_protocols\\ASVspoof2019.LA.cm.train.trn.txt', 'r')
filetext = textfile.read()
textfile.close()
matches = re.findall(regex, filetext)
col1 = []
col2 = []
for i in range(len(matches)):
    col1.append(matches[i][0])
    col2.append(matches[i][4])

col1 = pd.DataFrame(col1)
col2 = pd.DataFrame(col2)
all = pd.concat([col1,col2],axis=1)
print(all.sample(10))
all.to_csv("C:\\Users\\Boris\\Desktop\\ASRspoof\\train_labels.csv")
