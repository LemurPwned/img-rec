import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
from string import maketrans 

filename = "D:\\Dokumenty\\image-recognition\\parameters\\vectorsL.txt"
filename2 = "D:\\Dokumenty\\image-recognition\\parameters\\vectorsR.txt"

filename_final = "D:\\Dokumenty\\image-recognition\\parameters\\vectors_dataset.txt"
cols = ["X", "angle", "norm"]

dataL = pd.read_csv(filename, sep=";", header = None)
dataL.columns = cols

dataR = pd.read_csv(filename2, sep=";", header = None)
dataR.columns = cols


dataJ = dataL.join(dataR,lsuffix="Left", rsuffix="Right")
#labels = pd.DataFrame(np.zeros(data.shape[0]), columns=["Class"])

k_meansL = KMeans(n_clusters = 3)
k_meansL.fit(dataL)


k_meansR = KMeans(n_clusters = 3)
k_meansR.fit(dataR)

k_meansJ = KMeans(n_clusters = 4)
k_meansJ.fit(dataJ)

labelsL = pd.DataFrame(k_meansL.labels_, columns=["Class"])
labelsR = pd.DataFrame(k_meansR.labels_, columns=["Class"])

labelsJ = pd.DataFrame(k_meansJ.labels_, columns=["Class"])

dataL = dataL.join(labelsL)
dataR = dataR.join(labelsR)
dataJ = dataJ.join(labelsJ)

print dataL.describe()
print dataR.describe()
print dataJ.describe()

'''
#L, R, S, W
data = dataL.join(labelsL)

intab = "0123"
#don't know really, try it
outtab = "SRWL"
trans_table = maketrans(intab, outtab)
#data['Class'] = str(data['Class']).translate(trans_table)
'''

dataJ.to_csv(filename_final)
