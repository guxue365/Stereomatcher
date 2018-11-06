import json
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from ClusterResult import *
from ClassResult import *

def getFeatureMatrix(aData):
    m = len(aData)
    n = 5

    X = np.zeros((m, n))
    for i in range(0, m):
        X[i] = aData[i].getFeatureArray()
    return X


#aObjects = LoadClusterResultFromFile("E:/result_bm_scene1/result_cluster.json")
aObjects = LoadClusterResultFromFile("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm_scene1/result_cluster.json")

M = np.loadtxt("traindata2.txt", delimiter=",")
YTrain = M[:, 0]
XTrain = M[:, 1:]

neigh = KNeighborsRegressor(n_neighbors=1)
neigh.fit(XTrain, YTrain)

X = getFeatureMatrix(aObjects)
Y = neigh.predict(X)

aClassResult = []

for i in range(0, len(aObjects)):
    oClassResult = ClassResult(aObjects[i].mID, aObjects[i].mFrame, aObjects[i].mPosition, Y[i])
    aClassResult.append(oClassResult.todic())

#with open("E:/result_bm_scene1/result_class.json", "w") as write_file:
with open("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm_scene1/result_class.json", "w") as write_file:
    json.dump(aClassResult, write_file, indent=4)