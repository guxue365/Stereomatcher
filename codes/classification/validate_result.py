import json
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from ClusterResult import *
from ClassResult import *

def CreateFrameObjectMap(aData):
    aResult = {}
    for iID in aData:
        oObject = aData[iID]
        if oObject.mFrame not in aResult:
            aResult[oObject.mFrame] = {}
        aResult[oObject.mFrame][oObject.mID] = oObject
    return aResult

aResultData = LoadClassResultFromFile("E:/result_bm_scene1/result_class.json")
aGTData = LoadClassResultFromFile("E:/result_bm_scene1/gt.json")

aResultObjectMap = CreateFrameObjectMap(aResultData)
aGTObjectMap = CreateFrameObjectMap(aGTData)

iTruePositive = 0
iFalsePositive = 0
iFalseNegative = 0
iErrorLabel = 0
iErrorCount = 0

for iFrame in aGTObjectMap:
    if iFrame not in aResultObjectMap:
        iErrorCount+=len(aGTObjectMap[iFrame])
        iFalseNegative+=len(aGTObjectMap[iFrame])
        continue
        
    for iID in aGTObjectMap[iFrame]:
        oGTObject = aGTObjectMap[iFrame][iID]
        nObjectFound = False
        for iResultID in aResultObjectMap[iFrame]:
            oResultObject = aResultObjectMap[iFrame][iResultID]
            dDist = np.linalg.norm(oGTObject.mPosition-oResultObject.mPosition, ord=2)
            if dDist<1000:
                nObjectFound = True
                del aResultObjectMap[iFrame][iID]
                if oGTObject.mLabel!=oResultObject.mLabel:
                    iErrorLabel+=1
                    iErrorCount+=1
                else:
                    iTruePositive+=1
                break
        if nObjectFound==False:
            iFalseNegative+=1
            iErrorCount+=1


for iFrame in aResultObjectMap:
    iFalsePositive+=len(aResultObjectMap[iFrame])
    iErrorCount+=len(aResultObjectMap[iFrame])

print("Overall Errors: {}\nTrue Positive: {}\nFalse Positive: {}\nFalse Negative: {}\nWrong Labels: {}".format(iErrorCount, iTruePositive, iFalsePositive, iFalseNegative, iErrorLabel))

