import numpy as np
import json

def LoadClusterResultFromFile(sFilename):
    with open(sFilename) as f:
        aData = json.load(f)

    aObjects = {}

    for oData in aData:
        if oData is None:
            continue
        iFrame = oData[0]["frame"]

        for oObject in oData:
            oClusterResult = ClusterResult(oObject["id"], iFrame, oObject["position"], oObject["dimension"], oObject["eccentricity"])
            aObjects[oClusterResult.mID] = oClusterResult
    return aObjects

class ClusterResult:
    mID = 0
    mFrame = 0
    mPosition = np.array([0, 0, 0])
    mDimension = np.array([0, 0, 0])
    mEccentricity = np.array([0, 0])

    def __init__(self, ID, Frame, Position, Dimension, Eccentricity):
        self.mID = ID
        self.mFrame = Frame
        self.mPosition = np.array(Position)
        self.mDimension = np.array(Dimension)
        self.mEccentricity = np.array(Eccentricity)

    def __str__(self):
        sResult = "Frame: {}\nID: {}\n".format(self.mFrame, self.mID)
        return sResult

    def getFeatureArray(self):
        oResult = np.zeros(5)
        oResult[0] = self.mDimension[0]
        oResult[1] = self.mDimension[1]
        oResult[2] = self.mDimension[2]
        oResult[3] = self.mEccentricity[0]
        oResult[4] = self.mEccentricity[1]
        return oResult