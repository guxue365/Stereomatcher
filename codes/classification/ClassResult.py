import numpy as np
import json


def LoadClassResultFromFile(sFilename):
    with open(sFilename) as f:
        aData = json.load(f)

    aObjects = {}

    for oObject in aData:
        if oObject is None:
            continue

        oClassResult = ClassResult(oObject["id"], oObject["frame"], oObject["position"], oObject["label"])
        aObjects[oClassResult.mID] = oClassResult
    return aObjects

class ClassResult:
    mID = 0
    mFrame = 0
    mPosition = np.array([0, 0, 0])
    mLabel = 0

    def __init__(self, ID, Frame, Position, Label):
        self.mID = ID
        self.mFrame = Frame
        self.mPosition = np.array(Position)
        self.mLabel = Label

    def __str__(self):
        sResult = "Frame: {}\nID: {}\nLabel: {}\nPosition: {}|{}|{}".format(self.mFrame, self.mID, self.mLabel, self.mPosition[0], self.mPosition[1], self.mPosition[2])
        return sResult
    
    def todic(self):
        return {"id": self.mID, "frame": self.mFrame, "position": [self.mPosition[0], self.mPosition[1], self.mPosition[2]], "label": int(self.mLabel)}