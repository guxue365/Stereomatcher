import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

cameraMatrix1 = np.matrix([
    [2405.53394003284, 0, 0],
    [0, 2400.45634635254, 0],
    [1017.48338006588, 581.648876678341, 1]
]).transpose()

cameraMatrix2 = np.matrix([
    [2363.23085120949, 0, 0],
    [0, 2360.21245237401, 0],
    [1018.50992625364, 507.319902910560, 1]
]).transpose()

distCoeffs1 = np.matrix([-0.191784, 1.790997, -0.011658, -0.007531, -7.207820])

distCoeffs2 = np.matrix([ -0.072730519731600, -0.014992758230820, -0.010580053486047, -0.005443841816178, 0.347454004997192])

imageSize = (2048, 1024)

R = np.matrix([
    [0.999180452572029, -0.0367875884668465, 0.0168848018273893],
    [0.0381283095192544, 0.995434542927836, -0.0875003014809279],
    [-0.0135887899078690, 0.0880723797841485, 0.996021385667997]
]).transpose()

T = np.matrix([-66.9140725738007, 9.38902870045903, -8.75455185211030]).transpose()

alpha = 1.0
R1 = np.empty([3,3])
R2 = np.empty([3,3])
P1 = np.empty([3,3])
P2 = np.empty([3,3])
Q = np.empty([4,4])
newImageSize = (2048, 1024)

print(cameraMatrix1)
print(cameraMatrix2)
print(distCoeffs1)
print(distCoeffs2)
print(imageSize)
print(newImageSize)
print(R)
print(T)

#R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, 0, -1, newImageSize)
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T, R1, R2, P1, P2, Q, 0, alpha, newImageSize)

print("")
print("R1")
print("  data:", R1.flatten().tolist())
print("R2")
print("  data:", R2.flatten().tolist())
print("P1")
print("  data:", P1.flatten().tolist())
print("P2")
print("  data:", P2.flatten().tolist())
print("Q")
print("  data:", Q.flatten().tolist())