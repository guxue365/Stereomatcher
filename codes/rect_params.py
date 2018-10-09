import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

cameraMatrix1 = np.matrix([
    [2.500744557379985e+03, 0, 0],
    [0, 2.535988340208410e+03, 0],
    [1.260414892498634e+03, 5.260783408403682e+02, 1]
]).transpose()

cameraMatrix2 = np.matrix([
    [2.472932850391490e+03, 0, 0],
    [0, 2.493700599346621e+03, 0],
    [1.284997418662109e+03, 5.092517594678745e+02, 1]
]).transpose()

distCoeffs1 = np.matrix([-0.358747279897952, -5.482777505594711, -0.026661723125205, 0.030967281664763, 21.789612196728840])

distCoeffs2 = np.matrix([ -0.608013415673151, -1.374768603631144, 0.014811903964771, 0.006712822591838, 5.002510805942969])

imageSize = (2048, 1024)

R = np.matrix([
    [0.998818097562346,  -0.040321229984821,   0.027140493629374],
    [0.041806070439880,   0.997523047404669,  -0.056568740227069],
    [-0.024792346728593,   0.057636518883543,   0.998029744664294]
]).transpose()

T = np.matrix([-4.778183165716090e+02, 50.433147703073980, -52.114246295362406]).transpose()

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