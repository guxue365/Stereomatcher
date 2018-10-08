#!/usr/bin/env python3

'''
# Name: rect_text.py
# Author: Martin Herrmann
'''

# Test openCV rectifyStereo results

import cv2
import numpy as np
import sys
from matplotlib import pyplot as plt

if __name__ == '__main__':
    print('Rectify with opencv')

    #Input
    #cameraMatrix1 = np.matrix([[2381.732645, 0.000000, 1018.770273], [0.000000, 2376.728640, 564.777009], [0.000000, 0.000000, 1.000000]])
    cameraMatrix1 = np.matrix([
        [2405.53394003284, 0, 0],
        [0, 2400.45634635254, 0],
        [1017.48338006588, 581.648876678341, 1]
    ]).transpose()

    #cameraMatrix2 = np.matrix([[2340.879675, 0.000000, 1008.867552], [0.000000, 2335.456098, 509.483204], [0.000000, 0.000000, 1.000000]])
    cameraMatrix2 = np.matrix([
        [2363.23085120949, 0, 0],
        [0, 2360.21245237401, 0],
        [1018.50992625364, 507.319902910560, 1]
    ]).transpose()

    #distCoeffs1 = np.matrix([-0.160737, 1.254425, -0.012986, -0.006122, -4.978713])
    distCoeffs1 = np.matrix([-0.191784, 1.790997, -0.011658, -0.007531, -7.207820])

    #distCoeffs2 = np.matrix([-0.107755, 0.477548, -0.010597, -0.005455, -1.567498])
    distCoeffs2 = np.matrix([ -0.072730519731600, -0.014992758230820, -0.010580053486047, -0.005443841816178, 0.347454004997192])

    imageSize = (2048, 1024)

    #R = np.matrix([[0.999233557773272, 0.038062565720446, -0.009139918519620],[-0.037199786595769, 0.996023284739148, 0.080955494777486],[0.012184945506559, -0.080553444049370, 0.996675809757008]])
    R = np.matrix([
        [0.999180452572029, -0.0367875884668465, 0.0168848018273893],
        [0.0381283095192544, 0.995434542927836, -0.0875003014809279],
        [-0.0135887899078690, 0.0880723797841485, 0.996021385667997]
    ]).transpose()

    #T = np.matrix([[-4.781302063068419e+02],[66.170991687923570],[-80.597792795494980]]) / 100
    T = np.matrix([-66.9140725738007, 9.38902870045903, -8.75455185211030]).transpose()

    alpha = 1.0;
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
    print(Q)
    print(validPixROI1)
    print(validPixROI2)

    if len(sys.argv) == 3:
        imgL = cv2.imread(sys.argv[1])[0:imageSize[1],0:imageSize[0]]
        imgR = cv2.imread(sys.argv[2])[0:imageSize[1],0:imageSize[0]]

        map1, map2 = cv2.initUndistortRectifyMap(cameraMatrix1, distCoeffs1, R1, P1, newImageSize, cv2.CV_32FC1)
        map3, map4 = cv2.initUndistortRectifyMap(cameraMatrix2, distCoeffs2, R2, P2, newImageSize, cv2.CV_32FC1)

        imgLRect = cv2.remap(imgL, map1, map2, cv2.INTER_CUBIC)
        imgRRect = cv2.remap(imgR, map3, map4, cv2.INTER_CUBIC)
        imgJoined = np.concatenate((imgLRect, imgRRect), axis=1)

        for x in range(0, newImageSize[1], 100):
            for y in range(0, 2*newImageSize[0]):
                imgJoined[x, y] = 0

        cv2.imshow("Rectified image L", imgLRect)
        cv2.imshow("Rectified image R", imgRRect)
        cv2.imwrite("/tmp/testL.png", imgLRect)
        cv2.imwrite("/tmp/testR.png", imgRRect)
        cv2.imwrite("/tmp/testJ.png", imgJoined)

        min_disp = 0
        num_disp = 128
        window_size = 3
        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = 3,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 10,
            speckleWindowSize = 100,
            speckleRange = 32
        )

        disp = stereo.compute(imgLRect, imgRRect)
        plt.imshow(disp, 'gray')
        plt.show()

        while True:
            cv2.waitKey(0)
