#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp> 

#include <bgslibrary.h>

using namespace cv;
using namespace std;

int main() {
	VideoCapture oCamera("E:/dataset_changedetection/dataset2014/dataset/baseline/highway/input/in%06d.jpg");
	if (!oCamera.isOpened()) {
		throw std::string("Error opening camera");
	}

	Mat oFrame;
	Mat oLastFrame;
	Mat oFrameDiff;
	Mat oForeground;
	Mat oForegroundMorph;

	oCamera >> oLastFrame;
	cvtColor(oLastFrame, oLastFrame, CV_BGR2GRAY);

	IBGS* mpBGS = new PixelBasedAdaptiveSegmenter();
	mpBGS->setShowOutput(false);
	cv::Mat oMask, oBGModel, oResult;

	Mat oStrucElement = getStructuringElement(MORPH_RECT, Size(3, 3));

	for (int iFrame = 0;; ++iFrame) {
		oCamera >> oFrame;
		cvtColor(oFrame, oFrame, CV_BGR2GRAY);

		if (oFrame.empty())	break;

		absdiff(oFrame, oLastFrame, oFrameDiff);
		threshold(oFrameDiff, oForeground, 15.0, 255.0, THRESH_BINARY);

		morphologyEx(oForeground, oForegroundMorph, MORPH_OPEN, oStrucElement);

		mpBGS->process(oFrame, oMask, oBGModel);

		imshow("Frame", oFrame);
		imshow("Diff", oFrameDiff);
		imshow("Foreground", oForeground);
		imshow("Foreground Morph", oForegroundMorph);
		imshow("PBAS Mask", oMask);

		Mat oResult;
		hconcat(oFrame, oFrameDiff, oResult);
		hconcat(oResult, oForeground, oResult);
		hconcat(oResult, oForegroundMorph, oResult);
		hconcat(oResult, oMask, oResult);

		imshow("Result", oResult);

		imwrite("result/result_" + std::to_string(iFrame) + ".png", oResult);

		if (waitKey(50) == 27)		break;

		oLastFrame = oFrame;
	}


	return 0;
}