#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp> 

using namespace cv;
using namespace std;

int main() {
	VideoCapture oCamera(0);
	if (!oCamera.isOpened()) {
		throw std::string("Error opening camera");
	}

	Mat oFrame;
	Mat oLastFrame;
	Mat oFrameDiff;
	Mat oForeground;

	oCamera >> oLastFrame;
	cvtColor(oLastFrame, oLastFrame, CV_BGR2GRAY);

	for (int iFrame = 0;; ++iFrame) {
		oCamera >> oFrame;
		cvtColor(oFrame, oFrame, CV_BGR2GRAY);

		if (oFrame.empty())	break;

		absdiff(oFrame, oLastFrame, oFrameDiff);
		threshold(oFrameDiff, oForeground, 15.0, 255.0, THRESH_BINARY);

		imshow("Frame", oFrame);
		imshow("Diff", oFrameDiff);
		imshow("Foreground", oForeground);

		if (waitKey(50) == 27)		break;

		oLastFrame = oFrame;
	}


	return 0;
}