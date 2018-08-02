#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <stereomatch/BasicBlockMatcher.h>
#include <stereomatch/BasicBPMatcher.h>
#include <stereomatch/BasicSGMatcher.h>
#include <stereomatch/CustomBlockMatcher.h>
#include <stereomatch/CustomPyramidMatcher.h>
#include <stereomatch/CustomDiffMatcher.h>
#include <stereomatch/CustomCannyMatcher.h>

#include <evaluation/EvaluateBPP.h>

using namespace std;
using namespace cv;

Mat oFrameLeftColor;
Mat oFrameLeftGray;
Mat oFrameRightColor;
Mat oFrameRightGray;

Mat oCustomDisparity;
Mat oDisparityGT;

Mat oEvalCustomDisp;

BasicBlockMatcher oBasicBlockMatcher;
BasicSGMatcher oBasicSGMatcher;
BasicBPMatcher oBasicBPMatcher;

EvaluateBPP oEvalBPP(10.0);

IStereoMatch* pStereoMatcher = &oBasicBlockMatcher;

void onMouse(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN)		return;

	cout << "Mouse Event at: " << x << " | " << y << endl;
	cout << "GT: " << (int)oDisparityGT.at<uchar>(y, x) << endl;
	cout << "CT: " << (int)oCustomDisparity.at<uchar>(y, x) << endl;
}

int main() {
	VideoCapture oFilestreamLeft("E:/dataset_kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("E:/dataset_kitti/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("E:/dataset_kitti/data_scene_flow/training/disp_noc_1/%06d_10.png");

	if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened() || !oFilestreamGT.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	

	for (int iFrame = 0; ; ++iFrame) {
		oFilestreamLeft >> oFrameLeftColor;
		oFilestreamRight >> oFrameRightColor;
		oFilestreamGT >> oDisparityGT;
		oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 256.0);

		if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;

		cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
		cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

		oCustomDisparity = pStereoMatcher->Match(oFrameLeftGray, oFrameRightGray);

		double dEval = oEvalBPP.Evaluate(oDisparityGT, oCustomDisparity);
		oEvalCustomDisp = oEvalBPP.getVisualRepresentation();

		cout << "BPP Error: " << dEval << endl;

		/*oCustomDisparity *= 3;
		oDisparityGT *= 3;

		applyColorMap(oCustomDisparity, oCustomDisparity, COLORMAP_JET);
		applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);*/

		imshow("Disparity GT", oDisparityGT);
		imshow("Disparity Custom", oCustomDisparity);
		imshow("Evaluation", oEvalCustomDisp);

		setMouseCallback("Disparity GT", onMouse);
		setMouseCallback("Disparity Custom", onMouse);
		setMouseCallback("Evaluation", onMouse);

		char c = (char)waitKey();
		if (c == 27) 	break;
	}
	return 0;
}