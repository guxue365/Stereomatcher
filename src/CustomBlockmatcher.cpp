#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

cv::Mat ComputeOpenCVDisparity(const cv::Mat& rLeft, const cv::Mat& rRight);
cv::Mat ComputeCustomDisparity(const cv::Mat& rLeft, const cv::Mat& rRight);
double ComputeMatchingCost(size_t iRow, size_t iColLeft, size_t iColRight, const cv::Mat& rLeft, const cv::Mat& rRight);

int main()
{
	VideoCapture oFilestreamLeft("E:/dataset_kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("E:/dataset_kitti/data_scene_flow/training/image_3/%06d_10.png");

	if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	Mat oFrameLeft;
	Mat oFrameRight;
	Mat oDisparityOpenCV;
	Mat oDisparityCustom;

	for(int iFrame=0; ; ++iFrame) {
		oFilestreamLeft >> oFrameLeft;
		oFilestreamRight >> oFrameRight;

		if (oFrameLeft.empty() || oFrameRight.empty())	break;

		cvtColor(oFrameLeft, oFrameLeft, COLOR_BGR2GRAY);
		cvtColor(oFrameRight, oFrameRight, COLOR_BGR2GRAY);

		int iTargetRow = 200;
		int iTargetCol = 401;

		oDisparityOpenCV = ComputeOpenCVDisparity(oFrameLeft, oFrameRight);

		cout << "Disparity by OpenCV BM: " << (int)oDisparityOpenCV.at<uchar>(iTargetRow, iTargetCol) << endl;

		double dMin = std::numeric_limits<double>::max();
		int iCustomDisp = -1;
		for (int j = iTargetCol-128; j < iTargetCol; ++j) {
			double dCost = ComputeMatchingCost(iTargetRow, iTargetCol, j, oFrameLeft, oFrameRight);
			cout << "Computing cost for Disparity " << iTargetCol-j<<": "<<dCost << endl;

			if (dCost < dMin) {
				dMin = dCost;
				iCustomDisp = iTargetCol - j;
			}

			/*Rect rLeft(iTargetCol - 1, iTargetRow - 1, 3, 3);
			Rect rRight(j - 1, iTargetRow - 1, 3, 3);

			cout << j-iTargetCol << endl;
			cout << "Left: " << endl<<oFrameLeft(rLeft) << endl;
			cout << "Right: " << endl << oFrameRight(rLeft) << endl;*/
		}

		cout << "Custom Disp: " << iCustomDisp << endl;

		oDisparityCustom = ComputeCustomDisparity(oFrameLeft, oFrameRight);

		applyColorMap(oDisparityOpenCV, oDisparityOpenCV, COLORMAP_JET);
		applyColorMap(oDisparityCustom, oDisparityCustom, COLORMAP_JET);
		//imshow("Left Image", oFrameLeft);
		//imshow("Right Image", oFrameRight);
		imshow("Disparity OpenCV", oDisparityOpenCV);
		imshow("Disparity Custom", oDisparityCustom);

		char c = (char)waitKey();
		if (c == 27) 	break;
	}

    return 0;
}

cv::Mat ComputeOpenCVDisparity(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<StereoBM> pStereoBM = StereoBM::create(0, 7);
	pStereoBM->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0/16.0);

	return oResult;
}

cv::Mat ComputeCustomDisparity(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());

	size_t m = (size_t)rLeft.rows;
	size_t n = (size_t)rLeft.cols;

	cv::Mat oResult((int)m, (int)n, CV_8U, Scalar(0));

	for (size_t i = 3; i < m-3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (size_t j = 3; j < n-128; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost
			
			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			for (int k = j-128; k < j; ++k) {
				if (k < 4)		continue;
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCost(i, j, k, rLeft, rRight);
				if (dCost < dMin) {
					dMin = dCost;
					iCustomDisp = j - k;
				}
			}
			oResult.at<uchar>((int)i, (int)j) = (uchar)(iCustomDisp);
		}
	}
	return oResult;
} 

double ComputeMatchingCost(size_t iRow, size_t iColLeft, size_t iColRight, const cv::Mat& rLeft, const cv::Mat& rRight) {
	double dResult = 0.0;

	size_t iBoxSize = 7;
	for (size_t i = 0; i < iBoxSize; ++i) {
		size_t iCurrentRow = iRow + i - iBoxSize / 2;
		for (size_t j = 0; j < iBoxSize; ++j) {
			size_t iCurrentColLeft = iColLeft + j - iBoxSize / 2;
			size_t iCurrentColRight = iColRight + j - iBoxSize / 2;

			double dLeft = (double)rLeft.at<uchar>((int)iCurrentRow, (int)iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>((int)iCurrentRow, (int)iCurrentColRight);
			dResult += sqrt((dLeft - dRight)*(dLeft - dRight));
		}
	}

	return dResult;
}
