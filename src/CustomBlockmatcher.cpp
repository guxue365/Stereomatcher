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

		oDisparityOpenCV = ComputeOpenCVDisparity(oFrameLeft, oFrameRight);
		oDisparityCustom = ComputeCustomDisparity(oFrameLeft, oFrameRight);

		applyColorMap(oDisparityOpenCV, oDisparityOpenCV, COLORMAP_JET);
		imshow("Left Image", oFrameLeft);
		imshow("Right Image", oFrameRight);
		imshow("Disparity OpenCV", oDisparityOpenCV);
		imshow("Disparity Custom", oDisparityCustom);

		char c = (char)waitKey(1000);
		if (c == 27) 	break;
	}

    return 0;
}

cv::Mat ComputeOpenCVDisparity(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<StereoBM> pStereoBM = StereoBM::create(0, 7);
	pStereoBM->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U);

	return oResult;
}

cv::Mat ComputeCustomDisparity(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());

	size_t m = (size_t)rLeft.rows;
	size_t n = (size_t)rLeft.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (size_t i = 3; i < m-3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (size_t j = 3; j < n-3; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost
			vector<double> aMatchingCost(n);
			for (size_t k = 3; k < n-3; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				aMatchingCost[k] = ComputeMatchingCost(i, j, k, rLeft, rRight);
			}
			// find k with minimum cost and assign to result at (i, j)
			size_t iMatchingColRight = (min_element(aMatchingCost.begin(), aMatchingCost.end())-aMatchingCost.begin());

			oResult.at<uchar>(i, j) = (uchar)(abs(j - iMatchingColRight));
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

			double dLeft = (double)rLeft.at<uchar>(iCurrentRow, iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>(iCurrentRow, iCurrentColRight);
			dResult += sqrt((dLeft - dRight)*(dLeft - dRight));
		}
	}

	return dResult;
}
