#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudastereo.hpp>

using namespace std;
using namespace cv;

cv::Mat ComputeOpenCVBlockMatch(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize = 7);
cv::Mat ComputeOpenCVBeliefPropagation(const cv::Mat& rLeft, const cv::Mat& rRight);

cv::Mat ComputeDisparityPyramid(const cv::Mat& rPrecomputed, int iScaleSize, const cv::Mat& rLeft, const cv::Mat& rRight);
double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight);
bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues);

int main() {
	/*VideoCapture oFilestreamLeft("E:/dataset_kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("E:/dataset_kitti/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("E:/dataset_kitti/data_scene_flow/training/disp_noc_1/%06d_10.png");*/

	VideoCapture oFilestreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/kitti/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("/home/jung/2018EntwicklungStereoalgorithmus/data/kitti/data_scene_flow/training/disp_custom/%06d_10.png");

	if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened() || !oFilestreamGT.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	Mat oFrameLeftColor;
	Mat oFrameLeftGray;
	Mat oFrameRightColor;
	Mat oFrameRightGray;

	const int iNumGrids = 5;
	const double dGridScaling = 0.5;

	Mat oLeftReduced;
	Mat oRightReduced;
	Mat oDisparityReducedBM;
	Mat oDisparityReducedBP;
	Mat oDisparityPyramid;

	Mat oDisparityOpenCVBM;
	Mat oDisparityOpenCVBP;
	Mat oDisparityGT;

	for (int iFrame = 0; ; ++iFrame) {
		oFilestreamLeft >> oFrameLeftColor;
		oFilestreamRight >> oFrameRightColor;
		oFilestreamGT >> oDisparityGT;
		//oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 255.0);
		cvtColor(oDisparityGT, oDisparityGT, CV_BGR2GRAY);
		oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0);

		if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;

		cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
		cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

		resize(oFrameLeftGray, oLeftReduced, cv::Size(), 1.0 / 8.0, 1.0 / 8.0, INTER_LINEAR);
		resize(oFrameRightGray, oRightReduced, cv::Size(), 1.0 / 8.0, 1.0 / 8.0, INTER_LINEAR);

		oDisparityOpenCVBM = ComputeOpenCVBlockMatch(oFrameLeftGray, oFrameRightGray);
		oDisparityOpenCVBP = ComputeOpenCVBeliefPropagation(oFrameLeftGray, oFrameRightGray);
		oDisparityReducedBM = ComputeOpenCVBlockMatch(oLeftReduced, oRightReduced);
		oDisparityReducedBP = ComputeOpenCVBeliefPropagation(oLeftReduced, oRightReduced);

		resize(oDisparityReducedBM, oDisparityReducedBM, oFrameLeftGray.size(), 0.0, 0.0, INTER_LINEAR);
		resize(oDisparityReducedBP, oDisparityReducedBP, oFrameLeftGray.size(), 0.0, 0.0, INTER_LINEAR);

		oDisparityReducedBP *= 8;
		oDisparityReducedBM *= 8;

		oDisparityPyramid = ComputeDisparityPyramid(oDisparityReducedBP, 8, oFrameLeftGray, oFrameRightGray);
		//oDisparityPyramid = Mat::zeros(oDisparityReducedBP.size(), CV_8U);

		oDisparityOpenCVBM *= 3;
		oDisparityOpenCVBP *= 3;
		oDisparityReducedBM *= 3;
		oDisparityReducedBP *= 3;
		oDisparityPyramid *= 3;
		oDisparityGT *= 3;

		applyColorMap(oDisparityOpenCVBM, oDisparityOpenCVBM, COLORMAP_JET);
		applyColorMap(oDisparityOpenCVBP, oDisparityOpenCVBP, COLORMAP_JET);
		applyColorMap(oDisparityReducedBM, oDisparityReducedBM, COLORMAP_JET);
		applyColorMap(oDisparityReducedBP, oDisparityReducedBP, COLORMAP_JET);
		applyColorMap(oDisparityPyramid, oDisparityPyramid, COLORMAP_JET);
		applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);

		imshow("Left", oLeftReduced);
		imshow("Right", oRightReduced);
		imshow("Disp Blockmatching", oDisparityOpenCVBM);
		imshow("Disp Belief Propagation Reduced", oDisparityReducedBP);
		imshow("Disp Pyramid", oDisparityPyramid);
		imshow("Disparity GT", oDisparityGT);

		char c = (char)waitKey();
		if (c == 27) 	break;
	}

	return 0;
}

cv::Mat ComputeOpenCVBlockMatch(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize) {
	cv::Mat oResult;

	cv::Ptr<StereoBM> pStereoBM = StereoBM::create(0, iBoxSize);
	pStereoBM->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);

	return oResult;
}

cv::Mat ComputeOpenCVBeliefPropagation(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;
	cuda::GpuMat _pResult;


	cuda::GpuMat _pLeft(rLeft);
	cuda::GpuMat _pRight(rRight);

	cv::Ptr<cv::cuda::StereoBeliefPropagation> bp = cv::cuda::createStereoConstantSpaceBP();
	bp->compute(_pLeft, _pRight, _pResult);

	_pResult.download(oResult);

	oResult.convertTo(oResult, CV_8U);

	return oResult;
}

cv::Mat ComputeDisparityPyramid(const cv::Mat& rPrecomputed, int iScaleSize, const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	int m = rLeft.rows;
	int n = rLeft.cols;

	const int iMaxDisparity = 128;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (int i = 3; i < m - 3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (int j = 0; j < n - iMaxDisparity; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost

			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(2*iScaleSize);
			int iPrecomputedDisp = (int)rPrecomputed.at<uchar>(i, j);
			if (iPrecomputedDisp == 0)		continue;
			for (int k = j - iPrecomputedDisp - iScaleSize + 1; k < j - iPrecomputedDisp + iScaleSize + 1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostGray(i, j, k, rLeft, rRight);

				aDisp[j - k - iPrecomputedDisp + iScaleSize] = dCost;
				
				if (dCost < dMin) {
					dMin = dCost;
					iCustomDisp = j - k;
				}
			}

			auto itMin = std::min_element(aDisp.begin(), aDisp.end());
			int iMin = (int)std::distance(aDisp.begin(), itMin);
			double dMinVal = *itMin;

			if (isValidMinimumStrict(dMinVal, iMin, aDisp)) {
				oResult.at<uchar>(i, j) = (uchar)(iCustomDisp);
			}
		}
	}
	return oResult;
}

double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.type() == CV_8U);
	assert(rRight.type() == CV_8U);

	double dResult = 0.0;

	int iBoxSize = 7;
	for (int i = 0; i < iBoxSize; ++i) {
		int iCurrentRow = iRow + i - iBoxSize / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBoxSize; ++j) {
			int iCurrentColLeft = iColLeft + j - iBoxSize / 2;
			int iCurrentColRight = iColRight + j - iBoxSize / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			double dLeft = (double)rLeft.at<uchar>((int)iCurrentRow, (int)iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>((int)iCurrentRow, (int)iCurrentColRight);
			dResult += abs(dLeft - dRight);
		}
	}

	return dResult;
}

bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	double eps = 15.0;
	for (size_t l = 0; l < aValues.size(); ++l) {
		if (l == iIndexMin)		continue;
		if (abs(aValues[l] - dValMin) < eps) {
			return false;
		}
	}
	return true;
}
