#include <stereomatch/CustomPyramidMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomPyramidMatcher::CustomPyramidMatcher(IStereoMatch* pCoarseGridMatcher) : 
	miBlockSize(9),
	miNumDisparities(64),
	mpCoarseGridMatcher(pCoarseGridMatcher) {

}

CustomPyramidMatcher::~CustomPyramidMatcher() {

}

void CustomPyramidMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 == 1);

	miBlockSize = iBlockSize;
}

void CustomPyramidMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat CustomPyramidMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	Mat oLeftReduced;
	Mat oRightReduced;
	Mat oDisparityReduced;

	resize(rLeft, oLeftReduced, cv::Size(), 1.0 / 8.0, 1.0 / 8.0, INTER_LINEAR);
	resize(rRight, oRightReduced, cv::Size(), 1.0 / 8.0, 1.0 / 8.0, INTER_LINEAR);

	oDisparityReduced = mpCoarseGridMatcher->Match(oLeftReduced, oRightReduced);

	resize(oDisparityReduced, oDisparityReduced, rLeft.size(), 0.0, 0.0, INTER_LINEAR);

	oDisparityReduced *= 8;

	oResult = ComputeDisparityPyramid(oDisparityReduced, 8, rLeft, rRight);

	return oResult;
}

cv::Mat CustomPyramidMatcher::ComputeDisparityPyramid(const cv::Mat& rPrecomputed, int iScaleSize, const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	int m = rLeft.rows;
	int n = rLeft.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (int i = 3; i < m - 3; ++i) {
		for (int j = 0; j < n - miNumDisparities; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost

			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(2 * iScaleSize);
			int iPrecomputedDisp = (int)rPrecomputed.at<uchar>(i, j);
			if (iPrecomputedDisp == 0)		continue;
			for (int k = j - iPrecomputedDisp - iScaleSize + 1; k < j - iPrecomputedDisp + iScaleSize + 1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostGray(i, j, k, rLeft, rRight, miBlockSize);

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

double CustomPyramidMatcher::ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize) {
	assert(rLeft.type() == CV_8U);
	assert(rRight.type() == CV_8U);

	double dResult = 0.0;

	for (int i = 0; i < iBlockSize; ++i) {
		int iCurrentRow = iRow + i - iBlockSize / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBlockSize; ++j) {
			int iCurrentColLeft = iColLeft + j - iBlockSize / 2;
			int iCurrentColRight = iColRight + j - iBlockSize / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			double dLeft = (double)rLeft.at<uchar>((int)iCurrentRow, (int)iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>((int)iCurrentRow, (int)iCurrentColRight);
			dResult += abs(dLeft - dRight);
		}
	}

	return dResult;
}

bool CustomPyramidMatcher::isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	double eps = 15.0;
	for (size_t l = 0; l < aValues.size(); ++l) {
		if (l == iIndexMin)		continue;
		if (abs(aValues[l] - dValMin) < eps) {
			return false;
		}
	}
	return true;
}
