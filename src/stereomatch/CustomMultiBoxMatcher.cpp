#include <stereomatch/CustomMultiBoxMatcher.h>.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomMultiBoxMatcher::CustomMultiBoxMatcher() :
	miNumDisparities(64),
	miBlockWidth(9),
	miBlockHeight(9),
	mdBoxWidthScaling(2.0),
	mdBoxHeightScaling(2.0),
	mdTolerance(15.0) {

}

CustomMultiBoxMatcher::~CustomMultiBoxMatcher() {

}


void CustomMultiBoxMatcher::setBlockWidth(int iBlockWidth) {
	assert(iBlockWidth>0);
	assert(iBlockWidth%2==1);

	miBlockWidth = iBlockWidth;
}

void CustomMultiBoxMatcher::setBlockHeight(int iBlockHeight) {
	assert(iBlockHeight>0);
	assert(iBlockHeight%2==1);

	miBlockHeight = iBlockHeight;
}

void CustomMultiBoxMatcher::setBoxWidthScaling(double dBoxWidthScaling) {
	assert(dBoxWidthScaling>1.0);

	mdBoxWidthScaling = dBoxWidthScaling;
}

void CustomMultiBoxMatcher::setBoxHeightScaling(double dBoxHeightScaling) {
	assert(dBoxHeightScaling>1.0);

	mdBoxHeightScaling = dBoxHeightScaling;
}

void CustomMultiBoxMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

void CustomMultiBoxMatcher::setValidTolerance(double dTolerance) {
	assert(dTolerance>0.0);

	mdTolerance = dTolerance;
}

cv::Mat CustomMultiBoxMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.size() == rRight.size());
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	return ComputeCustomDisparityGray(rLeft, rRight);
}

cv::Mat CustomMultiBoxMatcher::ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	int m = rLeft.rows;
	int n = rLeft.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	int iBlockWidthBox = (int)((double)(miBlockWidth)*mdBoxWidthScaling);
	if (iBlockWidthBox % 2 == 1)		iBlockWidthBox++;
	int iBlockHeightBox = (int)((double)(miBlockHeight)*mdBoxHeightScaling);
	if (iBlockHeightBox % 2 == 1)		iBlockHeightBox++;

	for (int i = 3; i < m - 3; ++i) {
		for (int j = miNumDisparities; j < n - miNumDisparities; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost

			if(rLeft.at<uchar>(i, j)==0)	continue;

			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(miNumDisparities);
			for (int k = j - miNumDisparities + 1; k < j + 1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostGray(i, j, k, rLeft, rRight, miBlockWidth, miBlockHeight);
				dCost+=ComputeMatchingCostGray(i, j, k, rLeft, rRight, iBlockWidthBox, miBlockHeight);
				dCost+=ComputeMatchingCostGray(i, j, k, rLeft, rRight, miBlockWidth, iBlockHeightBox);
				dCost+=ComputeMatchingCostGray(i, j, k, rLeft, rRight, iBlockWidthBox, iBlockHeightBox);
				aDisp[j - k] = dCost;
				if (dCost < dMin) {
					dMin = dCost;
					iCustomDisp = j - k;
				}
			}

			auto itMin = std::min_element(aDisp.begin(), aDisp.end());
			int iMin = (int)std::distance(aDisp.begin(), itMin);
			double dMinVal = *itMin;

			if (isValidMinimumStrict(dMinVal, iMin, aDisp, mdTolerance)) {
				oResult.at<uchar>(i, j) = (uchar)(iCustomDisp);
			}
		}
	}
	return oResult;
}

double CustomMultiBoxMatcher::ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockWidth, int iBlockHeight) {
	assert(rLeft.type() == CV_8U);
	assert(rRight.type() == CV_8U);

	double dResult = 0.0;

	for (int i = 0; i < iBlockHeight; ++i) {
		int iCurrentRow = iRow + i - iBlockHeight / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBlockWidth; ++j) {
			int iCurrentColLeft = iColLeft + j - iBlockWidth / 2;
			int iCurrentColRight = iColRight + j - iBlockWidth / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			double dLeft = (double)rLeft.at<uchar>((int)iCurrentRow, (int)iCurrentColLeft);
			double dRight = (double)rRight.at<uchar>((int)iCurrentRow, (int)iCurrentColRight);
			dResult += (dLeft - dRight)*(dLeft - dRight);
		}
	}

	return sqrt(dResult/(double)(iBlockWidth*iBlockHeight));
}



bool CustomMultiBoxMatcher::isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance) {
	for (size_t l = 0; l < aValues.size(); ++l) {
		if (l == iIndexMin)		continue;
		if (abs(aValues[l] - dValMin) < dTolerance || abs(aValues[l] - dValMin)/(abs(dValMin))<0.05) {
			return false;
		}
	}
	return true;
}
