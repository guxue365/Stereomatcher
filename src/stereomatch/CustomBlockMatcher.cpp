#include <stereomatch/CustomBlockMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomBlockMatcher::CustomBlockMatcher() :
	miBlockSize(9),
	miNumDisparities(64) {

}

CustomBlockMatcher::~CustomBlockMatcher() {

}

void CustomBlockMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 == 1);

	miBlockSize = iBlockSize;
}

void CustomBlockMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat CustomBlockMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.size() == rRight.size());
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U || rLeft.type() == CV_8UC3);

	if (rLeft.type() == CV_8UC3) {
		return ComputeCustomDisparityColor(rLeft, rRight);
	}
	return ComputeCustomDisparityGray(rLeft, rRight);
}

cv::Mat CustomBlockMatcher::ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8U);

	int m = rLeft.rows;
	int n = rLeft.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

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
				double dCost = ComputeMatchingCostGray(i, j, k, rLeft, rRight, miBlockSize);
				aDisp[j - k] = dCost;
				
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

cv::Mat CustomBlockMatcher::ComputeCustomDisparityColor(const cv::Mat& rLeft, const cv::Mat& rRight) {
	assert(rLeft.rows == rRight.rows);
	assert(rLeft.cols == rRight.cols);
	assert(rLeft.type() == rRight.type());
	assert(rLeft.type() == CV_8UC3);

	int m = rLeft.rows;
	int n = rLeft.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));

	for (int i = 3; i < m - 3; ++i) {
		cout << "Row " << i << " from " << m << endl;
		for (int j = miNumDisparities; j < n - miNumDisparities; ++j) {
			// match pixel rLeft(i, j) to any Pixel(i, *) on the right image
			// -> iterate through row i on the right image and compute cost

			vector<double> aMatchingCost(n);
			double dMin = std::numeric_limits<double>::max();
			int iCustomDisp = -1;
			vector<double> aDisp(miNumDisparities);
			for (int k = j - miNumDisparities + 1; k < j + 1; ++k) {
				//compute cost for pixel (i, j) and (i, k)
				double dCost = ComputeMatchingCostColor(i, j, k, rLeft, rRight, miBlockSize);
				aDisp[j - k] = dCost;
				
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

double CustomBlockMatcher::ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize) {
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

double CustomBlockMatcher::ComputeMatchingCostColor(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize) {
	assert(rLeft.type() == CV_8UC3);
	assert(rRight.type() == CV_8UC3);

	double dResult = 0.0;

	for (int i = 0; i < iBlockSize; ++i) {
		int iCurrentRow = iRow + i - iBlockSize / 2;
		if (iCurrentRow < 0 || iCurrentRow >= rLeft.rows)	continue;

		for (int j = 0; j < iBlockSize; ++j) {
			int iCurrentColLeft = iColLeft + j - iBlockSize / 2;
			int iCurrentColRight = iColRight + j - iBlockSize / 2;
			if (iCurrentColLeft < 0 || iCurrentColRight<0 || iCurrentColLeft >= rLeft.cols || iCurrentColRight>rLeft.cols)	continue;

			Vec3b dLeft = (Vec3b)rLeft.at<Vec3b>((int)iCurrentRow, (int)iCurrentColLeft);
			Vec3b dRight = (Vec3b)rRight.at<Vec3b>((int)iCurrentRow, (int)iCurrentColRight);
			double d11 = (double)dLeft[0];
			double d12 = (double)dLeft[1];
			double d13 = (double)dLeft[2];

			double d21 = (double)dRight[0];
			double d22 = (double)dRight[1];
			double d23 = (double)dRight[2];

			dResult += abs(d11 - d21) + abs(d12 - d22) + abs(d13 - d23);
		}
	}

	return dResult;
}

bool CustomBlockMatcher::isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	double eps = 15.0;
	for (size_t l = 0; l < aValues.size(); ++l) {
		if (l == iIndexMin)		continue;
		if (abs(aValues[l] - dValMin) < eps) {
			return false;
		}
	}
	return true;
}

bool CustomBlockMatcher::isValidMinimumVar(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	int n = 10;

	vector<double> aCopy(aValues);
	aCopy[iIndexMin] = std::numeric_limits<double>::max();

	sort(aCopy.begin(), aCopy.end());
	aCopy.resize(n);

	double dMean = 0.0;
	for (auto& dVal : aCopy) {
		dMean += dVal;
	}
	dMean /= (double)(aCopy.size());

	double dVar = 0.0;
	for (auto& dVal : aCopy) {
		dVar += (dVal - dMean)*(dVal - dMean);
	}
	dVar = sqrt(dVar / (double)(aCopy.size() - 1));

	if (dValMin + sqrt(dVar) < aCopy[0]) {
		return true;
	}

	return false;
}

bool CustomBlockMatcher::isValidMinimumStrict2(double dValMin, int iIndexMin, const std::vector<double>& aValues) {
	vector<double> aCopy(aValues);

	double eps = 0.01;

	sort(aCopy.begin(), aCopy.end());

	if (abs((aCopy[0] - aCopy[1]) / aCopy[0]) < eps) {
		return false;
	}
	return true;
}
