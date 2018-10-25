#pragma once

#include <IStereoMatch.h>

class CustomMultiBoxMatcher : public IStereoMatch {
public:
	CustomMultiBoxMatcher();
	virtual ~CustomMultiBoxMatcher();

	void setBlockWidth(int iWidth);
	void setBlockHeight(int iHeight);
	void setNumDisparities(int iNumDisparities);
	void setBoxWidthScaling(double dBoxWidthScaling);
	void setBoxHeightScaling(double dBoxHeightScaling);
	void setValidTolerance(double dTolerance);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockWidth;
	int miBlockHeight;
	double mdBoxWidthScaling;
	double mdBoxHeightScaling;
	double mdTolerance;

	cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight);

	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockWidth, int iBlockHeight);

	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance);
};
