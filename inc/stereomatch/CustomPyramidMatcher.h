#pragma once

#include "IStereoMatch.h"

class CustomPyramidMatcher : public IStereoMatch {
public:
	CustomPyramidMatcher(IStereoMatch* pCoarseGridMatcher);
	virtual ~CustomPyramidMatcher();

	void setBlockWidth(int iWidth);
	void setBlockHeight(int iHeight);
	void setNumDisparities(int iNumDisparities);
	void setValidTolerance(double dTolerance);

	void setScalingWidth(double dScalingWidth);
	void setScalingHeight(double dScalingHeight);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);

private:
	int miNumDisparities;
	int miBlockWidth;
	int miBlockHeight;
	double mdTolerance;
	double mdScalingWidth;
	double mdScalingHeight;

	IStereoMatch* mpCoarseGridMatcher;

	cv::Mat ComputeDisparityPyramid(const cv::Mat& rPrecomputed, int iScaleSize, const cv::Mat& rLeft, const cv::Mat& rRight);
	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockWidth, int iBlockHeight);
	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance);
};
