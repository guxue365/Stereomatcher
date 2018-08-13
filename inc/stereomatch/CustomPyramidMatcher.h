#pragma once

#include "IStereoMatch.h"

class CustomPyramidMatcher : public IStereoMatch {
public:
	CustomPyramidMatcher(IStereoMatch* pCoarseGridMatcher);
	virtual ~CustomPyramidMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);

private:
	int miNumDisparities;
	int miBlockSize;

	IStereoMatch* mpCoarseGridMatcher;

	cv::Mat ComputeDisparityPyramid(const cv::Mat& rPrecomputed, int iScaleSize, const cv::Mat& rLeft, const cv::Mat& rRight);
	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize);
	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues);
};
