#pragma once

#include <IStereoMatch.h>

class CustomBlockMatcher : public IStereoMatch {
public:
	CustomBlockMatcher();
	virtual ~CustomBlockMatcher();

	void setBlockWidth(int iWidth);
	void setBlockHeight(int iHeight);
	void setNumDisparities(int iNumDisparities);
	void setValidTolerance(double dTolerance);
	void setUseStrictTolerance(bool bUseStrictTolerance);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockWidth;
	int miBlockHeight;
	double mdTolerance;
	bool mbUseStrictTolerance;

	cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight);
	cv::Mat ComputeCustomDisparityColor(const cv::Mat& rLeft, const cv::Mat& rRight);

	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockWidth, int iBlockHeight);
	double ComputeMatchingCostColor(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockWidth, int iBlockHeight);

	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance);
	bool isValidMinimumRelative(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance);
	bool isValidMinimumVar(double dValMin, int iIndexMin, const std::vector<double>& aValues);
};
