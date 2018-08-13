#pragma once

#include <IStereoMatch.h>

class CustomBlockMatcher : public IStereoMatch {
public:
	CustomBlockMatcher();
	virtual ~CustomBlockMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;

	cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight);
	cv::Mat ComputeCustomDisparityColor(const cv::Mat& rLeft, const cv::Mat& rRight);

	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize);
	double ComputeMatchingCostColor(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, int iBlockSize);

	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues);
	bool isValidMinimumStrict2(double dValMin, int iIndexMin, const std::vector<double>& aValues);
	bool isValidMinimumVar(double dValMin, int iIndexMin, const std::vector<double>& aValues);
};
