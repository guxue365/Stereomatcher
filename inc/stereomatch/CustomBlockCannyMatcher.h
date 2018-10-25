#pragma once

#include <IStereoMatch.h>

class CustomBlockCannyMatcher : public IStereoMatch {
public:
	CustomBlockCannyMatcher();
	virtual ~CustomBlockCannyMatcher();

	void setBlockWidth(int iWidth);
	void setBlockHeight(int iHeight);
	void setNumDisparities(int iNumDisparities);
	void setValidTolerance(double dTolerance);
	void setThreshold1(double dThreshold1);
	void setThreshold2(double dThreshold2);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockWidth;
	int miBlockHeight;
	double mdTolerance;
	double mdThreshold1;
	double mdThreshold2;

	cv::Mat ComputeCustomDisparityGray(const cv::Mat& rLeft, const cv::Mat& rRight, const cv::Mat& rCannyLeft, const cv::Mat& rCannyRight);

	double ComputeMatchingCostGray(int iRow, int iColLeft, int iColRight, const cv::Mat& rLeft, const cv::Mat& rRight, const cv::Mat& rCannyLeft,
			const cv::Mat& rCannyRight, int iBlockWidth, int iBlockHeight);

	bool isValidMinimumStrict(double dValMin, int iIndexMin, const std::vector<double>& aValues, double dTolerance);
};
