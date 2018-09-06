#pragma once

#include <IStereoMatch.h>

class CustomCannyMatcher : public IStereoMatch {
public:
	CustomCannyMatcher();
	virtual ~CustomCannyMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);
	void setThreshold1(double dThreshold1);
	void setThreshold2(double dThreshold2);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;
	double mdThreshold1;
	double mdThreshold2;
};