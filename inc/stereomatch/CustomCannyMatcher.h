#pragma once

#include <IStereoMatch.h>

class CustomCannyMatcher : public IStereoMatch {
public:
	CustomCannyMatcher();
	virtual ~CustomCannyMatcher();

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
};
