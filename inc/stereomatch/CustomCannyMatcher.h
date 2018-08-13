#pragma once

#include <IStereoMatch.h>

class CustomCannyMatcher : public IStereoMatch {
public:
	CustomCannyMatcher();
	virtual ~CustomCannyMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;
};