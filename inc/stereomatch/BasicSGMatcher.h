#pragma once

#include "IStereoMatch.h"

class BasicSGMatcher : public IStereoMatch {
public:
	BasicSGMatcher();
	virtual ~BasicSGMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;
};
