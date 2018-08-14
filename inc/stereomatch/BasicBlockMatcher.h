#pragma once

#include "IStereoMatch.h"

class BasicBlockMatcher : public IStereoMatch {
public:
	BasicBlockMatcher();
	virtual ~BasicBlockMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;
};
