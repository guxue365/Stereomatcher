#pragma once

#include <IStereoMatch.h>

class CustomDiffMatcher : public IStereoMatch {
public:
	CustomDiffMatcher();
	virtual ~CustomDiffMatcher();

	void setBlockSize(int iBlockSize);
	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockSize;
};