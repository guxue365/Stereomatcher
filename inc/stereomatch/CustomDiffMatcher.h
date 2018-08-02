#pragma once

#include <IStereoMatch.h>

class CustomDiffMatcher : public IStereoMatch {
public:
	CustomDiffMatcher();
	virtual ~CustomDiffMatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};