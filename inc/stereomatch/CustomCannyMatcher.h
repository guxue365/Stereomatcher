#pragma once

#include <IStereoMatch.h>

class CustomCannyMatcher : public IStereoMatch {
public:
	CustomCannyMatcher();
	virtual ~CustomCannyMatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};