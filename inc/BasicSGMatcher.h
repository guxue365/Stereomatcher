#pragma once

#include "IStereoMatch.h"

class BasicSGMatcher : public IStereoMatch {
public:
	BasicSGMatcher();
	virtual ~BasicSGMatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};
