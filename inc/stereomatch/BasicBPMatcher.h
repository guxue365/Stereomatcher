#pragma once

#include "IStereoMatch.h"

class BasicBPMatcher : public IStereoMatch {
public:
	BasicBPMatcher();
	virtual ~BasicBPMatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};
