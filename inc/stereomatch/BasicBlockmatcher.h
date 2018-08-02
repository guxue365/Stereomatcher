#pragma once

#include "IStereoMatch.h"

class BasicBlockMatcher : public IStereoMatch {
public:
	BasicBlockMatcher();
	virtual ~BasicBlockMatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};
