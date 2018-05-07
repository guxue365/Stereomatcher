#pragma once

#include "IStereoMatch.h"

class BasicBlockmatcher : public IStereoMatch {
public:
	BasicBlockmatcher();
	virtual ~BasicBlockmatcher();

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
};
