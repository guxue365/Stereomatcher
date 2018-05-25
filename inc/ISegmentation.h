#pragma once

#include <opencv2/core.hpp>

class ISegmentation {
public:
	virtual ~ISegmentation() {};

	virtual cv::Mat Segment(const cv::Mat&) = 0;
};
