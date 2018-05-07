#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

class IStereoMatch {
public:
	virtual ~IStereoMatch() {};

	virtual cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight) = 0;
};
