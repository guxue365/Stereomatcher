#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

class IBackgroundSubtraction {
public:
	virtual ~IBackgroundSubtraction() {};

	virtual cv::Mat SubtractLeft(const cv::Mat&) = 0;
	virtual cv::Mat SubtractRight(const cv::Mat&) = 0;
};
