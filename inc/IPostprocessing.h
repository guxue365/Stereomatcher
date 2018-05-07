#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

class IPostProcessing {
public:
	virtual ~IPostProcessing() {};

	virtual cv::Mat Postprocess(const cv::Mat&) = 0;
};
