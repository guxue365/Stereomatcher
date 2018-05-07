#pragma once

#include <opencv2/core.hpp>

class IPreprocessing {
public:
	virtual ~IPreprocessing() {};

	virtual cv::Mat Preprocess(const cv::Mat&) = 0;
};
