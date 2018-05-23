#pragma once

#include <iostream>
#include <vector>

#include <opencv2/core.hpp>

class IImageLoader {
public:
	virtual ~IImageLoader() {};

	virtual bool Init(const std::string& sFolder) = 0;

	virtual cv::Mat getNextLeftImage() = 0;
	virtual cv::Mat getNextRightImage() = 0;
};
