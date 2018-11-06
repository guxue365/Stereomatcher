#pragma once

#include <IImageLoader.h>

#include <vector>
#include <string>

#include <opencv2/videoio.hpp>

class SkipImageloader : public IImageLoader {
public:
	SkipImageloader();
	virtual ~SkipImageloader();

	bool Init(const std::string& sFolder);

	cv::Mat getNextLeftImage();
	cv::Mat getNextRightImage();
private:
	cv::VideoCapture moLeftImages;
	cv::VideoCapture moRightImages;
};
