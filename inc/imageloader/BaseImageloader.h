#pragma once

#include <IImageLoader.h>

#include <vector>
#include <string>

#include <opencv2/videoio.hpp>

class BaseImageloader : public IImageLoader {
public:
	BaseImageloader();
	virtual ~BaseImageloader();

	bool Init(const std::string& sFolder);

	cv::Mat getNextLeftImage();
	cv::Mat getNextRightImage();
private:
	cv::VideoCapture moLeftImages;
	cv::VideoCapture moRightImages;
};
