#pragma once

#include <IImageLoader.h>

#include <vector>
#include <string>

#include <opencv2/videoio.hpp>

class CustomImageloader : public IImageLoader {
public:
	CustomImageloader();
	virtual ~CustomImageloader();

	bool Init(const std::string& sFolder);

	cv::Mat getNextLeftImage();
	cv::Mat getNextRightImage();
private:
	cv::VideoCapture moLeftImages;
	cv::VideoCapture moRightImages;
};
