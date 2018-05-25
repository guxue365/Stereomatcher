#pragma once

#include "IBackgroundSubtraction.h"

#include <bgslibrary.h>

class CustomFrameDifference : public IBackgroundSubtraction {
public:
	CustomFrameDifference();
	virtual ~CustomFrameDifference();

	cv::Mat SubtractLeft(const cv::Mat& rImage);
	cv::Mat SubtractRight(const cv::Mat& rImage);
private:
	IBGS* mpBGSLeft;
	IBGS* mpBGSRight;
};
