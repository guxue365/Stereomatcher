#pragma once

#include "IBackgroundSubtraction.h"

#include <bgslibrary.h>

class CustomPixelBasedAdaptiveSegmenter : public IBackgroundSubtraction {
public:
	CustomPixelBasedAdaptiveSegmenter();
	virtual ~CustomPixelBasedAdaptiveSegmenter();

	cv::Mat SubtractLeft(const cv::Mat& rImage);
	cv::Mat SubtractRight(const cv::Mat& rImage);
private:
	IBGS* mpBGSLeft;
	IBGS* mpBGSRight;
};
