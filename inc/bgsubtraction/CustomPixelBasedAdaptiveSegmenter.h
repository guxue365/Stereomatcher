#pragma once

#include <vector>

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

	std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, int iNumRectangles);
	void FillMatrixWithRect(const cv::Mat& rOriginal, cv::Mat& rOutput, const std::vector<cv::Rect>& aROIs);
};
