#pragma once

#include "IPostprocessing.h"

class BasePostprocessor : public IPostProcessing {
public:
	BasePostprocessor();
	virtual ~BasePostprocessor();

	cv::Mat Postprocess(const cv::Mat& rImage);
};
