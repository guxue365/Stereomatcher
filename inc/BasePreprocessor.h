#pragma once

#include "IPreprocessing.h"

class BasePreprocessor : public IPreprocessing {
public:
	BasePreprocessor();
	virtual ~BasePreprocessor();

	cv::Mat Preprocess(const cv::Mat& rImage);
};
