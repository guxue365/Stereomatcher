#pragma once

#include "IPreprocessing.h"

class PreprocessMask : public IPreprocessing {
public:
	PreprocessMask();
	virtual ~PreprocessMask();

	cv::Mat Preprocess(const cv::Mat& rImage, int iSide);
};
