#pragma once

#include "ISegmentation.h"

class PCLSegmentation : public ISegmentation {
public:
	PCLSegmentation();
	virtual ~PCLSegmentation();

	//cv::Mat Segment(const cv::Mat& rImage);
	std::vector<Cluster> Segment(const cv::Mat& rImage);
};
