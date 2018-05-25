#pragma once

#include "ISegmentation.h"

class RegionGrowing : public ISegmentation {
public:
	RegionGrowing();
	virtual ~RegionGrowing();

	cv::Mat Segment(const cv::Mat& rImage);
private:
	void GrowRegion(const cv::Mat& rInput, cv::Mat& rResult);
	std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed);
};
