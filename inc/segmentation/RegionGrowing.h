#pragma once

#include "ISegmentation.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class RegionGrowing : virtual public ISegmentation {
public:
	RegionGrowing();
	virtual ~RegionGrowing();

	virtual std::vector<Cluster> Segment(const cv::Mat& rImage);
public:
	cv::Mat GrowRegion(const cv::Mat& rInput, uchar& rNumLabel);
	std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed);
};
