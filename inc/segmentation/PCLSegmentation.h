#pragma once

#include "ISegmentation.h"

#include <pcl/point_types.h>

class PCLSegmentation : public ISegmentation {
public:
	PCLSegmentation();
	virtual ~PCLSegmentation();

	//cv::Mat Segment(const cv::Mat& rImage);
	std::vector<Cluster> Segment(const cv::Mat& rImage);
private:
	std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity);
};
