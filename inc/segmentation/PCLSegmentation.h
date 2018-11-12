#pragma once

#include "ISegmentation.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class PCLSegmentation : virtual public ISegmentation {
public:
	PCLSegmentation();
	virtual ~PCLSegmentation();

	virtual std::vector<Cluster> Segment(const cv::Mat& rImage);
};
