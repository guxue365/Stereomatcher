#pragma once

#include "ISegmentation.h"

#include "RegionGrowing.h"
#include "PCLSegmentation.h"

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

class TwoStepSegmentation : public RegionGrowing, public PCLSegmentation {
public:
	TwoStepSegmentation();
	virtual ~TwoStepSegmentation();

	std::vector<Cluster> Segment(const cv::Mat& rImage);
private:
	std::vector<cv::Rect2i> ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions);
	std::vector<cv::Rect2i> MergeRegions(const std::vector<cv::Rect2i>& aRegions);
};
