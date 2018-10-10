#pragma once

#include <vector>

#include <opencv2/core.hpp>

struct Cluster {
	cv::Vec3d oPosition;
	cv::Vec3d aDimension;
	cv::Vec2d aEccentricity;
};

class ISegmentation {
public:
	virtual ~ISegmentation() {};

	//virtual cv::Mat Segment(const cv::Mat&) = 0;
	virtual std::vector<Cluster> Segment(const cv::Mat&) = 0;
};
