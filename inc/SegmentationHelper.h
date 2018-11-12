#pragma once

#include <vector>
#include <iostream>

#include <opencv2/core.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/features/moment_of_inertia_estimation.h>

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, std::vector<double>& rDimension, std::vector<double>& rEccentricity,
		pcl::PointXYZ& rPosition, pcl::PointXYZ& rOBBPosition, pcl::PointXYZ& rOBBMin, pcl::PointXYZ& rOBBMax, Eigen::Matrix3f& rOBBRot);

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity);
