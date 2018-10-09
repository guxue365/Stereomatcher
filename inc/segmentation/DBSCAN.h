#pragma once

#include <vector>
#include <functional>

#include "ISegmentation.h"

class DBSCAN : public ISegmentation {
public:
	DBSCAN();
	virtual ~DBSCAN();

	void setEps(double dEps);
	void setMinPts(unsigned int iMinPoints);

	//cv::Mat Segment(const cv::Mat& rImage);
	std::vector<Cluster> Segment(const cv::Mat& rImage);
private:
	double mdEps;
	unsigned int miMinPoints;

	std::vector<cv::Point3d> ExtractPoints(const cv::Mat& rImage);
	std::vector<unsigned int> rundbscan(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps, unsigned int iMinPoints);
	std::vector<unsigned int> FindNeighbors(unsigned int iPointPtr, const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps);
	cv::Mat CreateImageFromLabels(const cv::Mat& rImage, const std::vector<cv::Point3d>& aPoints, const std::vector<unsigned int>& aLabels);
};
