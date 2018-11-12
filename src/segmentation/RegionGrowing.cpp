#include "segmentation/RegionGrowing.h"

#include <iostream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <SegmentationHelper.h>

using namespace std;
using namespace cv;


RegionGrowing::RegionGrowing() {


}

RegionGrowing::~RegionGrowing() {

}

std::vector<Cluster> RegionGrowing::Segment(const cv::Mat& rImage) {
	std::vector<Cluster> aResult;

	uchar rNumLabel;
	cv::Mat oRegion = GrowRegion(rImage, rNumLabel);

	int iNumLabel = (int)rNumLabel;

	for(int iLabel = 1; iLabel<=iNumLabel; ++iLabel) {

		cv::Mat oDummyMat = cv::Mat::zeros(rImage.rows, rImage.cols, CV_8U);

		for(int i=0; i<oRegion.rows; ++i) {
			for(int j=0; j<oRegion.cols; ++j) {
				if((int)(oRegion.at<uchar>(i, j))==iLabel) {
					oDummyMat.at<uchar>(i, j) = rImage.at<uchar>(i, j);
				}
			}
		}

		auto aPoints = Extract3DPoints(oDummyMat);

		pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
		for (auto& oPoint : aPoints) {
			pCloud->push_back(oPoint);
		}

		vector<double> aDimension;
		vector<double> aEccentricity;
		pcl::PointXYZ oPosition;
		pcl::PointXYZ oOBBPosition;
		pcl::PointXYZ oOBBMin;
		pcl::PointXYZ oOBBMax;
		Eigen::Matrix3f oOBBRot;

		AnalysePointcloud(pCloud, aDimension, aEccentricity, oPosition, oOBBPosition, oOBBMin, oOBBMax, oOBBRot);

		Eigen::Vector3f position (oOBBPosition.x, oOBBPosition.y, oOBBPosition.z);
		Eigen::Quaternionf quat (oOBBRot);

		cv::Vec3d oCVPosition = {oPosition.x, oPosition.y, oPosition.z};
		cv::Vec3d oCVDimension = {aDimension[0], aDimension[1], aDimension[2]};
		cv::Vec2d oCVEccentricity = {aEccentricity[0], aEccentricity[1]};
		aResult.push_back({oCVPosition, oCVDimension, oCVEccentricity});
	}

	return aResult;
}

cv::Mat RegionGrowing::GrowRegion(const cv::Mat& rInput, uchar& rNumLabel) {
	assert(rInput.type() == CV_8U);
	int m = rInput.rows;
	int n = rInput.cols;

	// init result as unlabeled
	Mat oResult = cv::Mat::zeros(m, n, CV_8U);
	uchar iLabel = 0;

	// iterate all matrix entries
	for (int i = 1; i < m-1; ++i) {
		for (int j = 1; j < n-1; ++j) {
			// if a active pixel is detected and not labeled yet ...

			if (rInput.at<uchar>(i, j) > 0 && oResult.at<uchar>(i, j)==0) {
				++iLabel;

				// expand region recursively by iterating through all neighbors
				vector<cv::Point2i> aNeighbors = { cv::Point2i(j, i) };
				for(size_t k=0; k<aNeighbors.size(); ++k) {
					// ignore neighbor is already labeled
					if (oResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) > 0)	continue;
					
					// assign current label
					oResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) = iLabel;

					// find new neighbors
					vector<cv::Point2i> aCurrentNeighbors = getNeighbors(rInput, aNeighbors[k]);

					// add new neighbors to queue
					aNeighbors.insert(aNeighbors.end(), aCurrentNeighbors.begin(), aCurrentNeighbors.end());
				}
			}
		}
	}

	rNumLabel = iLabel;

	return oResult;
}

std::vector<cv::Point2i> RegionGrowing::getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed) {
	std::vector<cv::Point2i> oResult;

	// iterate in a 3x3 field around the seed point
	for (int i = -1; i < 2; ++i) {
		for (int j = -1; j < 2; ++j) {
			// ignore the seed point itself
			if (i == 0 && j == 0)	continue;

			// compute x and y coordinates
			int x = iSeed.x + j;
			int y = iSeed.y + i;
			if(x<0 || y<0 || x>=rRegion.cols || y>=rRegion.rows) 	continue;

			//when the neighbor pixel is active -> add to list
			if (rRegion.at<uchar>(y, x) > 0) {
				oResult.push_back(cv::Point2i(x, y));
			}
		}
	}

	return oResult;
}
