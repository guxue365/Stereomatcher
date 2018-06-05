#include "segmentation/RegionGrowing.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

RegionGrowing::RegionGrowing() {


}

RegionGrowing::~RegionGrowing() {

}

cv::Mat RegionGrowing::Segment(const cv::Mat& rImage) {
	cv::Mat oResult;
	GrowRegion(rImage, oResult);
	return oResult;
}

void RegionGrowing::GrowRegion(const cv::Mat& rInput, cv::Mat& rResult) {
	assert(rInput.type() == CV_8U);
	int m = rInput.rows;
	int n = rInput.cols;

	// init result as unlabeled
	rResult = cv::Mat::zeros(m, n, CV_8U);
	uchar iLabel = 1;

	// iterate all matrix entries
	for (int i = 1; i < m-1; ++i) {
		for (int j = 1; j < n-1; ++j) {
			// if a active pixel is detected and not labeled yet ...

			if (rInput.at<uchar>(i, j) > 0 && rResult.at<uchar>(i, j)==0) {
				++iLabel;

				// expand region recursively by iterating through all neighbors
				vector<cv::Point2i> aNeighbors = { cv::Point2i(j, i) };
				for(size_t k=0; k<aNeighbors.size(); ++k) {
					// ignore neighbor is already labeled
					if (rResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) > 0)	continue;

					// assign current label
					rResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) = iLabel;

					// find new neighbors
					vector<cv::Point2i> aCurrentNeighbors = getNeighbors(rInput, aNeighbors[k]);

					// add new neighbors to queue
					aNeighbors.insert(aNeighbors.end(), aCurrentNeighbors.begin(), aCurrentNeighbors.end());
				}
			}
		}
	}
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
