#include "segmentation/TwoStepSegmentation.h"

#include <iostream>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include <SegmentationHelper.h>

using namespace std;
using namespace cv;


TwoStepSegmentation::TwoStepSegmentation() {


}

TwoStepSegmentation::~TwoStepSegmentation() {

}

std::vector<Cluster> TwoStepSegmentation::Segment(const cv::Mat& rImage) {
	std::vector<Cluster> aResult;

	Mat oImageCoarse;

	double dScaling = 0.025;
	resize(rImage, oImageCoarse, Size(), dScaling, dScaling, INTER_LINEAR);

	uchar cNumLabel;
	Mat oRegionCoarse = GrowRegion(oImageCoarse, cNumLabel);
	int iNumLabel = (int)cNumLabel;

	vector<Rect2i> aRegionsRaw = ExtractRegions(oRegionCoarse, iNumLabel);

	vector<Rect2i> aRegions = MergeRegions(aRegionsRaw);

	cout<<"Found "<<aRegions.size()<<" Regions"<<endl;

	for (size_t iRegion = 0; iRegion < aRegions.size(); ++iRegion) {
		aRegions[iRegion].x = (int)((double)(aRegions[iRegion].x-1) / (dScaling));
		aRegions[iRegion].y = (int)((double)(aRegions[iRegion].y-1) / (dScaling));
		aRegions[iRegion].width = (int)((double)(aRegions[iRegion].width+2) / (dScaling));
		aRegions[iRegion].height = (int)((double)(aRegions[iRegion].height+2) / (dScaling));
		if (aRegions[iRegion].x < 0)		aRegions[iRegion].x = 0;
		if (aRegions[iRegion].y < 0)		aRegions[iRegion].y = 0;
		if (aRegions[iRegion].width + aRegions[iRegion].x >= rImage.cols)	aRegions[iRegion].width = rImage.cols - aRegions[iRegion].x - 1;
		if (aRegions[iRegion].height + aRegions[iRegion].y > rImage.rows)	aRegions[iRegion].height = rImage.rows - aRegions[iRegion].y - 1;

		cv::Mat oDummyMat = cv::Mat::zeros(rImage.rows, rImage.cols, CV_8U);

		for(int i=aRegions[iRegion].y; i<aRegions[iRegion].y+aRegions[iRegion].height; ++i) {
			for(int j=aRegions[iRegion].x; j<aRegions[iRegion].x+aRegions[iRegion].width; ++j) {
				oDummyMat.at<uchar>(i, j) = rImage.at<uchar>(i, j);
			}
		}

		auto aTmpCluster = PCLSegmentation::Segment(oDummyMat);
		aResult.insert(aResult.end(), aTmpCluster.begin(), aTmpCluster.end());
	}

	return aResult;
}

std::vector<cv::Rect2i> TwoStepSegmentation::ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions) {
	vector<cv::Rect2i> aResult(iNumRegions);

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i] = Rect2i(-1, -1, -1, -1);
	}

	for (int i = 0; i < rCoarseRegions.rows; ++i) {
		for (int j = 0; j < rCoarseRegions.cols; ++j) {
			if (rCoarseRegions.at<uchar>(i, j) > 0) {
				int iRegionIndex = (int)(rCoarseRegions.at<uchar>(i, j))-1;

				Rect2i& rRegion = aResult[iRegionIndex];
				if (rRegion.x == -1) {
					rRegion.x = j;
					rRegion.width = j;
					rRegion.y = i;
					rRegion.height = i;
				}
				else {
					if (j < rRegion.x) {
						rRegion.x = j;
					}
					if (j > rRegion.width) {
						rRegion.width = j;
					}
					if (i < rRegion.y) {
						rRegion.y = i;
					}
					if (i > rRegion.height) {
						rRegion.height = i;
					}
				}
			}
		}
	}

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i].width = aResult[i].width - aResult[i].x+1;
		aResult[i].height = aResult[i].height - aResult[i].y+1;
	}

	return aResult;
}

std::vector<cv::Rect2i> TwoStepSegmentation::MergeRegions(const std::vector<cv::Rect2i>& aRegions) {
	vector<cv::Rect2i> aRegionCopy(aRegions);
	vector<cv::Rect2i> aResult;

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		for (size_t j = i + 1; j < aRegionCopy.size(); ++j) {
			if ((aRegionCopy[i] & aRegionCopy[j]).area() > 0) {
				aRegionCopy[i] = (aRegionCopy[i] | aRegionCopy[j]);
				aRegionCopy[j] = cv::Rect2i();
			}
		}
	}

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		if (aRegionCopy[i].area() > 0) {
			aResult.push_back(aRegionCopy[i]);
		}
	}

	return aResult;
}
