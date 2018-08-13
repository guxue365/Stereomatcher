#include "segmentation/DBSCAN.h"

#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

double norm3(const cv::Point3d& p1, const cv::Point3d& p2) {
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
}

DBSCAN::DBSCAN() : 
	mdEps(5.0),
	miMinPoints(20) {

}

DBSCAN::~DBSCAN() {

}

void DBSCAN::setEps(double dEps) {
	mdEps = dEps;
}

void DBSCAN::setMinPts(unsigned int iMinPoints) {
	miMinPoints = iMinPoints;
}

cv::Mat DBSCAN::Segment(const cv::Mat& rImage) {
	cv::Mat oResult;

	auto aPoints = ExtractPoints(rImage);

	if(aPoints.size()>20000) 	return rImage;

	auto aLabels = rundbscan(aPoints, norm3, mdEps, miMinPoints);
	oResult = CreateImageFromLabels(rImage, aPoints, aLabels);

	return oResult;
}

std::vector<cv::Point3d> DBSCAN::ExtractPoints(const cv::Mat& rImage) {
	assert(rImage.type()==CV_8U);

	vector<cv::Point3d> aResult;

	for(int i=0; i<rImage.rows; ++i) {
		for(int j=0; j<rImage.cols; ++j) {
			uchar val = rImage.at<uchar>(i, j);
			if(val>0) {
				aResult.push_back(cv::Point3d((double)i, (double)j, (double)val));
			}
		}
	}

	return aResult;
}

std::vector<unsigned int> DBSCAN::rundbscan(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps, unsigned int iMinPoints) {
	unsigned int iNumPoints = aPoints.size();

	vector<unsigned int> aLabel(iNumPoints);
	unsigned int iCurrentLabel = 3;
	for(size_t i=0; i<iNumPoints; ++i) {
		aLabel[i] = 0;
	}

	for(size_t i=0; i<iNumPoints; ++i) {
		if(aLabel[i]>0) 	continue; // schon besucht

		vector<unsigned int> aNeighborPtr = FindNeighbors(i, aPoints, fNorm, dEps);
		if(aNeighborPtr.size()<iMinPoints) {
			aLabel[i] = 3; // label as noise
			continue;
		}
		iCurrentLabel++;
		aLabel[i] = iCurrentLabel;
		for(size_t j=0; j<aNeighborPtr.size(); ++j) {
			size_t iNeighborPtr = aNeighborPtr[j];
			if(aLabel[iNeighborPtr]==3) 	aLabel[iNeighborPtr] = 2; // label as border point
			if(aLabel[iNeighborPtr]>3) 	continue;
			aLabel[iNeighborPtr] = iCurrentLabel; // label as current cluster
			vector<unsigned int> aNextNeightborPointer = FindNeighbors(iNeighborPtr, aPoints, fNorm, dEps);
			if(aNextNeightborPointer.size()>=iMinPoints) {
				for(size_t k=0; k<aNextNeightborPointer.size(); ++k) {
					if(find(aNeighborPtr.begin(), aNeighborPtr.end(), aNextNeightborPointer[k])==aNeighborPtr.end()) {
						aNeighborPtr.push_back(aNextNeightborPointer[k]);
					}
				}
			}
		}
	}
	return aLabel;
}

std::vector<unsigned int> DBSCAN::FindNeighbors(unsigned int iPointPtr, const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps) {
	vector<unsigned int> aResult;

	for(size_t i=0; i<aPoints.size(); ++i) {
		if(i==iPointPtr) 	continue;
		if(fNorm(aPoints[iPointPtr], aPoints[i])<=dEps) {
			aResult.push_back(i);
		}
	}

	return aResult;
}

cv::Mat DBSCAN::CreateImageFromLabels(const cv::Mat& rImage, const std::vector<cv::Point3d>& aPoints, const std::vector<unsigned int>& aLabels) {
	cv::Mat oResult = cv::Mat::zeros(rImage.rows, rImage.cols, CV_8U);

	for(size_t k=0; k<aPoints.size(); ++k) {
		int i = (int)aPoints[k].x;
		int j = (int)aPoints[k].y;
		uchar val = (uchar)aLabels[k];
		oResult.at<uchar>(i, j) = val;
	}

	return oResult;
}
