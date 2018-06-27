#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv/cv.hpp>

using namespace std;
using namespace cv;

cv::Mat CreateImageFromLabels(const cv::Mat& rImage, const std::vector<cv::Point3d>& aPoints, const std::vector<unsigned int>& aLabels);
double norm3(const cv::Point3d& p1, const cv::Point3d& p2);
std::vector<cv::Point3d> ExtractPoints(const cv::Mat& rImage);

cv::Mat Segment(const cv::Mat& rImage);
std::vector<std::vector<unsigned int> > ComputeNeighbors(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double eps);
std::vector<unsigned int> rundbscan(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps, unsigned int iMinPoints);

int main() {
	Mat oOriginal = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/region_example2.png", CV_LOAD_IMAGE_GRAYSCALE);
	threshold(oOriginal, oOriginal, 128, 255, THRESH_BINARY);

	imshow("Original", oOriginal);

	if(oOriginal.depth()!=CV_8U || oOriginal.channels()!=1) {
		cout<<"Error: Matrix has wrong depth"<<endl;
		return -1;
	}

	Mat oSegmentation = Segment(oOriginal);
	normalize(oSegmentation, oSegmentation, 255.0, 0.0, CV_MINMAX);
	cvtColor(oSegmentation, oSegmentation, CV_GRAY2BGR);
	applyColorMap(oSegmentation, oSegmentation, COLORMAP_JET);

	imshow("Segmentation", oSegmentation);

	imwrite("/home/jung/2018EntwicklungStereoalgorithmus/data/segmentation_result.png", oSegmentation);


	waitKey(0);

	return 0;
}


cv::Mat Segment(const cv::Mat& rImage) {
	cv::Mat oResult;

	auto aPoints = ExtractPoints(rImage);
	cout<<"DBSCAN: Extracted "<<aPoints.size()<<" Points"<<endl;

	auto aLabels = rundbscan(aPoints, norm3, 20.0, 10);
	oResult = CreateImageFromLabels(rImage, aPoints, aLabels);

	return oResult;
}

std::vector<cv::Point3d> ExtractPoints(const cv::Mat& rImage) {
	assert(rImage.type()==CV_8U);

	vector<cv::Point3d> aResult;

	for(int i=0; i<rImage.rows; ++i) {
		for(int j=0; j<rImage.cols; ++j) {
			uchar val = rImage.at<uchar>(i, j);
			if(val>250) {
				aResult.push_back(cv::Point3d((double)i, (double)j, (double)val));
			}
		}
	}

	return aResult;
}

std::vector<unsigned int> rundbscan(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double dEps, unsigned int iMinPoints) {
	unsigned int iNumPoints = aPoints.size();

	vector<vector<unsigned int> > aComputedNeighbors = ComputeNeighbors(aPoints, fNorm, dEps);

	vector<unsigned int> aLabel(iNumPoints);
	unsigned int iCurrentLabel = 3;
	for(size_t i=0; i<iNumPoints; ++i) {
		aLabel[i] = 0;
	}

	for(size_t i=0; i<iNumPoints; ++i) {
		if(aLabel[i]>0) 	continue; // schon besucht

		//vector<unsigned int> aNeighborPtr = FindNeighbors(i, aPoints, fNorm, dEps);
		vector<unsigned int> aNeighborPtr = aComputedNeighbors[i];
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
			//vector<unsigned int> aNextNeightborPointer = FindNeighbors(iNeighborPtr, aPoints, fNorm, dEps);
			vector<unsigned int> aNextNeightborPointer = aComputedNeighbors[iNeighborPtr];
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

std::vector<std::vector<unsigned int> > ComputeNeighbors(const std::vector<cv::Point3d>& aPoints, std::function<double(const cv::Point3d&, const cv::Point3d&)> fNorm, double eps) {
	std::vector<std::vector<unsigned int> > aResult(aPoints.size());

	for(size_t i=0; i<aPoints.size(); ++i) {
		vector<unsigned int> aNeighbors;
		for(size_t j=0; j<aPoints.size(); ++j) {
			if(fNorm(aPoints[i], aPoints[j])<eps) {
				aNeighbors.push_back(j);
			}
		}
		aResult[i] = aNeighbors;
	}

	return aResult;
}

cv::Mat CreateImageFromLabels(const cv::Mat& rImage, const std::vector<cv::Point3d>& aPoints, const std::vector<unsigned int>& aLabels) {
	assert(aPoints.size()==aLabels.size());

	cv::Mat oResult = cv::Mat::zeros(rImage.rows, rImage.cols, CV_8U);

	cout<<"Writing "<<aPoints.size()<<" | "<<aLabels.size()<<" Points"<<endl;
	for(size_t k=0; k<aPoints.size(); ++k) {
		int i = (int)aPoints[k].x;
		int j = (int)aPoints[k].y;
		uchar val = (uchar)aLabels[k];
		oResult.at<uchar>(i, j) = val;
	}

	return oResult;
}

double norm3(const cv::Point3d& p1, const cv::Point3d& p2) {
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y)+(p1.z-p2.z)*(p1.z-p2.z));
}
