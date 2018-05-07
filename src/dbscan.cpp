#include <iostream>
#include <vector>
#include <functional>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

std::vector<unsigned int> dbscan(const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps, unsigned int iMinPoints);
std::vector<unsigned int> FindNeighbors(unsigned int iPointPtr, const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps);
double norm2(const cv::Point2d&, const cv::Point2d&);

int main() {
	Mat oOriginal = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/clustering_example.png", CV_LOAD_IMAGE_GRAYSCALE);

	imshow("Original", oOriginal);

	if(oOriginal.depth()!=CV_8U || oOriginal.channels()!=1) {
		cout<<"Error: Matrix has wrong depth"<<endl;
		return -1;
	}

	vector<cv::Point2d> aPoints;
	unsigned char* data = oOriginal.data;
	for(int i=0; i<oOriginal.rows; ++i) {
		for(int j=0; j<oOriginal.cols; ++j) {
			if(data[i*oOriginal.cols+j]>128) {
				aPoints.push_back(cv::Point2d((double)i, (double)j));
			}
		}
	}

	cout<<"Found "<<aPoints.size()<<" points"<<endl;

	Mat oResult(oOriginal.rows, oOriginal.cols, CV_8UC3, Scalar(0, 0, 0));

	auto aLabel = dbscan(aPoints, norm2, 10.0, 3);

	uchar* pData = oResult.data;
	for(size_t k=0; k<aPoints.size(); ++k) {
		cout<<aLabel[k]<<endl;
		int i = (int)aPoints[k].x;
		int j = (int)aPoints[k].y;
		switch(aLabel[k]) {
		case 3: {  // noise
			pData[3*(i*oResult.cols+j)+0] = 0;
			pData[3*(i*oResult.cols+j)+1] = 0;
			pData[3*(i*oResult.cols+j)+2] = 255;
			break;
		}
		case 4: {  // cluster 1
			pData[3*(i*oResult.cols+j)+0] = 0;
			pData[3*(i*oResult.cols+j)+1] = 255;
			pData[3*(i*oResult.cols+j)+2] = 0;
			break;
		}
		case 5: {  // cluster 2
			pData[3*(i*oResult.cols+j)+0] = 255;
			pData[3*(i*oResult.cols+j)+1] = 0;
			pData[3*(i*oResult.cols+j)+2] = 0;
			break;
		}
		case 6: {  // cluster 3
			pData[3*(i*oResult.cols+j)+0] = 255;
			pData[3*(i*oResult.cols+j)+1] = 0;
			pData[3*(i*oResult.cols+j)+2] = 255;
			break;
		}
		default: {
			pData[3*(i*oResult.cols+j)+0] = 255;
			pData[3*(i*oResult.cols+j)+1] = 255;
			pData[3*(i*oResult.cols+j)+2] = 255;
			break;
		}
		}
	}

	imshow("Label", oResult);

	waitKey(0);

	return 0;
}

std::vector<unsigned int> dbscan(const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps, unsigned int iMinPoints) {
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

std::vector<unsigned int> FindNeighbors(unsigned int iPointPtr, const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps) {
	vector<unsigned int> aResult;

	for(size_t i=0; i<aPoints.size(); ++i) {
		if(i==iPointPtr) 	continue;
		if(fNorm(aPoints[iPointPtr], aPoints[i])<=dEps) {
			aResult.push_back(i);
		}
	}

	return aResult;
}

double norm2(const cv::Point2d& p1, const cv::Point2d& p2) {
	return sqrt((p1.x-p2.x)*(p1.x-p2.x)+(p1.y-p2.y)*(p1.y-p2.y));
}
