#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <bgslibrary.h>

using namespace std;
using namespace cv;

std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, int iNumRectangles);
void RegionGrowing(const cv::Mat& rInput, cv::Mat& rResult);
std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed);

std::vector<unsigned int> dbscan(const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps, unsigned int iMinPoints, cv::Mat& rOutput);
std::vector<unsigned int> FindNeighbors(unsigned int iPointPtr, const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps);
std::vector<cv::Point2d> ExtractPointsFromMat(const cv::Mat& rMat);
double norm2(const cv::Point2d&, const cv::Point2d&);

void TestRegionGrowing();

int main() {
	cv::VideoCapture oImageStream("/home/jung/2018EntwicklungStereoalgorithmus/data/changedetection/dataset/baseline/highway/input/in%06d.jpg");
	//cv::VideoCapture oImageStream("E:/dataset_changedetection/dataset2014/dataset/baseline/highway/input/in%06d.jpg");
	if(!oImageStream.isOpened()) {
		cout<<"Error opening files"<<endl;
		return -1;
	}

	IBGS* bgs1 = new DPWrenGA();
	IBGS* bgs2 = new PixelBasedAdaptiveSegmenter();
	IBGS* bgs3 = new FrameDifference();

	bgs1->setShowOutput(false);
	bgs2->setShowOutput(false);
	bgs3->setShowOutput(false);

	try {
		Mat frame;
		Mat img_mask1, img_bkgmodel1;
		Mat img_mask2, img_bkgmodel2;
		Mat img_mask3, img_bkgmodel3;
		Mat dpwren_2;
		Mat oRegion;

		for(int iFrame=0;;++iFrame) {
			cout<<"Frame: "<<iFrame<<endl;

			oImageStream>>frame;
			if(frame.empty()) break;

			bgs1->process(frame, img_mask1, img_bkgmodel1);
			bgs2->process(frame, img_mask2, img_bkgmodel2);
			bgs3->process(frame, img_mask3, img_bkgmodel3);


			auto kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
			auto kernel2 = getStructuringElement(MORPH_CROSS, Size(5, 5));
			morphologyEx(img_mask2, dpwren_2, MORPH_OPEN, kernel, Size(-1, -1), 2);
			morphologyEx(dpwren_2, dpwren_2, MORPH_DILATE, kernel2, Size(-1, -1), 3);

			RegionGrowing(dpwren_2, oRegion);
			/*oRegion = cv::Mat::zeros(frame.rows, frame.cols, CV_8U);
			auto aPoints = ExtractPointsFromMat(dpwren_2);
			cout<<"Extracted "<<aPoints.size()<<" Points"<<endl;
			dbscan(aPoints, norm2, 5.0, 10, oRegion);*/
			normalize(oRegion, oRegion, 255, 0, CV_MINMAX);
			Mat oRegionColor;
			applyColorMap(oRegion, oRegionColor, COLORMAP_JET);

			std::vector<cv::Rect> aBounds = FindRectangles(dpwren_2, 4);
			cvtColor(dpwren_2, dpwren_2, COLOR_GRAY2BGR, 3);
			for(auto& rBound: aBounds) {
				rectangle(dpwren_2, rBound, cv::Scalar(255.0, 0.0, 255.0));
			}

			imshow("Original", frame);
			imshow("DPWrenGA", img_mask1);
			imshow("PBAS", img_mask2);
			imshow("FD", img_mask3);
			imshow("dpwren2 filter", dpwren_2);
			imshow("region", oRegionColor);

			Mat row1;
			hconcat(frame, img_mask1, row1);
			imshow("row1", row1);

			char c = (char)waitKey(50);
			if(c==27) 	break;
		}
	} catch(std::exception& e) {
		cout<<"Exception: "<<e.what()<<endl;
		return -2;
	}

	return 0;
}

std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, int iNumRectangles) {
	vector<Rect> aResult(iNumRectangles);

	std::vector<int> aBounds(iNumRectangles);
	int iRectSize = rInput.rows/iNumRectangles;
	for(int i=0; i<iNumRectangles; ++i) {
		aBounds[i] = i*iRectSize;
	}
	aBounds[iNumRectangles-1] = rInput.rows-iRectSize;

	for(size_t i=0; i<iNumRectangles; ++i) {
		cv::Rect oTarget(0, aBounds[i], rInput.cols, iRectSize);

		aResult[i] = boundingRect(rInput(oTarget));
		aResult[i].y+=aBounds[i];
	}

	return aResult;
}

void RegionGrowing(const cv::Mat& rInput, cv::Mat& rResult) {
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

std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed) {
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

void TestRegionGrowing() {
	Mat oImage = imread("../data/region_example.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (oImage.empty()) {
		cout << "Error loading file" << endl;
		return;
	}

	Mat oRegion;
	RegionGrowing(oImage, oRegion);
	normalize(oRegion, oRegion, 255, 0, CV_MINMAX);
	Mat oRegionColor;
	applyColorMap(oRegion, oRegionColor, COLORMAP_JET);



	resize(oImage, oImage, Size(800, 800), 0.0, 0.0, INTER_LANCZOS4);
	resize(oRegionColor, oRegionColor, Size(800, 800), 0.0, 0.0, INTER_LANCZOS4);
	imshow("image", oImage);
	imshow("region", oRegionColor);

	for (;;) {
		char c = (char)waitKey(25);
		if (c == 27) 	break;
	}
}

std::vector<unsigned int> dbscan(const std::vector<cv::Point2d>& aPoints, std::function<double(const cv::Point2d&, const cv::Point2d&)> fNorm, double dEps, unsigned int iMinPoints, cv::Mat& rOutput) {
	assert(rOutput.type()==CV_8U);

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

	for(size_t i=0; i<iNumPoints; ++i) {
		rOutput.at<uchar>(aPoints[i].y, aPoints[i].x) = (uchar)aLabel[i];
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

std::vector<cv::Point2d> ExtractPointsFromMat(const cv::Mat& rMat) {
	assert(rMat.type()==CV_8U);
	vector<cv::Point2d> oResult;

	for(int i=0; i<rMat.rows; ++i) {
		for(int j=0; j<rMat.cols; ++j) {
			if(rMat.at<uchar>(i, j)>0) {
				oResult.push_back(cv::Point2d((double)j, (double)i));
			}
		}
	}

	return oResult;
}
