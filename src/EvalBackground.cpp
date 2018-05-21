#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <bgslibrary.h>

using namespace std;
using namespace cv;

std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, int iNumRectangles);
cv::Mat RegionGrowing(const cv::Mat& rInput);
std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed);

int main() {
	Mat oImage = imread("region_example.png", CV_LOAD_IMAGE_GRAYSCALE);
	if (oImage.empty()) {
		cout << "Error loading file" << endl;
		return -1;
	}

	Mat oRegion = RegionGrowing(oImage);
	normalize(oRegion, oRegion, 255, 0, CV_MINMAX);
	Mat oRegionColor;
	applyColorMap(oRegion, oRegionColor, COLORMAP_JET);

	

	resize(oImage, oImage, Size(800, 800));
	resize(oRegionColor, oRegionColor, Size(800, 800));
	imshow("image", oImage);
	imshow("region", oRegionColor);

	for (;;) {

		

		char c = (char)waitKey(25);
		if (c == 27) 	break;
	}

	return 0;
	cv::VideoCapture oImageStream("E:/dataset_changedetection/dataset2014/dataset/baseline/highway/input/in%06d.jpg");
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

	while(true) {
		Mat frame;
		Mat img_mask1, img_bkgmodel1;
		Mat img_mask2, img_bkgmodel2;
		Mat img_mask3, img_bkgmodel3;
		Mat dpwren_2;

		oImageStream>>frame;
		if(frame.empty()) break;

		bgs1->process(frame, img_mask1, img_bkgmodel1);
		bgs2->process(frame, img_mask2, img_bkgmodel2);
		bgs3->process(frame, img_mask3, img_bkgmodel3);

		auto kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
		morphologyEx(img_mask1, dpwren_2, MORPH_OPEN, kernel, Size(-1, -1), 2);
		auto aBounds = FindRectangles(dpwren_2, 4);
		cvtColor(dpwren_2, dpwren_2, COLOR_GRAY2BGR, 3);
		for(auto& rBound: aBounds) {
			rectangle(dpwren_2, rBound, cv::Scalar(255.0, 0.0, 255.0));
		}

		imshow("Original", frame);
		imshow("DPWrenGA", img_mask1);
		imshow("PBAS", img_mask2);
		imshow("FD", img_mask3);
		imshow("dpwren2 filter", dpwren_2);




		char c = (char)waitKey(25);
		if(c==27) 	break;
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

cv::Mat RegionGrowing(const cv::Mat& rInput) {
	assert(rInput.type() == CV_8U);
	int m = rInput.rows;
	int n = rInput.cols;

	cv::Mat oResult(m, n, CV_8U, Scalar(0));
	uchar iLabel = 0;

	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			if (rInput.at<uchar>(i, j) > 0 && oResult.at<uchar>(i, j)==0) {
				++iLabel;
				cout << "New Label: " << (int)iLabel << endl;
				std::vector<cv::Point2i> aNeighbors = { cv::Point2i(j, i) };
				for (size_t k = 0; k < aNeighbors.size(); ++k) {
					if (oResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) > 0)	continue;
					oResult.at<uchar>(aNeighbors[k].y, aNeighbors[k].x) = iLabel;
					auto aCurrentNeighbors = getNeighbors(rInput, aNeighbors[k]);
					aNeighbors.insert(aNeighbors.end(), aCurrentNeighbors.begin(), aCurrentNeighbors.end());
				}
			}
		}
	}

	return oResult;
}

std::vector<cv::Point2i> getNeighbors(const cv::Mat& rRegion, cv::Point2i iSeed) {
	std::vector<cv::Point2i> oResult;

	for (int i = -1; i < 2; ++i) {
		for (int j = -1; j < 2; ++j) {
			if (i == iSeed.y && j == iSeed.x)	continue;
			int x = iSeed.x + j;
			int y = iSeed.y + i;
			if (rRegion.at<uchar>(y, x) > 0) {
				oResult.push_back(cv::Point2i(x, y));
			}
		}
	}

	return oResult;
}