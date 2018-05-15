#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <bgslibrary.h>

using namespace std;
using namespace cv;

std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, size_t iNumRectangles);

int main() {
	cv::VideoCapture oImageStream("/home/jung/2018EntwicklungStereoalgorithmus/data/changedetection/dataset/baseline/highway/input/in%06d.jpg");
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

std::vector<cv::Rect> FindRectangles(const cv::Mat& rInput, size_t iNumRectangles) {
	vector<Rect> aResult(iNumRectangles);

	std::vector<int> aBounds(iNumRectangles);
	int iRectSize = rInput.rows/iNumRectangles;
	for(size_t i=0; i<iNumRectangles; ++i) {
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
