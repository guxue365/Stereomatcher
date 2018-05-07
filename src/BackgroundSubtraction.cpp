#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/video.hpp>

using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main() {


	VideoCapture cap("/home/jung/2018EntwicklungStereoalgorithmus/data/bmc/111_png/input/%1d.png");
	VideoCapture capgt("/home/jung/2018EntwicklungStereoalgorithmus/data/bmc/111_png/truth/%1d.png");

	if(!cap.isOpened()) {
		cout<<"Error opening files"<<endl;
		return -1;
	}

	Mat lastframe;
	Mat mask;
	Ptr<BackgroundSubtractor> pMOG2 = createBackgroundSubtractorMOG2(10);
	while(true) {
		Mat frame;

		cap>>frame;
		if(frame.empty()) break;

		Mat bg;
		Mat gray;
		Mat dil;
		Mat nrm;
		bg = cv::abs(frame-lastframe);
		cvtColor(bg, gray, cv::COLOR_BGR2GRAY);
		dilate(gray, dil, Mat());
		normalize(dil, nrm, 0, 255, CV_MINMAX);

		pMOG2->apply(frame, mask);

		Mat gt;
		capgt>>gt;

		int left = 2000;
		int right = 0;
		int top = 2000;
		int bottom = 0;


		unsigned char* data = (unsigned char*)nrm.data;
		for(int i=0; i<nrm.rows; ++i) {
			for(int j=0; j<nrm.cols; ++j) {
				unsigned char val = data[nrm.cols*i+j];
				if(val>1e5) {
					if(j<left) {
						left = j;
					}
					if(j>right) {
						right = j;
					}
				}
			}
			if(i<top) {
				top = i;
			}
			if(i>bottom) {
				bottom = i;
			}
		}
		rectangle(nrm, cvPoint(left, top), cvPoint(right, bottom), Scalar(255, 0, 255), 5);

		imshow("frame", frame);
		imshow("bg", nrm);
		imshow("mask", mask);
		imshow("gt", gt);


		char c = (char)waitKey(25);
		if(c==27) 	break;

		lastframe = frame;
	}

	cap.release();

	return 0;
}
