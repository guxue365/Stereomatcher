/*
 * TestStereoKitty.cpp
 *
 *  Created on: 02.05.2018
 *      Author: jung
 */




#include <iostream>
#include <chrono>
#include <fstream>
#include <string>
#include <sys/stat.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include "EvaluateBPP.h"
#include "EvaluateRMS.h"

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


	Mat ImageLeft = imread("left.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat ImageRight = imread("right.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat ImageGT = imread("gt.png", CV_LOAD_IMAGE_GRAYSCALE | CV_LOAD_IMAGE_ANYDEPTH);
	ImageGT/=256;

	Mat ImageStereo;

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(0, 9);
	sbm->compute(ImageLeft, ImageRight, ImageStereo);

	Rect cut(258, 258, 10, 10);
	cout<<ImageStereo(cut)<<endl;

	Mat Result;
	ImageStereo.convertTo(Result, CV_16U, 1.0/16.0);

	cout<<"---------------------------------------"<<endl;
	cout<<Result(cut)<<endl;

	cout<<"---------------------------------------"<<endl;
	cout<<ImageGT(cut)<<endl;

	EvaluateBPP eval(10);
	EvaluateRMS eval2;
	double err = eval.Evaluate(ImageGT(cut), Result(cut));
	double err2 = eval2.Evaluate(ImageGT(cut), Result(cut));
	cout<<"Err: "<<err<<endl;
	cout<<"Err2: "<<err2<<endl;

	cout<<"---------------------------------------"<<endl;
	cout<<eval2.getVisualRepresentation()<<endl;

	normalize(Result, Result, 0, 255, CV_MINMAX, CV_8U);
	applyColorMap(Result, Result, COLORMAP_JET);
	imshow("result", Result);
	waitKey(0);

	return 0;
}
