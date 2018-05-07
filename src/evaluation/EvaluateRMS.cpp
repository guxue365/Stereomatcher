#include "evaluation/EvaluateRMS.h"

#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;


EvaluateRMS::EvaluateRMS() {

}

EvaluateRMS::~EvaluateRMS() {

}

double EvaluateRMS::Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage) {
	assert(rGroundTruth.data!=NULL);
	assert(rDisparityImage.data!=NULL);
	assert(rGroundTruth.rows==rDisparityImage.rows);
	assert(rGroundTruth.cols==rDisparityImage.cols);
	assert(rGroundTruth.depth()==CV_16U);
	assert(rDisparityImage.depth()==CV_16U);
	assert(rGroundTruth.channels()==1);
	assert(rDisparityImage.channels()==1);

	double dError = 0.0;

	moErrorMap = cv::Mat(rGroundTruth.rows, rGroundTruth.cols, CV_32F);

	for(int i=0; i<rGroundTruth.rows; ++i) {
		for(int j=0; j<rGroundTruth.cols; ++j) {
			double gt = (double)(rGroundTruth.at<ushort>(i, j));
			double di = (double)(rDisparityImage.at<ushort>(i, j));
			double err = (gt-di)*(gt-di);
			moErrorMap.at<float>(i, j) = (float)(sqrt(err));
			dError+=err;
		}
	}

	dError/=((double)(rGroundTruth.rows)*(double)(rGroundTruth.cols));

	return sqrt(dError);
}

cv::Mat EvaluateRMS::getVisualRepresentation() {
	return moErrorMap;
}
