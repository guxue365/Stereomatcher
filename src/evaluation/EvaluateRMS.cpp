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
	assert(rGroundTruth.type()==CV_8U);
	assert(rDisparityImage.type()==CV_8U);

	double dError = 0.0;

	moErrorMap = Mat::zeros(rGroundTruth.size(), CV_32F);

	int iValidCount = 0;
	for(int i=0; i<rGroundTruth.rows; ++i) {
		for(int j=0; j<rGroundTruth.cols; ++j) {
			double gt = (double)(rGroundTruth.at<uchar>(i, j));
			double di = (double)(rDisparityImage.at<uchar>(i, j));

			if (gt <= 0.0)	continue;  // only validate valid gt-data

			iValidCount++;
			double err = (gt-di)*(gt-di);
			moErrorMap.at<float>(i, j) = (float)(sqrt(err));
			dError+=err;
		}
	}

	dError/=((double)(iValidCount)*(double)(iValidCount));

	return sqrt(dError);
}

cv::Mat EvaluateRMS::getVisualRepresentation() {
	return moErrorMap;
}
