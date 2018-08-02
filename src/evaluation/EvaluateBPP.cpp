#include "evaluation/EvaluateBPP.h"

#include <iostream>

using namespace std;

EvaluateBPP::EvaluateBPP(double dTolerance) : mdTolerance(dTolerance) {

}

EvaluateBPP::~EvaluateBPP() {

}

double EvaluateBPP::Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage) {
	assert(rGroundTruth.rows==rDisparityImage.rows);
	assert(rGroundTruth.cols==rDisparityImage.cols);
	assert(rGroundTruth.type()==CV_8U);
	assert(rDisparityImage.type()==CV_8U);

	double dError = 0.0;
	double dCount = 0.0;

	moErrorMap = cv::Mat::zeros(rGroundTruth.size(), CV_8U);

	for(int i=0; i<rGroundTruth.rows; ++i) {
		for(int j=0; j<rGroundTruth.cols; ++j) {
			double gt = (double)(rGroundTruth.at<uchar>(i, j));
			double di = (double)(rDisparityImage.at<uchar>(i, j));

			if (gt <= 0.0)	continue;  // only validate valid gt-data
			dCount += 1.0;
			if(abs(gt-di)>=mdTolerance) {
				dError+=1.0;
				moErrorMap.at<uchar>(i, j) = 255;
			}
		}
	}
	dError /= dCount;

	return dError;
}

cv::Mat EvaluateBPP::getVisualRepresentation() {
	return moErrorMap;
}
