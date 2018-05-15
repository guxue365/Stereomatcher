#include "evaluation/EvaluateBPP.h"

#include <iostream>

using namespace std;

EvaluateBPP::EvaluateBPP(double dTolerance) : mdTolerance(dTolerance) {

}

EvaluateBPP::~EvaluateBPP() {

}

double EvaluateBPP::Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage) {
	assert(rGroundTruth.data!=NULL);
	assert(rDisparityImage.data!=NULL);
	assert(rGroundTruth.rows==rDisparityImage.rows);
	assert(rGroundTruth.cols==rDisparityImage.cols);
	assert(rGroundTruth.depth()==CV_16U);
	assert(rDisparityImage.depth()==CV_16U);
	assert(rGroundTruth.channels()==1);
	assert(rDisparityImage.channels()==1);

	double dError = 0.0;

	moErrorMap = cv::Mat(rGroundTruth.rows, rGroundTruth.cols, CV_32F, 0.0f);

	for(int i=0; i<rGroundTruth.rows; ++i) {
		for(int j=0; j<rGroundTruth.cols; ++j) {
			double gt = (double)(rGroundTruth.at<ushort>(i, j));
			double di = (double)(rDisparityImage.at<ushort>(i, j));

			if(abs(gt-di)>=mdTolerance) {
				dError+=1.0;
				moErrorMap.at<float>(i, j) = 255.0f;
			}
		}
	}
	dError/=((double)(rGroundTruth.rows)*(double)(rGroundTruth.cols));

	return dError;
}

cv::Mat EvaluateBPP::getVisualRepresentation() {
	return moErrorMap;
}
