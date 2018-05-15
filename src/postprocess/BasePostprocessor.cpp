#include "postprocess/BasePostprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

BasePostprocessor::BasePostprocessor() {

}

BasePostprocessor::~BasePostprocessor() {

}

cv::Mat BasePostprocessor::Postprocess(const cv::Mat& rImage) {
	assert(rImage.type()==CV_16S);
	cv::Mat oResult;

	rImage.convertTo(oResult, CV_16U, 256.0);

	cv::blur(oResult, oResult, cv::Size(5, 5));

	return oResult;
}
