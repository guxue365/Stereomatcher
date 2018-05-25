#include "postprocess/BasePostprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

BasePostprocessor::BasePostprocessor() {

}

BasePostprocessor::~BasePostprocessor() {

}

cv::Mat BasePostprocessor::Postprocess(const cv::Mat& rImage) {
	cv::Mat oResult;

	cv::blur(rImage, oResult, cv::Size(5, 5));
	cv::threshold(oResult, oResult, 10.0, 255.0, THRESH_BINARY);

	return oResult;
}
