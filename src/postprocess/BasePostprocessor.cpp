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

	bilateralFilter(rImage, oResult, 15, 150.0, 150.0);

	return oResult;
}
