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
	threshold(oResult, oResult, 30.0, 255.0, THRESH_BINARY);

	auto kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	morphologyEx(oResult, oResult, MORPH_OPEN, kernel, Size(-1, -1), 2);

	return oResult;
}
