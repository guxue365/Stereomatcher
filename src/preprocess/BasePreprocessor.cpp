#include "preprocess/BasePreprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;

BasePreprocessor::BasePreprocessor() {


}

BasePreprocessor::~BasePreprocessor() {

}

cv::Mat BasePreprocessor::Preprocess(const cv::Mat& rImage, int iSide) {
	assert(rImage.type()==CV_8UC3);

	Mat oResult(rImage);

	cvtColor(oResult, oResult, COLOR_RGB2GRAY, 1);

	return oResult;
}
