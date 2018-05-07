#include "BasePreprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

BasePreprocessor::BasePreprocessor() {


}

BasePreprocessor::~BasePreprocessor() {

}

cv::Mat BasePreprocessor::Preprocess(const cv::Mat& rImage) {
	cv::Mat oResult;

	cvtColor(rImage, oResult, CV_BGR2GRAY);

	return oResult;
}
