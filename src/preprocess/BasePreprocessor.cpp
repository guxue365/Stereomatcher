#include "preprocess/BasePreprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

BasePreprocessor::BasePreprocessor() {


}

BasePreprocessor::~BasePreprocessor() {

}

cv::Mat BasePreprocessor::Preprocess(const cv::Mat& rImage) {
	return rImage;
}
