#include "stereomatch/BasicSGMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

BasicSGMatcher::BasicSGMatcher() {

}

BasicSGMatcher::~BasicSGMatcher() {

}

cv::Mat BasicSGMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 7);
	sgbm->compute(rLeft, rRight, oResult);

	return oResult;
}
