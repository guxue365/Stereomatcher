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

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create();
	sgbm->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);
	//normalize(oResult, oResult, 0.0, 255.0, CV_MINMAX);

	return oResult;
}
