#include "stereomatch/BasicBlockMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

BasicBlockMatcher::BasicBlockMatcher() {

}

BasicBlockMatcher::~BasicBlockMatcher() {

}

cv::Mat BasicBlockMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(0.0, 7);
	sbm->compute(rLeft, rRight, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0/16.0);
	//normalize(oResult, oResult, 0.0, 255.0, CV_MINMAX);

	return oResult;
}
