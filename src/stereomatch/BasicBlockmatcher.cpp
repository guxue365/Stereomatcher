#include "stereomatch/BasicBlockmatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

BasicBlockmatcher::BasicBlockmatcher() {

}

BasicBlockmatcher::~BasicBlockmatcher() {

}

cv::Mat BasicBlockmatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult(rLeft.rows, rLeft.cols, CV_32F);

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(0, 21);
	//sbm->setNumDisparities(0);
	sbm->setMinDisparity(0);
	sbm->compute(rLeft, rRight, oResult);

	return oResult;
}
