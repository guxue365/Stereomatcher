#include "stereomatch/BasicSGMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

BasicSGMatcher::BasicSGMatcher() :
	miBlockSize(9),
	miNumDisparities(64) {

}

BasicSGMatcher::~BasicSGMatcher() {

}

void BasicSGMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 = 1);

	miBlockSize = iBlockSize;
}

void BasicSGMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat BasicSGMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, miNumDisparities, miBlockSize);
	sgbm->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0);
	//normalize(oResult, oResult, 0.0, 255.0, CV_MINMAX);

	return oResult;
}
