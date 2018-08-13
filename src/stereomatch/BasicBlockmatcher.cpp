#include "stereomatch/BasicBlockMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

BasicBlockMatcher::BasicBlockMatcher() : 
	miBlockSize(9),
	miNumDisparities(64) {

}

BasicBlockMatcher::~BasicBlockMatcher() {

}

void BasicBlockMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 = 1);

	miBlockSize = iBlockSize;
}

void BasicBlockMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat BasicBlockMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(rLeft, rRight, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0/16.0);

	return oResult;
}
