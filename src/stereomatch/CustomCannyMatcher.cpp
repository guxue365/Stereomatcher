#include <stereomatch/CustomCannyMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomCannyMatcher::CustomCannyMatcher() :
	miBlockSize(9),
	miNumDisparities(64),
	mdThreshold1(100.0),
	mdThreshold2(200.0) {

}

CustomCannyMatcher::~CustomCannyMatcher() {

}

void CustomCannyMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 == 1);

	miBlockSize = iBlockSize;
}

void CustomCannyMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

void CustomCannyMatcher::setThreshold1(double dThreshold1) {
	mdThreshold1 = dThreshold1;
}

void CustomCannyMatcher::setThreshold2(double dThreshold2) {
	mdThreshold2 = dThreshold2;
}

cv::Mat CustomCannyMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;
	cv::Mat oLeftCanny, oRightCanny;

	Canny(rLeft, oLeftCanny, mdThreshold1, mdThreshold2);
	Canny(rRight, oRightCanny, mdThreshold1, mdThreshold2);

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(oLeftCanny, oRightCanny, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);

	return oResult;
}
