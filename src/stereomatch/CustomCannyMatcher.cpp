#include <stereomatch/CustomCannyMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomCannyMatcher::CustomCannyMatcher() :
	miBlockSize(9),
	miNumDisparities(64) {

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

cv::Mat CustomCannyMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;
	cv::Mat oLeftCanny, oRightCanny;

	Canny(rLeft, oLeftCanny, 100.0, 200.0);
	Canny(rRight, oRightCanny, 100.0, 200.0);

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(oLeftCanny, oRightCanny, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);

	return oResult;
}
