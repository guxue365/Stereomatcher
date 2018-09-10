#include <stereomatch/CustomCannyMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <stereomatch/CustomBlockMatcher.h>

using namespace std;
using namespace cv;

CustomCannyMatcher::CustomCannyMatcher() :
	miNumDisparities(64),
	miBlockWidth(9),
	miBlockHeight(9),
	mdTolerance(15.0),
	mdThreshold1(100.0),
	mdThreshold2(200.0) {

}

CustomCannyMatcher::~CustomCannyMatcher() {

}

void CustomCannyMatcher::setBlockWidth(int iBlockWidth) {
	assert(iBlockWidth>0);
	assert(iBlockWidth%2==1);

	miBlockWidth = iBlockWidth;
}

void CustomCannyMatcher::setBlockHeight(int iBlockHeight) {
	assert(iBlockHeight>0);
	assert(iBlockHeight%2==1);

	miBlockHeight = iBlockHeight;
}

void CustomCannyMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

void CustomCannyMatcher::setValidTolerance(double dTolerance) {
	assert(dTolerance>0.0);

	mdTolerance = dTolerance;
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

	/*cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(oLeftCanny, oRightCanny, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);*/

	CustomBlockMatcher oBlockMatcher;
	oBlockMatcher.setBlockWidth(miBlockWidth);
	oBlockMatcher.setBlockHeight(miBlockHeight);
	oBlockMatcher.setNumDisparities(miNumDisparities);
	oBlockMatcher.setValidTolerance(mdTolerance);

	oResult = oBlockMatcher.Match(oLeftCanny, oRightCanny);

	return oResult;
}
