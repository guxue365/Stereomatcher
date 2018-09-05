#include <stereomatch/CustomDiffMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

CustomDiffMatcher::CustomDiffMatcher() :
	miBlockSize(9),
	miNumDisparities(64) {

}

CustomDiffMatcher::~CustomDiffMatcher() {

}

void CustomDiffMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 == 1);

	miBlockSize = iBlockSize;
}

void CustomDiffMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat CustomDiffMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Mat oDiffLeft, oDiffRight;
	Sobel(rLeft, oDiffLeft, CV_8U, 1, 1);
	Sobel(rRight, oDiffRight, CV_8U, 1, 1);

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(oDiffLeft, oDiffRight, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);

	return oResult;
}
