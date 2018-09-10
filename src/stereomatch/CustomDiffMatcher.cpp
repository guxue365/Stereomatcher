#include <stereomatch/CustomDiffMatcher.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

#include <stereomatch/CustomBlockMatcher.h>

using namespace std;
using namespace cv;

CustomDiffMatcher::CustomDiffMatcher() :
	miNumDisparities(64),
	miBlockWidth(9),
	miBlockHeight(9),
	mdTolerance(15.0),
	miDiffOrderX(1),
	miDiffOrderY(1),
	miSobelKernelSize(3) {

}

CustomDiffMatcher::~CustomDiffMatcher() {

}

void CustomDiffMatcher::setBlockWidth(int iBlockWidth) {
	assert(iBlockWidth>0);
	assert(iBlockWidth%2==1);

	miBlockWidth = iBlockWidth;
}

void CustomDiffMatcher::setBlockHeight(int iBlockHeight) {
	assert(iBlockHeight>0);
	assert(iBlockHeight%2==1);

	miBlockHeight = iBlockHeight;
}

void CustomDiffMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

void CustomDiffMatcher::setValidTolerance(double dTolerance) {
	assert(dTolerance>0.0);

	mdTolerance = dTolerance;
}

void CustomDiffMatcher::setDiffOrderX(int iDiffOrderX) {
	assert(iDiffOrderX>0);

	miDiffOrderX = iDiffOrderX;
}

void CustomDiffMatcher::setDiffOrderY(int iDiffOrderY) {
	assert(iDiffOrderY>0);

	miDiffOrderY = iDiffOrderY;
}

void CustomDiffMatcher::setSobelKernelSize(int iSobelKernelSize) {
	assert(iSobelKernelSize==1 || iSobelKernelSize==3 || iSobelKernelSize==5 || iSobelKernelSize==7);

	miSobelKernelSize = iSobelKernelSize;
}

cv::Mat CustomDiffMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;

	cv::Mat oDiffLeft, oDiffRight;
	Sobel(rLeft, oDiffLeft, CV_8U, miDiffOrderX, miDiffOrderY, miSobelKernelSize);
	Sobel(rRight, oDiffRight, CV_8U, miDiffOrderX, miDiffOrderY, miSobelKernelSize);

	/*cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(oDiffLeft, oDiffRight, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);*/

	CustomBlockMatcher oBlockMatcher;
	oBlockMatcher.setBlockWidth(miBlockWidth);
	oBlockMatcher.setBlockHeight(miBlockHeight);
	oBlockMatcher.setNumDisparities(miNumDisparities);
	oBlockMatcher.setValidTolerance(mdTolerance);

	oResult = oBlockMatcher.Match(oDiffLeft, oDiffRight);

	return oResult;
}
