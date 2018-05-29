#include <bgsubtraction/CustomFrameDifference.h>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>


CustomFrameDifference::CustomFrameDifference() :
	mpBGSLeft(new FrameDifference()),
	mpBGSRight(new FrameDifference()) {

	mpBGSLeft->setShowOutput(false);
	mpBGSRight->setShowOutput(false);
}

CustomFrameDifference::~CustomFrameDifference() {
	delete mpBGSLeft;
	delete mpBGSRight;
}

cv::Mat CustomFrameDifference::SubtractLeft(const cv::Mat& rImage) {
	assert(rImage.type()==CV_8U);

	cv::Mat oMask, oBGModel, oResult;

	mpBGSLeft->process(rImage, oMask, oBGModel);

	rImage.copyTo(oResult, oMask);

	return oResult;
}

cv::Mat CustomFrameDifference::SubtractRight(const cv::Mat& rImage) {
	assert(rImage.type()==CV_8U);

	cv::Mat oMask, oBGModel, oResult;

	mpBGSRight->process(rImage, oMask, oBGModel);

	rImage.copyTo(oResult, oMask);

	return oResult;
}
