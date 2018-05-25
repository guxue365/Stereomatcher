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
	cv::Mat oResult, oBGModel;

	mpBGSLeft->process(rImage, oResult, oBGModel);

	return oResult;
}

cv::Mat CustomFrameDifference::SubtractRight(const cv::Mat& rImage) {
	cv::Mat oResult, oBGModel;

	mpBGSRight->process(rImage, oResult, oBGModel);

	return oResult;
}
