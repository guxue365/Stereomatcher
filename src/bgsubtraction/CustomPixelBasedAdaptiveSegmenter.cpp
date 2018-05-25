#include "bgsubtraction/CustomPixelBasedAdaptiveSegmenter.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>




CustomPixelBasedAdaptiveSegmenter::CustomPixelBasedAdaptiveSegmenter() :
	mpBGSLeft(new PixelBasedAdaptiveSegmenter()),
	mpBGSRight(new PixelBasedAdaptiveSegmenter()) {

	mpBGSLeft->setShowOutput(false);
	mpBGSRight->setShowOutput(false);
}

CustomPixelBasedAdaptiveSegmenter::~CustomPixelBasedAdaptiveSegmenter() {
	delete mpBGSLeft;
	delete mpBGSRight;
}

cv::Mat CustomPixelBasedAdaptiveSegmenter::SubtractLeft(const cv::Mat& rImage) {
	cv::Mat oResult, oBGModel;

	mpBGSLeft->process(rImage, oResult, oBGModel);

	return oResult;
}

cv::Mat CustomPixelBasedAdaptiveSegmenter::SubtractRight(const cv::Mat& rImage) {
	cv::Mat oResult, oBGModel;

	mpBGSRight->process(rImage, oResult, oBGModel);

	return oResult;
}
