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
	cv::Mat oMask, oBGModel, oResult;

	mpBGSLeft->process(rImage, oMask, oBGModel);
	rImage.copyTo(oResult, oMask);

	//auto aROIs = FindRectangles(oMask, 4);
	//FillMatrixWithRect(rImage, oResult, aROIs);

	return oResult;
}

cv::Mat CustomPixelBasedAdaptiveSegmenter::SubtractRight(const cv::Mat& rImage) {
	cv::Mat oMask, oBGModel, oResult;

	mpBGSRight->process(rImage, oMask, oBGModel);
	rImage.copyTo(oResult, oMask);

	//auto aROIs = FindRectangles(oMask, 4);
	//FillMatrixWithRect(rImage, oResult, aROIs);

	return oResult;
}

std::vector<cv::Rect> CustomPixelBasedAdaptiveSegmenter::FindRectangles(const cv::Mat& rInput, int iNumRectangles) {
	vector<Rect> aResult(iNumRectangles);

	std::vector<int> aBounds(iNumRectangles);
	int iRectSize = rInput.rows/iNumRectangles+1;
	for(int i=0; i<iNumRectangles; ++i) {
		aBounds[i] = i*iRectSize;
	}
	aBounds[iNumRectangles-1] = rInput.rows-iRectSize;

	for(size_t i=0; i<iNumRectangles; ++i) {
		cv::Rect oTarget(0, aBounds[i], rInput.cols, iRectSize);

		aResult[i] = boundingRect(rInput(oTarget));
		aResult[i].y+=aBounds[i];
	}

	return aResult;
}

void CustomPixelBasedAdaptiveSegmenter::FillMatrixWithRect(const cv::Mat& rOriginal, cv::Mat& rOutput, const std::vector<cv::Rect>& aROIs) {
	assert(rOriginal.type()==CV_8U);

	rOutput = Mat::zeros(rOriginal.rows, rOriginal.cols, CV_8U);

	for(size_t k=0; k<aROIs.size(); ++k) {

		auto& rRect = aROIs[k];
		for(size_t i=rRect.y; i<rRect.y+rRect.height; ++i) {
			for(size_t j=rRect.x; j<rRect.x+rRect.width; ++j) {
				rOutput.at<uchar>(i, j) = rOriginal.at<uchar>(i, j);
			}
		}
	}
}
