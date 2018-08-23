#include "preprocess/PreprocessMask.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

PreprocessMask::PreprocessMask() {


}

PreprocessMask::~PreprocessMask() {

}

cv::Mat PreprocessMask::Preprocess(const cv::Mat& rImage, int iSide) {
	assert(rImage.type()==CV_8UC3);

	Mat oResult(rImage);
	Mat oResultWithMask;

	cvtColor(oResult, oResult, COLOR_RGB2GRAY, 1);

	if(iSide==-1) {
		Mat oMaskLeft = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/mask_left.png", IMREAD_GRAYSCALE);
		oResult.copyTo(oResultWithMask, oMaskLeft);
	}
	else if(iSide==1) {
		Mat oMaskRight = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/mask_right.png", IMREAD_GRAYSCALE);
		oResult.copyTo(oResultWithMask, oMaskRight);
	}



	return oResultWithMask;
}
