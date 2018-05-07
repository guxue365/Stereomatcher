#include "postprocess/BasePostprocessor.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

BasePostprocessor::BasePostprocessor() {

}

BasePostprocessor::~BasePostprocessor() {

}

cv::Mat BasePostprocessor::Postprocess(const cv::Mat& rImage) {
	assert(rImage.type()==CV_16S);
	cv::Mat oResult;

	short* data = (short*)rImage.data;
	for(int i=0; i<rImage.rows*rImage.cols; ++i) {
		if(data[i]<0) 	data[i] = 0;
	}

	rImage.convertTo(oResult, CV_16U, 256.0/16.0);

	return oResult;
}
