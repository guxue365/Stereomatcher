#pragma once

#include <IPostprocessing.h>

class PostInterpolation : public IPostProcessing {
public:
	PostInterpolation();
	virtual ~PostInterpolation();

	cv::Mat Postprocess(const cv::Mat& rImage);
private:
	cv::Mat InterpolateKittiGT(const cv::Mat& rImage);
	double getMeanNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize);
	double getMedianNeighborValues(const cv::Mat& rImage, int row, int col, int boxsize);
};
