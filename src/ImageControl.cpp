#include "ImageControl.h"

#include <cassert>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

ImageControl::ImageControl(IImageLoader& rImageLoader, IPreprocessing& rPreprocessor, IPostProcessing& rPostProcessor, IStereoMatch& rStereomatcher) :
	mrImageLoader(rImageLoader),
	mrPreprocessor(rPreprocessor),
	mrPostprocessor(rPostProcessor),
	mrStereomatcher(rStereomatcher) {

}

ImageControl::~ImageControl() {

}

void ImageControl::Run() {
	for(;;) {
		cv::Mat oLeftImage = mrImageLoader.getNextLeftImage();
		cv::Mat oRightImage = mrImageLoader.getNextRightImage();

		if(oLeftImage.empty() || oRightImage.empty())	 break;

		cv::Mat oLeftPreprocessed = mrPreprocessor.Preprocess(oLeftImage);
		cv::Mat oRightPreprocessed = mrPreprocessor.Preprocess(oRightImage);

		cv::Mat oDisparity = mrStereomatcher.Match(oLeftPreprocessed, oRightPreprocessed);

		/*cv::Mat oPostprocessed = mrPostprocessor.Postprocess(oDisparity);


		maLeftImages.push_back(oLeftImage);
		maRightImages.push_back(oRightImage);
		maPreprocessLeft.push_back(oLeftPreprocessed);
		maPreprocessRight.push_back(oRightPreprocessed);
		maForegroundLeft.push_back(oLeftPreprocessed);
		maForegroundRight.push_back(oRightPreprocessed);
		maDisparity.push_back(oDisparity);
		maPostprocessImages.push_back(oPostprocessed);*/

		cv::Mat oOriginal;
		hconcat(oLeftImage, oRightImage, oOriginal);
		resize(oOriginal, oOriginal, Size(1280, 320));

		cv::Mat oPreprocess;
		hconcat(oLeftPreprocessed, oRightPreprocessed, oPreprocess);
		resize(oPreprocess, oPreprocess, Size(1280, 320));

		cv::Mat oResult;
		vconcat(oOriginal, oPreprocess, oResult);

		imshow("Result", oResult);

		char c = (char)waitKey(50);
		if(c==27) 	break;
	}
}

const std::vector<cv::Mat>& ImageControl::getPreprocessLeft() const  {
	return maPreprocessLeft;
}

const std::vector<cv::Mat>& ImageControl::getPreprocessRight() const  {
	return maPreprocessRight;
}

const std::vector<cv::Mat>& ImageControl::getForegroundLeft() const  {
	return maForegroundLeft;
}

const std::vector<cv::Mat>& ImageControl::getForegroundRight() const  {
	return maForegroundRight;
}

const std::vector<cv::Mat>& ImageControl::getDisparity() const  {
	return maDisparity;
}

const std::vector<cv::Mat>& ImageControl::getPostprocessImages() const  {
	return maPostprocessImages;
}
