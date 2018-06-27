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

ImageControl::ImageControl(IImageLoader& rImageLoader, IPreprocessing& rPreprocessor, IBackgroundSubtraction& rBackgroundSubtraction,
		IStereoMatch& rStereomatcher, IPostProcessing& rPostProcessor, ISegmentation& rSegmentation) :
	mrImageLoader(rImageLoader),
	mrPreprocessor(rPreprocessor),
	mrBackgroundSubtraction(rBackgroundSubtraction),
	mrStereomatcher(rStereomatcher),
	mrPostprocessor(rPostProcessor),
	mrSegmentation(rSegmentation) {

}

ImageControl::~ImageControl() {

}

void ImageControl::Run() {

	for(size_t i=0;;++i) {
		cv::Mat oLeftImage = mrImageLoader.getNextLeftImage();
		cv::Mat oRightImage = mrImageLoader.getNextRightImage();


		if(oLeftImage.empty() || oRightImage.empty())	 break;

		cv::Mat oLeftPreprocessed = mrPreprocessor.Preprocess(oLeftImage);
		cv::Mat oRightPreprocessed = mrPreprocessor.Preprocess(oRightImage);


		cv::Mat oForegroundLeft = mrBackgroundSubtraction.SubtractLeft(oLeftPreprocessed);
		cv::Mat oForegroundRight = mrBackgroundSubtraction.SubtractRight(oRightPreprocessed);

		cv::Mat oDisparity = mrStereomatcher.Match(oForegroundLeft, oForegroundRight);

		cv::Mat oPostprocess = mrPostprocessor.Postprocess(oDisparity);

		//cv::Mat oSegmentation = mrSegmentation.Segment(oPostprocess);
		cv::Mat oSegmentation = oPostprocess;

		normalize(oSegmentation, oSegmentation, 255.0, 0.0, CV_MINMAX);


		maLeftImages.push_back(oLeftImage);
		maRightImages.push_back(oRightImage);
		maPreprocessLeft.push_back(oLeftPreprocessed);
		maPreprocessRight.push_back(oRightPreprocessed);
		maForegroundLeft.push_back(oForegroundLeft);
		maForegroundRight.push_back(oForegroundRight);
		//maDisparity.push_back(oDisparity);
		maPostprocessImages.push_back(oPostprocess);
		maSegmentation.push_back(oSegmentation);


		int iWidth = 960;
		int iHeight = 240;

		//cvtColor(oLeftImage, oLeftImage, CV_GRAY2BGR);
		cvtColor(oLeftPreprocessed, oLeftPreprocessed, CV_GRAY2BGR);
		cvtColor(oForegroundLeft, oForegroundLeft, CV_GRAY2BGR);
		cvtColor(oForegroundRight, oForegroundRight, CV_GRAY2BGR);
		cvtColor(oPostprocess, oPostprocess, CV_GRAY2BGR);
		cvtColor(oSegmentation, oSegmentation, CV_GRAY2BGR);

		applyColorMap(oDisparity, oDisparity, COLORMAP_JET);
		applyColorMap(oSegmentation, oSegmentation, COLORMAP_JET);

		maDisparity.push_back(oDisparity);

		cv::Mat oLeftPrep;
		hconcat(oLeftImage, oLeftPreprocessed, oLeftPrep);
		resize(oLeftPrep, oLeftPrep, Size(iWidth, iHeight));

		cv::Mat oForeground;
		hconcat(oForegroundLeft, oDisparity, oForeground);
		resize(oForeground, oForeground, Size(iWidth, iHeight));

		cv::Mat oStereoPostp;
		hconcat(oPostprocess, oSegmentation, oStereoPostp);
		resize(oStereoPostp, oStereoPostp, Size(iWidth, iHeight));


		cv::Mat oResult;
		vconcat(oLeftPrep, oForeground, oResult);
		vconcat(oResult, oStereoPostp, oResult);

		imshow("Result", oResult);


		int c = (char)waitKey(25);
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

const std::vector<cv::Mat>& ImageControl::getSegmentation() const {
	return maSegmentation;
}
