#include "ImageControl.h"

#include <cassert>

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

ImageControl::ImageControl(ImageHandler& rImageHandler, IPreprocessing& rPreprocessor, IPostProcessing& rPostProcessor, IStereoMatch& rStereomatcher) :
	mrImageHandler(rImageHandler),
	mrPreprocessor(rPreprocessor),
	mrPostprocessor(rPostProcessor),
	mrStereomatcher(rStereomatcher) {

}

ImageControl::~ImageControl() {

}

void ImageControl::LoadImages() {
	maLeftImages = mrImageHandler.LoadLeftImages();
	maRightImages = mrImageHandler.LoadRightImages();
	maFilenames = mrImageHandler.GetAllFileNames();

	assert(maLeftImages.size()==maRightImages.size());
	assert(maLeftImages.size()==maFilenames.size());

	maPreprocessImages.resize(maLeftImages.size());
	maForegroundImages.resize(maLeftImages.size());
	maStereoImages.resize(maLeftImages.size());
	maPostprocessImages.resize(maLeftImages.size());
}

void ImageControl::Run() {
	for(size_t i=0; i<maLeftImages.size(); ++i) {
		cv::Mat& rLeftImage = maLeftImages[i];
		cv::Mat& rRightImage = maRightImages[i];

		cv::Mat oLeftPreprocessed = mrPreprocessor.Preprocess(rLeftImage);
		cv::Mat oRightPreprocessed = mrPreprocessor.Preprocess(rRightImage);

		cv::Mat oDisparity = mrStereomatcher.Match(oLeftPreprocessed, oRightPreprocessed);

		cv::Mat oPostprocessed = mrPostprocessor.Postprocess(oDisparity);

		maPreprocessImages[i] = oLeftPreprocessed;
		maForegroundImages[i] = oLeftPreprocessed;
		maStereoImages[i] = oDisparity;
		maPostprocessImages[i] = oPostprocessed;
	}
}

void ImageControl::StoreResults() {
	for(size_t i=0; i<maPreprocessImages.size(); ++i) {
		mrImageHandler.StorePreprocess(maPreprocessImages[i], maFilenames[i]);
		mrImageHandler.StoreForeground(maForegroundImages[i], maFilenames[i]);
		mrImageHandler.StoreStereo(maStereoImages[i], maFilenames[i]);
		mrImageHandler.StorePostprocess(maPostprocessImages[i], maFilenames[i]);
	}
}

const std::vector<cv::Mat>& ImageControl::getLeftImages() const {
	return maLeftImages;
}

const std::vector<cv::Mat>& ImageControl::getRightImages() const {
	return maRightImages;
}

const std::vector<cv::Mat>& ImageControl::getPreprocessImages() const {
	return maPreprocessImages;
}

const std::vector<cv::Mat>& ImageControl::getForegroundImages() const {
	return maForegroundImages;
}

const std::vector<cv::Mat>& ImageControl::getStereoImages() const {
	return maStereoImages;
}

const std::vector<cv::Mat>& ImageControl::getPostprocessImages() const {
	return maPostprocessImages;
}
