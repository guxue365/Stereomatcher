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

ImageControl::ImageControl(ImageHandler& rImageHandler, IPreprocessing& rPreprocessor, IPostProcessing& rPostProcessor, IStereoMatch& rStereomatcher, IStereoEvaluation& rStereoEvaluation) :
	mrImageHandler(rImageHandler),
	mrPreprocessor(rPreprocessor),
	mrPostprocessor(rPostProcessor),
	mrStereomatcher(rStereomatcher),
	mrStereoEvaluation(rStereoEvaluation) {

}

ImageControl::~ImageControl() {

}

void ImageControl::LoadImages() {
	maLeftImages = mrImageHandler.LoadLeftImages();
	maRightImages = mrImageHandler.LoadRightImages();
	maEvaluationImages = mrImageHandler.LoadEvaluationImages();
	maFilenames = mrImageHandler.GetAllFileNames();

	assert(maLeftImages.size()==maRightImages.size());
	assert(maLeftImages.size()==maFilenames.size());

	maResultImages.resize(maLeftImages.size());
}

void ImageControl::Run() {
	for(size_t i=0; i<maLeftImages.size(); ++i) {
		cv::Mat& rLeftImage = maLeftImages[i];
		cv::Mat& rRightImage = maRightImages[i];

		cv::Mat oLeftPreprocessed = mrPreprocessor.Preprocess(rLeftImage);
		cv::Mat oRightPreprocessed = mrPreprocessor.Preprocess(rRightImage);

		cv::Mat oDisparity = mrStereomatcher.Match(oLeftPreprocessed, oRightPreprocessed);

		cv::Mat oPostprocessed = mrPostprocessor.Postprocess(oDisparity);

		maResultImages[i] = oPostprocessed;
	}
}

void ImageControl::StoreResults() {
	for(size_t i=0; i<maResultImages.size(); ++i) {
		mrImageHandler.StoreResult(maResultImages[i], maFilenames[i]);
	}
}

void ImageControl::Evaluate() {
	for(size_t i=0; i<maEvaluationImages.size(); ++i) {
		double dError = mrStereoEvaluation.Evaluate(maEvaluationImages[i], maResultImages[i]);
		cv::Mat oEvaluationImage = mrStereoEvaluation.getVisualRepresentation();
		mrImageHandler.StoreEvaluation(oEvaluationImage, maFilenames[i]);
		cout<<"File "<<maFilenames[i]<<" Error: "<<dError<<endl;
	}
}

const std::vector<cv::Mat>& ImageControl::getResultImages() const {
	return maResultImages;
}

const std::vector<cv::Mat>& ImageControl::getLeftImages() const {
	return maLeftImages;
}

const std::vector<cv::Mat>& ImageControl::getRightImages() const {
	return maRightImages;
}

const std::vector<cv::Mat>& ImageControl::getEvaluationImages() const {
	return maEvaluationImages;
}
