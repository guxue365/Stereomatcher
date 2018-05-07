#include "ImageHandler.h"

#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

ImageHandler::ImageHandler(const std::string& sFolderLeft, const std::string& sFolderRight, const std::string& sFolderEvaluation, const std::string& sFolderResult) :
		msFolderLeft(sFolderLeft),
		msFolderRight(sFolderRight),
		msFolderEvaluation(sFolderEvaluation),
		msFolderResult(sFolderResult) {

}

ImageHandler::~ImageHandler() {

}

std::vector<cv::Mat> ImageHandler::LoadLeftImages() {
	vector<string> aFileNames = GetAllFileNames();

	vector<cv::Mat> aResult(aFileNames.size());
	for(size_t i=0; i<aResult.size(); ++i) {
		aResult[i] = imread(msFolderLeft+aFileNames[i]);
		if(aResult[i].data==NULL) 	throw std::string("Error loading file: "+msFolderLeft+aFileNames[i]);
	}
	return aResult;
}

std::vector<cv::Mat> ImageHandler::LoadRightImages() {
	vector<string> aFileNames = GetAllFileNames();

	vector<cv::Mat> aResult(aFileNames.size());
	for(size_t i=0; i<aResult.size(); ++i) {
		aResult[i] = imread(msFolderRight+aFileNames[i]);
		if(aResult[i].data==NULL) 	throw std::string("Error loading file: "+msFolderRight+aFileNames[i]);
	}
	return aResult;
}

std::vector<cv::Mat> ImageHandler::LoadEvaluationImages() {
	vector<string> aFileNames = GetAllFileNames();

	vector<cv::Mat> aResult(aFileNames.size());
	for(size_t i=0; i<aResult.size(); ++i) {
		aResult[i] = imread(msFolderEvaluation+aFileNames[i], CV_LOAD_IMAGE_ANYDEPTH);
		if(aResult[i].data==NULL) 	throw std::string("Error loading file: "+msFolderEvaluation+aFileNames[i]);
	}
	return aResult;
}

std::vector<std::string> ImageHandler::GetAllFileNames() {
	vector<string> Result;

	for(boost::filesystem::directory_iterator it(msFolderLeft); it!=boost::filesystem::directory_iterator(); it++) {
		Result.push_back(it->path().filename().c_str());
	}

	return Result;
}

void ImageHandler::StoreResult(const cv::Mat& rImage, const std::string& sFilename) {
	imwrite(msFolderResult+sFilename, rImage);
}

void ImageHandler::StoreEvaluation(const cv::Mat& rImageEval, const std::string& sFilename) {
	imwrite(msFolderResult+"eval/"+sFilename, rImageEval);
}
