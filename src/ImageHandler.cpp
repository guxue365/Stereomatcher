#include "ImageHandler.h"

#include <opencv2/highgui.hpp>
#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;

ImageHandler::ImageHandler(const std::string& sFolderLeft, const std::string& sFolderRight, const std::string& sFolderResult) :
		msFolderLeft(sFolderLeft),
		msFolderRight(sFolderRight),
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

std::vector<std::string> ImageHandler::GetAllFileNames() {
	vector<string> Result;

	for(boost::filesystem::directory_iterator it(msFolderLeft); it!=boost::filesystem::directory_iterator(); it++) {
		Result.push_back(it->path().filename().c_str());
	}

	return Result;
}

void ImageHandler::StorePreprocess(const cv::Mat& rImage, const std::string& sFilename) {
	imwrite(msFolderResult+"preprocess/"+sFilename, rImage);
}

void ImageHandler::StoreForeground(const cv::Mat& rImage, const std::string& sFilename) {
	imwrite(msFolderResult+"foreground/"+sFilename, rImage);
}

void ImageHandler::StoreStereo(const cv::Mat& rImage, const std::string& sFilename)  {
	imwrite(msFolderResult+"stereo/"+sFilename, rImage);
}

void ImageHandler::StorePostprocess(const cv::Mat& rImage, const std::string& sFilename) {
	imwrite(msFolderResult+"postprocess/"+sFilename, rImage);
}

