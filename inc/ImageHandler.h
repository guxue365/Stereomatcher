#pragma once

#include <iostream>
#include <vector>
#include <tuple>

#include <opencv2/core.hpp>

class ImageHandler {
public:
	ImageHandler(const std::string& sFolderLeft, const std::string& sFolderRight, const std::string& sFolderResult);
	virtual ~ImageHandler();

	std::vector<cv::Mat> LoadLeftImages();
	std::vector<cv::Mat> LoadRightImages();

	std::vector<std::string> GetAllFileNames();

	void StorePreprocess(const cv::Mat& rImage, const std::string& sFilename);
	void StoreForeground(const cv::Mat& rImage, const std::string& sFilename);
	void StoreStereo(const cv::Mat& rImage, const std::string& sFilename);
	void StorePostprocess(const cv::Mat& rImage, const std::string& sFilename);
private:
	std::string msFolderLeft;
	std::string msFolderRight;
	std::string msFolderResult;
};
