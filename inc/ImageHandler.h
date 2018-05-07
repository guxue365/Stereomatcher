#pragma once

#include <iostream>
#include <vector>
#include <tuple>

#include <opencv2/core.hpp>

class ImageHandler {
public:
	ImageHandler(const std::string& sFolderLeft, const std::string& sFolderRight, const std::string& sFolderEvaluation, const std::string& sFolderResult);
	virtual ~ImageHandler();

	std::vector<cv::Mat> LoadLeftImages();
	std::vector<cv::Mat> LoadRightImages();
	std::vector<cv::Mat> LoadEvaluationImages();

	std::vector<std::string> GetAllFileNames();

	void StoreResult(const cv::Mat& rImage, const std::string& sFilename);
	void StoreEvaluation(const cv::Mat& rImageEval, const std::string& sFilename);
private:
	std::string msFolderLeft;
	std::string msFolderRight;
	std::string msFolderEvaluation;
	std::string msFolderResult;
};
