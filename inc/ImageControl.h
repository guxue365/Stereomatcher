#pragma once

#include <vector>

#include "ImageHandler.h"
#include "IPreprocessing.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"
#include "IStereoEvaluation.h"

class ImageControl {
public:
	ImageControl(ImageHandler& rImageHandler, IPreprocessing& rPreprocessor, IPostProcessing& rPostProcessor, IStereoMatch& rStereomatcher);
	virtual ~ImageControl();

	void LoadImages();
	void Run();
	void StoreResults();

	const std::vector<cv::Mat>& getLeftImages() const;
	const std::vector<cv::Mat>& getRightImages() const;

	const std::vector<cv::Mat>& getPreprocessImages() const;
	const std::vector<cv::Mat>& getForegroundImages() const;
	const std::vector<cv::Mat>& getStereoImages() const;
	const std::vector<cv::Mat>& getPostprocessImages() const;
private:
	ImageHandler& mrImageHandler;
	IPreprocessing& mrPreprocessor;
	IPostProcessing& mrPostprocessor;
	IStereoMatch& mrStereomatcher;

	std::vector<cv::Mat> maLeftImages;
	std::vector<cv::Mat> maRightImages;
	std::vector<std::string> maFilenames;

	std::vector<cv::Mat> maPreprocessImages;
	std::vector<cv::Mat> maForegroundImages;
	std::vector<cv::Mat> maStereoImages;
	std::vector<cv::Mat> maPostprocessImages;
};
