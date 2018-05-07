#pragma once

#include <vector>

#include "ImageHandler.h"
#include "IPreprocessing.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"
#include "IStereoEvaluation.h"

class ImageControl {
public:
	ImageControl(ImageHandler& rImageHandler, IPreprocessing& rPreprocessor, IPostProcessing& rPostProcessor, IStereoMatch& rStereomatcher, IStereoEvaluation& rStereoEvaluation);
	virtual ~ImageControl();

	void LoadImages();
	void Run();
	void StoreResults();
	void Evaluate();

	const std::vector<cv::Mat>& getLeftImages() const;
	const std::vector<cv::Mat>& getRightImages() const;
	const std::vector<cv::Mat>& getResultImages() const;
	const std::vector<cv::Mat>& getEvaluationImages() const;
private:
	ImageHandler& mrImageHandler;
	IPreprocessing& mrPreprocessor;
	IPostProcessing& mrPostprocessor;
	IStereoMatch& mrStereomatcher;
	IStereoEvaluation& mrStereoEvaluation;

	std::vector<cv::Mat> maLeftImages;
	std::vector<cv::Mat> maRightImages;
	std::vector<std::string> maFilenames;

	std::vector<cv::Mat> maResultImages;
	std::vector<cv::Mat> maEvaluationImages;
};
