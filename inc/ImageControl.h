#pragma once

#include <vector>

#include "IImageLoader.h"
#include "IPreprocessing.h"
#include "IBackgroundSubtraction.h"
#include "ISegmentation.h"
#include "IPostprocessing.h"
#include "IStereoMatch.h"

class ImageControl {
public:
	ImageControl(IImageLoader& rImageLoader, IPreprocessing& rPreprocessor, IBackgroundSubtraction& rBackgroundSubtraction,
			IStereoMatch& rStereomatcher, IPostProcessing& rPostProcessor, ISegmentation& rSegmentation);
	virtual ~ImageControl();

	void Run();

	const std::vector<cv::Mat>& getLeftImages() const;
	const std::vector<cv::Mat>& getRightImages() const;
	const std::vector<cv::Mat>& getPreprocessLeft() const;
	const std::vector<cv::Mat>& getPreprocessRight() const;
	const std::vector<cv::Mat>& getForegroundLeft() const;
	const std::vector<cv::Mat>& getForegroundRight() const;
	const std::vector<cv::Mat>& getDisparity() const;
	const std::vector<cv::Mat>& getPostprocessImages() const;
	const std::vector<cv::Mat>& getSegmentation() const;
	const std::vector<std::vector<Cluster> >& getCluster() const;
private:
	IImageLoader& mrImageLoader;
	IPreprocessing& mrPreprocessor;
	IBackgroundSubtraction& mrBackgroundSubtraction;
	IStereoMatch& mrStereomatcher;
	IPostProcessing& mrPostprocessor;
	ISegmentation& mrSegmentation;

	std::vector<cv::Mat> maLeftImages;
	std::vector<cv::Mat> maRightImages;
	std::vector<cv::Mat> maPreprocessLeft;
	std::vector<cv::Mat> maPreprocessRight;
	std::vector<cv::Mat> maForegroundLeft;
	std::vector<cv::Mat> maForegroundRight;
	std::vector<cv::Mat> maDisparity;
	std::vector<cv::Mat> maPostprocessImages;
	std::vector<cv::Mat> maSegmentation;
	std::vector<std::vector<Cluster> > maCluster;
};
