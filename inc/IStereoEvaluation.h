#pragma once

#include <opencv2/core.hpp>

class IStereoEvaluation {
public:
	virtual ~IStereoEvaluation() {};

	virtual double Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage) = 0;

	virtual cv::Mat getVisualRepresentation() = 0;
};
