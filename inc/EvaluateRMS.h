#pragma once

#include "IStereoEvaluation.h"

class EvaluateRMS : public IStereoEvaluation {
public:
	EvaluateRMS();
	virtual ~EvaluateRMS();

	double Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage);

	cv::Mat getVisualRepresentation();
private:
	cv::Mat moErrorMap;
};

