#pragma once

#include "IStereoEvaluation.h"

class EvaluateBPP : public IStereoEvaluation {
public:
	EvaluateBPP(double dTolerance = 1.0);
	virtual ~EvaluateBPP();

	double Evaluate(const cv::Mat& rGroundTruth, const cv::Mat& rDisparityImage);

	cv::Mat getVisualRepresentation();
private:
	cv::Mat moErrorMap;
	double mdTolerance;
};
