#pragma once

#include "IStereoMatch.h"

class BasicBPMatcher : public IStereoMatch {
public:
	BasicBPMatcher();
	virtual ~BasicBPMatcher();

	void setNumDisparities(int iNumDisparities);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
};
