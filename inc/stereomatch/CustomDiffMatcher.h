#pragma once

#include <IStereoMatch.h>

class CustomDiffMatcher : public IStereoMatch {
public:
	CustomDiffMatcher();
	virtual ~CustomDiffMatcher();

	void setNumDisparities(int iNumDisparities);
	void setBlockWidth(int iBlockWidth);
	void setBlockHeight(int iBlockHeight);
	void setValidTolerance(double dTolerance);
	void setDiffOrderX(int iDiffOrderX);
	void setDiffOrderY(int iDiffOrderY);
	void setSobelKernelSize(int iSobelKernelSize);

	cv::Mat Match(const cv::Mat& rLeft, const cv::Mat& rRight);
private:
	int miNumDisparities;
	int miBlockWidth;
	int miBlockHeight;
	double mdTolerance;
	int miDiffOrderX;
	int miDiffOrderY;
	int miSobelKernelSize; //it must be 1, 3, 5, or 7.
};
