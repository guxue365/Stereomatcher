#include "stereomatch/BasicBPMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudastereo.hpp>

using namespace std;
using namespace cv;

BasicBPMatcher::BasicBPMatcher() {

}

BasicBPMatcher::~BasicBPMatcher() {

}

cv::Mat BasicBPMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;
	cuda::GpuMat _pResult;

	cuda::GpuMat _pLeft(rLeft);
	cuda::GpuMat _pRight(rRight);
	cv::Ptr<cv::cuda::StereoBeliefPropagation> bp = cv::cuda::createStereoConstantSpaceBP();
	bp->compute(_pLeft, _pRight, _pResult);

	_pResult.download(oResult);

	oResult.convertTo(oResult, CV_8U, 1.0);
	//normalize(oResult, oResult, 0.0, 255.0, CV_MINMAX);

	return oResult;
}
