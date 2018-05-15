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
	cv::Ptr<cv::cuda::StereoBeliefPropagation> bp = cv::cuda::createStereoBeliefPropagation(64, 5, 5);
	bp->compute(_pLeft, _pRight, _pResult);

	_pResult.download(oResult);

	return oResult;
}
