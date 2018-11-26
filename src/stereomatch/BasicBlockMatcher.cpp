#include "stereomatch/BasicBlockMatcher.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudastereo.hpp>

using namespace std;
using namespace cv;

BasicBlockMatcher::BasicBlockMatcher() : 
	miBlockSize(9),
	miNumDisparities(64) {

}

BasicBlockMatcher::~BasicBlockMatcher() {

}

void BasicBlockMatcher::setBlockSize(int iBlockSize) {
	assert(iBlockSize > 0);
	assert(iBlockSize % 2 == 1);

	miBlockSize = iBlockSize;
}

void BasicBlockMatcher::setNumDisparities(int iNumDisparities) {
	assert(iNumDisparities > 0);

	miNumDisparities = iNumDisparities;
}

cv::Mat BasicBlockMatcher::Match(const cv::Mat& rLeft, const cv::Mat& rRight) {
	/*cv::Mat oResult;

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create(miNumDisparities, miBlockSize);
	sbm->compute(rLeft, rRight, oResult);


	oResult.convertTo(oResult, CV_8U, 1.0/16.0);

	return oResult;*/

	cv::Mat oResult;
	cuda::GpuMat _pResult;

	cuda::GpuMat _pLeft(rLeft);
	cuda::GpuMat _pRight(rRight);
	cv::Ptr<cv::cuda::StereoBM> bp = cv::cuda::createStereoBM(miNumDisparities, miBlockSize);

	bp->compute(_pLeft, _pRight, _pResult);

	_pResult.download(oResult);

	oResult.convertTo(oResult, CV_8U, 1.0);

	return oResult;
}
