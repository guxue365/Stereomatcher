#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/cudastereo.hpp>

using namespace std;
using namespace cv;

cv::Mat ComputeOpenCVBlockMatch(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize = 7);
cv::Mat ComputeOpenCVBeliefPropagation(const cv::Mat& rLeft, const cv::Mat& rRight);

int main() {
	VideoCapture oFilestreamLeft("E:/dataset_kitti/data_scene_flow/training/image_2/%06d_10.png");
	VideoCapture oFilestreamRight("E:/dataset_kitti/data_scene_flow/training/image_3/%06d_10.png");
	VideoCapture oFilestreamGT("E:/dataset_kitti/data_scene_flow/training/disp_noc_1/%06d_10.png");

	if (!oFilestreamLeft.isOpened() || !oFilestreamRight.isOpened() || !oFilestreamGT.isOpened()) {
		cout << "Error opening files" << endl;
		return -1;
	}

	Mat oFrameLeftColor;
	Mat oFrameLeftGray;
	Mat oFrameRightColor;
	Mat oFrameRightGray;

	Mat oDisparityOpenCVBM;
	Mat oDisparityOpenCVBP;
	Mat oDisparityGT;

	for (int iFrame = 0; ; ++iFrame) {
		oFilestreamLeft >> oFrameLeftColor;
		oFilestreamRight >> oFrameRightColor;
		oFilestreamGT >> oDisparityGT;
		oDisparityGT.convertTo(oDisparityGT, CV_8U, 1.0 / 255.0);

		if (oFrameLeftColor.empty() || oFrameRightColor.empty() || oDisparityGT.empty())	break;

		cvtColor(oFrameLeftColor, oFrameLeftGray, COLOR_BGR2GRAY);
		cvtColor(oFrameRightColor, oFrameRightGray, COLOR_BGR2GRAY);

		oDisparityOpenCVBM = ComputeOpenCVBlockMatch(oFrameLeftGray, oFrameRightGray);
		oDisparityOpenCVBP = ComputeOpenCVBeliefPropagation(oFrameLeftGray, oFrameRightGray);

		oDisparityOpenCVBM *= 3;
		oDisparityOpenCVBP *= 3;
		oDisparityGT *= 3;

		applyColorMap(oDisparityOpenCVBM, oDisparityOpenCVBM, COLORMAP_JET);
		applyColorMap(oDisparityOpenCVBP, oDisparityOpenCVBP, COLORMAP_JET);
		applyColorMap(oDisparityGT, oDisparityGT, COLORMAP_JET);

		imshow("Disparity OpenCV Blockmatching", oDisparityOpenCVBM);
		imshow("Disparity OpenCV Belief Propagation", oDisparityOpenCVBP);
		imshow("Disparity GT", oDisparityGT);

		char c = (char)waitKey();
		if (c == 27) 	break;
	}

	return 0;
}

cv::Mat ComputeOpenCVBlockMatch(const cv::Mat& rLeft, const cv::Mat& rRight, int iBoxSize) {
	cv::Mat oResult;

	cv::Ptr<StereoBM> pStereoBM = StereoBM::create(0, iBoxSize);
	pStereoBM->compute(rLeft, rRight, oResult);

	oResult.convertTo(oResult, CV_8U, 1.0 / 16.0);

	return oResult;
}

cv::Mat ComputeOpenCVBeliefPropagation(const cv::Mat& rLeft, const cv::Mat& rRight) {
	cv::Mat oResult;
	cuda::GpuMat _pResult;


	cuda::GpuMat _pLeft(rLeft);
	cuda::GpuMat _pRight(rRight);

	cv::Ptr<cv::cuda::StereoBeliefPropagation> bp = cv::cuda::createStereoConstantSpaceBP();
	bp->compute(_pLeft, _pRight, _pResult);

	_pResult.download(oResult);

	oResult.convertTo(oResult, CV_8U);

	return oResult;
}