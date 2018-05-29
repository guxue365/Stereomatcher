#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

int main() {
	//cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/20180515RecordingCalibration/2018-05-15_11-09-07_ZA2_LargePattern/MatlabCalibration/Left/l_img_%05d.pgm");
	//cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/20180515RecordingCalibration/2018-05-15_11-09-07_ZA2_LargePattern/MatlabCalibration/Right/r_img_%05d.pgm");

	//cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/opencv-3.4.1/samples/data/left%02d.jpg");
	//cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/opencv-3.4.1/samples/data/right%02d.jpg");

	cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/20180515RecordingCalibration/2018-05-15_11-23-04_ZA1_LargePattern/RenamedImages/img_%05d_c0.pgm");
	cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/20180515RecordingCalibration/2018-05-15_11-23-04_ZA1_LargePattern/RenamedImages/img_%05d_c1.pgm");


	if(!oImageStreamLeft.isOpened() || !oImageStreamRight.isOpened()) {
		cout<<"Error opening files"<<endl;
		return -1;
	}


	Mat oFrameLeft;
	Mat oFrameRight;

	for(size_t i=0;;++i) {
		oImageStreamLeft>>oFrameLeft;
		oImageStreamRight>>oFrameRight;

		if(oFrameLeft.empty() || oFrameRight.empty()) 	break;

		demosaicing(oFrameLeft, oFrameLeft, COLOR_BayerGR2RGB);
		demosaicing(oFrameRight, oFrameRight, COLOR_BayerGR2RGB);

		oFrameLeft.convertTo(oFrameLeft, CV_8UC3, 255.0 / 4096.0);
		oFrameRight.convertTo(oFrameRight, CV_8UC3, 255.0 / 4096.0);

		cvtColor(oFrameLeft, oFrameLeft, COLOR_BGR2GRAY);
		cvtColor(oFrameRight, oFrameRight, COLOR_BGR2GRAY);

		Size oPatternSize(3, 4);
		vector<Point2f> aCornersLeft;
		vector<Point2f> aCornersRight;

		bool bLeftFound = findChessboardCorners(oFrameLeft, oPatternSize, aCornersLeft,  CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
		        + CALIB_CB_FAST_CHECK);

		bool bRightFound = findChessboardCorners(oFrameRight, oPatternSize, aCornersRight,  CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
				        + CALIB_CB_FAST_CHECK);

		cout<<"Analysing left chessboard: "<<(bLeftFound ? "found" : "not found")<<endl;
		cout<<"Analysing right chessboard: "<<(bRightFound ? "found" : "not found")<<endl;
		cout<<bLeftFound<<endl;

		imshow("left", oFrameLeft);
		imshow("right", oFrameRight);

		char c = (char)waitKey(0);
		if(c==27) 	break;
	}


	return 0;
}
