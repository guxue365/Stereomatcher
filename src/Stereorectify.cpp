#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

void TryCalibration(const Mat& oFrameLeft, const Mat& oFrameRight, Mat& oOutLeft, Mat& oOutRight);

int main() {


	cv::VideoCapture oImageStreamLeft("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c0.pgm");
	cv::VideoCapture oImageStreamRight("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c1.pgm");

	Mat oFrameLeft;
	Mat oFrameRight;

	Mat oRectLeft;
	Mat oRectRight;
	for(size_t i=0;;++i) {

		oImageStreamLeft>>oFrameLeft;
		oImageStreamRight>>oFrameRight;

		oFrameLeft.convertTo(oFrameLeft, CV_8U, 255.0 / 4096.0);
		oFrameRight.convertTo(oFrameRight, CV_8U, 255.0 / 4096.0);



		Mat oOriginal;
		hconcat(oFrameLeft, oFrameRight, oOriginal);
		resize(oOriginal, oOriginal, Size(1680, 420), INTER_CUBIC);

		TryCalibration(oFrameLeft, oFrameRight, oRectLeft, oRectRight);

		Mat oRect;
		hconcat(oRectLeft, oRectRight, oRect);
		resize(oRect, oRect, Size(1680, 420), INTER_CUBIC);

		Mat oResult;
		vconcat(oOriginal, oRect, oResult);

		imshow("Result", oResult);

		imwrite("result.png", oResult);

		char c = (char)waitKey(0);
		if(c==27) 	break;
	}

	return 0;
}

void TryCalibration(const Mat& oFrameLeft, const Mat& oFrameRight, Mat& oOutLeft, Mat& oOutRight) {
	double fx1 = 2.500744557379985e+03;
	double fy1 = 2.535988340208410e+03;
	double cx1 = 1.260414892498634e+03;
	double cy1 = 5.260783408403682e+02;

	double fx2 = 2.472932850391490e+03;
	double fy2 = 2.493700599346621e+03;
	double cx2 = 1.284997418662109e+03;
	double cy2 = 5.092517594678745e+02;

	double k11 = -0.358747279897952;
	double k21 = -5.482777505594711;
	double k31 = 21.789612196728840;
	double p11 = -0.026661723125205;
	double p21 = 0.030967281664763;

	double k12 = -0.608013415673151;
	double k22 = -1.374768603631144;
	double k32 = 5.002510805942969;
	double p12 = 0.014811903964771;
	double p22 = 0.006712822591838;

	Mat oCameraMatrix1 = (Mat_<double>(3, 3)<<fx1, 0.0, cx1, 0.0, fy1, cy1, 0.0, 0.0, 1.0);
	Mat oCameraMatrix2 = (Mat_<double>(3, 3)<<fx2, 0.0, cx2, 0.0, fy2, cy2, 0.0, 0.0, 1.0);

	Mat oDistortionVec1 = (Mat_<double>(5, 1)<<k11, k21, p11, p21, k31);
	Mat oDistortionVec2 = (Mat_<double>(5, 1)<<k12, k22, p12, p22, k32);

	Size oSize = Size(1024, 2048);

	Mat oRotationMatrix = (Mat_<double>(3, 3)<<0.998818097562346, -0.040321229984821, 0.027140493629374,
												0.041806070439880, 0.997523047404669, -0.056568740227069,
												-0.024792346728593, 0.057636518883543, 0.998029744664294);

	Mat oTranslationVector = (Mat_<double>(3, 1)<<-4.778183165716090e+02, 50.433147703073980, -52.114246295362406);

	cout<<"Camera 1 Matrix: "<<endl;
	cout<<oCameraMatrix1<<endl;

	cout<<"Camera 2 Matrix: "<<endl;
	cout<<oCameraMatrix2<<endl;

	cout<<"Distortion Vec 1: "<<endl;
	cout<<oDistortionVec1<<endl;

	cout<<"Distortion Vec 2: "<<endl;
	cout<<oDistortionVec2<<endl;

	cout<<"Size: "<<endl;
	cout<<oSize<<endl;

	cout<<"Rotation Matrix: "<<endl;
	cout<<oRotationMatrix<<endl;

	cout<<"Translation Vec: "<<endl;
	cout<<oTranslationVector<<endl;

	Mat R1, R2;
	Mat P1, P2;
	Mat Q;

	stereoRectify(oCameraMatrix1, oDistortionVec1, oCameraMatrix2, oDistortionVec2, oSize, oRotationMatrix, oTranslationVector,
			R1, R2, P1, P2, Q);

	cout<<"R1: "<<endl<<R1<<endl;
	cout<<"R2: "<<endl<<R2<<endl;
	cout<<"P1: "<<endl<<P1<<endl;
	cout<<"P2: "<<endl<<P2<<endl;
	cout<<"Q: "<<endl<<Q<<endl;

	Mat map11, map21;
	Mat map12, map22;

	initUndistortRectifyMap(oCameraMatrix1, oDistortionVec1, R1, P1, oSize, CV_32FC1, map11, map21);
	initUndistortRectifyMap(oCameraMatrix2, oDistortionVec2, R2, P2, oSize, CV_32FC1, map12, map22);


	remap(oFrameLeft, oOutLeft, map11, map21, INTER_LINEAR, BORDER_CONSTANT);
	remap(oFrameRight, oOutRight, map12, map22, INTER_LINEAR, BORDER_CONSTANT);
}
