#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

int main() {
	cv::VideoCapture oImageStreamLeft("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c0.pgm");
	cv::VideoCapture oImageStreamRight("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c1.pgm");

	//cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_2/%06d_10.png");
	//cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_3/%06d_10.png");

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create();

	Mat oFrameLeft;
	Mat oFrameRight;
	Mat oDisparity;
	for(size_t i=0;; ++i) {
		oImageStreamLeft>>oFrameLeft;
		oImageStreamRight>>oFrameRight;

		oFrameLeft.convertTo(oFrameLeft, CV_8U, 255.0 / 4096.0);
		oFrameRight.convertTo(oFrameRight, CV_8U, 255.0 / 4096.0);

		//cvtColor(oFrameLeft, oFrameLeft, CV_BGR2GRAY);
		//cvtColor(oFrameRight, oFrameRight, CV_BGR2GRAY);

		sbm->compute(oFrameLeft, oFrameRight, oDisparity);

		oDisparity.convertTo(oDisparity, CV_8U, 1.0/16.0);
		normalize(oDisparity, oDisparity, 0.0, 255.0, CV_MINMAX);

		//applyColorMap(oDisparity, oDisparity, COLORMAP_JET);

		imshow("frame left", oFrameLeft);
		imshow("frame right", oFrameRight);
		imshow("disparity", oDisparity);

		char c = (char)waitKey(5000);
		if(c==27) 	break;
	}

	return 0;
}
