#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

int main() {
	//cv::VideoCapture oImageStreamLeft("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c0.pgm");
	//cv::VideoCapture oImageStreamRight("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c1.pgm");
	//cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/2018-05-16_07-35-15_ZA1/RectifiedImages/img_%05d_c0.pgm");
	//cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/2018-05-16_07-35-15_ZA1/RectifiedImages/img_%05d_c1.pgm");
	cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/Rec01_ZA1/RectifiedLeft/img_%1d.png");
	cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/Rec01_ZA1/RectifiedRight/img_%1d.png");
	//cv::VideoCapture oImageStreamLeft("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_2/%06d_10.png");
	//cv::VideoCapture oImageStreamRight("/home/jung/2018EntwicklungStereoalgorithmus/data/kitty/data_scene_flow/training/image_3/%06d_10.png");

	if(!oImageStreamLeft.isOpened()) {
		cout<<"Failed open left"<<endl;
	}
	if(!oImageStreamRight.isOpened()) {
		cout<<"Failed open right"<<endl;
	}

	cv::Ptr<cv::StereoBM> sbm = cv::StereoBM::create();

	Mat oFrameLeft;
	Mat oFrameRight;

	Mat oMaskLeft = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/mask_left.png", IMREAD_GRAYSCALE);
	Mat oMaskRight = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/mask_right.png", IMREAD_GRAYSCALE);


	double min, max;
	for(size_t i=0;; ++i) {
		oImageStreamLeft>>oFrameLeft;
		oImageStreamRight>>oFrameRight;

		Mat oFrameWithMask;

		cvtColor(oFrameLeft, oFrameLeft, COLOR_RGB2GRAY, 1);
		cvtColor(oFrameRight, oFrameRight, COLOR_RGB2GRAY, 1);

		imshow("left", oFrameLeft);
		imshow("right", oFrameRight);

		oFrameLeft.copyTo(oFrameWithMask, oMaskLeft);

		imshow("with mask", oFrameWithMask);

		char c = (char)waitKey();
		if(c==27) 	break;
	}

	return 0;
}
