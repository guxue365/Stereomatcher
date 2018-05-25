#include <imageloader/BaseImageloader.h>

#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

BaseImageloader::BaseImageloader() {

}

BaseImageloader::~BaseImageloader() {

}

bool BaseImageloader::Init(const std::string& sFolder) {
	moLeftImages.open(sFolder+"img_%05d_c0.pgm");
	moRightImages.open(sFolder+"img_%05d_c1.pgm");

	return (moLeftImages.isOpened() && moRightImages.isOpened());
}

cv::Mat BaseImageloader::getNextLeftImage() {
	cv::Mat oFrame;
	moLeftImages>>oFrame;
	oFrame.convertTo(oFrame, CV_8U, 255.0/4096.0);
	return oFrame;
}

cv::Mat BaseImageloader::getNextRightImage() {
	cv::Mat oFrame;
	moRightImages>>oFrame;
	oFrame.convertTo(oFrame, CV_8U, 255.0/4096.0);
	return oFrame;
}
