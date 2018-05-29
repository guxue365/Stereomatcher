#include <imageloader/BaseImageloader.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

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
	demosaicing(oFrame, oFrame, COLOR_BayerGR2RGB);
	oFrame.convertTo(oFrame, CV_8UC3, 1.0/256.0);
	return oFrame;
}

cv::Mat BaseImageloader::getNextRightImage() {
	cv::Mat oFrame;
	moRightImages>>oFrame;
	demosaicing(oFrame, oFrame, COLOR_BayerGR2RGB);
	oFrame.convertTo(oFrame, CV_8UC3, 1.0/256.0);
	return oFrame;
}
