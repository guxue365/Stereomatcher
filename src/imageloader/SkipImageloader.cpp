#include <imageloader/SkipImageloader.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

SkipImageloader::SkipImageloader() {

}

SkipImageloader::~SkipImageloader() {

}

bool SkipImageloader::Init(const std::string& sFolder) {
	moLeftImages.open(sFolder+"img_%d_c0.png");
	moRightImages.open(sFolder+"img_%d_c1.png");
	if(!moLeftImages.isOpened()) {
		cout<<"Failed open left"<<endl;
	}
	if(!moRightImages.isOpened()) {
		cout<<"Failed open right"<<endl;
	}

	return (moLeftImages.isOpened() && moRightImages.isOpened());
}

cv::Mat SkipImageloader::getNextLeftImage() {
	cv::Mat oFrame;
	moLeftImages>>oFrame;
	return oFrame;
}

cv::Mat SkipImageloader::getNextRightImage() {
	cv::Mat oFrame;
	moRightImages>>oFrame;
	return oFrame;
}
