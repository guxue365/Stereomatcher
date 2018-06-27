#include <imageloader/CustomImageloader.h>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;

CustomImageloader::CustomImageloader() {

}

CustomImageloader::~CustomImageloader() {

}

bool CustomImageloader::Init(const std::string& sFolder) {
	//moLeftImages.open(sFolder+"RectifiedLeft/img_%1d.png");
	//moRightImages.open(sFolder+"RectifiedRight/img_%1d.png");
	moLeftImages.open("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/Rec01_ZA1/RectifiedLeft/img_%1d.png");
	moRightImages.open("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/Rec01_ZA1/RectifiedRight/img_%1d.png");
	if(!moLeftImages.isOpened()) {
		cout<<"Failed open left"<<endl;
	}
	if(!moRightImages.isOpened()) {
		cout<<"Failed open right"<<endl;
	}

	return (moLeftImages.isOpened() && moRightImages.isOpened());
}

cv::Mat CustomImageloader::getNextLeftImage() {
	cv::Mat oFrame;
	moLeftImages>>oFrame;
	return oFrame;
}

cv::Mat CustomImageloader::getNextRightImage() {
	cv::Mat oFrame;
	moRightImages>>oFrame;
	return oFrame;
}
