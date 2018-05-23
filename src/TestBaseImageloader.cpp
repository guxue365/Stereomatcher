#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace std;
using namespace cv;

int main() {
	cv::VideoCapture oImageStream("/home/herrmann/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_%05d_c0.pgm");

	Mat oFrame;
	for(size_t i=0;; ++i) {
		oImageStream>>oFrame;
		oFrame.convertTo(oFrame, CV_8U, 255.0 / 4096.0);
		imshow("frame", oFrame);

		char c = (char)waitKey(50);
		if(c==27) 	break;
	}

	return 0;
}
