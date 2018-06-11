#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/opengl.hpp>

using namespace std;
using namespace cv;


int main() {

	Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_00000_c0.pgm", IMREAD_GRAYSCALE);
	//demosaicing(oImage, oImage, COLOR_BayerGR2RGB);
	oImage.convertTo(oImage, CV_8UC1, 16.0);

	Size oSize(oImage.cols, oImage.rows);

	Point2f p1(1000.0f, 300.0f);
	Point2f p2(1000.0f, 1300.0f);
	Point2f p3(250.0f, 1450.0f);
	Point2f p4(250.0f, 1700.0f);

	Point2f p5(500.0f, 0.0f);
	Point2f p6(500.0f, 1000.0f);
	Point2f p7(0.0f, 0.0f);
	Point2f p8(0.0f, 1000.0f);

	Point2f v1[] = {p1, p2, p3, p4};
	Point2f v2[] = {p5, p6, p7, p8};

	namedWindow("Original");


	for(;;) {

		imshow("Original", oImage);

		char c = (char)waitKey();
		if(c==27) 	break;
	}

	return 0;
}
