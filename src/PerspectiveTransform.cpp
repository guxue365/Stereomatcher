#include <iostream>

#include <Windows.h>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/opengl.hpp>

#include <gl/GL.h>

using namespace std;
using namespace cv;
using namespace cv::ogl;


std::vector<cv::Vec3d> Extract3DPoints(const cv::Mat& oDisparity);

vector<cv::Vec3d> aPoints;


void on_opengl(void* param)
{
	glLoadIdentity();
	glTranslated(-0.5, -0.5, -2.0);
	glRotatef(0, 1, 0, 0);
	glRotatef(0, 0, 1, 0);
	glRotatef(45, 0, 0, 1);

	glColor3ub(20, 100, 42);
	glBegin(GL_POINTS);
	/*for (int j = 0; j < 4; ++j) {
		glVertex3d(0.2 * coords[i][j][0], 0.2 * coords[i][j][1], 0.2 * coords[i][j][2]);
	}*/
	cout << "Drawing points" << endl;
	for (int i = 0; i < aPoints.size(); ++i) {
		glVertex3d(aPoints[i][0], aPoints[i][1], aPoints[i][2]);
	}
	cout << "finished" << endl;
	glEnd();
}

int main() {
	cv::Vec3d a;
	//Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_00000_c0.pgm", IMREAD_GRAYSCALE);
	Mat oImage = imread("E:/data_disparity/with_background/img_19.png");
	//Mat oImage = Mat::zeros(100, 100, CV_8U);

	aPoints = Extract3DPoints(oImage);

	cout << "Num Pixels: " << (size_t)(oImage.cols*oImage.rows) << endl;
	cout << "Found Points: " << aPoints.size() << endl;

	namedWindow("Original", WINDOW_OPENGL+WINDOW_AUTOSIZE);

	setOpenGlDrawCallback("Original", on_opengl);

	for(;;) {

		imshow("Image", oImage);
		updateWindow("Original");

		char c = (char)waitKey();
		if(c==27) 	break;
	}

	return 0;
}

std::vector<cv::Vec3d> Extract3DPoints(const cv::Mat& oDisparity) {
	std::vector<cv::Vec3d> aResult((size_t)(oDisparity.rows*oDisparity.cols));
	size_t iNumPoints = 0;

	double dMinDisp;
	double dMaxDisp;
	minMaxLoc(oDisparity, &dMinDisp, &dMaxDisp);

	for (size_t i = 0; i < (size_t)oDisparity.rows; ++i) {
		for (size_t j = 0; j < (size_t)oDisparity.cols; ++j) {
			uchar cDisparity = oDisparity.at<uchar>((int)i, (int)j);
			if (cDisparity > 0) {
				double x = (double)(j) / (double)(oDisparity.rows);
				double y = 1.0-(double)(i) / (double)(oDisparity.rows);
				//double z = (double)(cDisparity)/(double)(dMaxDisp);
				double z = 1.0;
				aResult[iNumPoints] = cv::Vec3d(y, x, z);
				iNumPoints++;
			}
		}
	}
	aResult.resize(iNumPoints);
	return aResult;
}