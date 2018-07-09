#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/viz.hpp>

using namespace std;
using namespace cv;


std::vector<cv::Vec3d> Extract3DPoints(const cv::Mat& oDisparity);

vector<cv::Vec3d> aPoints;

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
	cv::Vec3d a;
	//Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/2018-05-16_07-35-15_ZA1/RenamedImages/img_00000_c0.pgm", IMREAD_GRAYSCALE);
	Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/disparity/img_0.png", IMREAD_GRAYSCALE);
	//Mat oImage = Mat::zeros(100, 100, CV_8U);

	cout<<"Image Format: "<<type2str(oImage.type())<<endl;

	aPoints = Extract3DPoints(oImage);
	Mat oPointCloud(aPoints.size(), 1, CV_64FC3);
	for(int i=0; i<aPoints.size(); ++i) {
		oPointCloud.at<Vec3d>(i, 0) = aPoints[i];
	}

	cout << "Num Pixels: " << (size_t)(oImage.cols*oImage.rows) << endl;
	cout << "Found Points: " << aPoints.size() << endl;


	viz::Viz3d WindowVIZ("Viz Demo");

	WindowVIZ.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	double cx = 1.260414892498634e+03;
	double cx2 = 1.284997418662109e+03;
	double cy = 5.260783408403682e+02;
	double Tx = 483.2905;
	double f = 2.500744557379985e+03;
	Mat Q = Mat::eye(4, 4, CV_64F);
	Q.at<double>(0, 3) = -cx;
	Q.at<double>(1, 3) = -cy;
	Q.at<double>(2, 3) = f;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(3, 2) = -1.0/Tx;
	Q.at<double>(3, 3) = (cx-cx2)/Tx;

	cout<<Q<<endl;

	cv::Mat xyz;
	reprojectImageTo3D(oImage, xyz, Q, true);

	viz::WCloud Cloud(xyz);
	WindowVIZ.showWidget("Cloud", Cloud);

	for(;;) {

		imshow("Original", oImage);

		char c = (char)waitKey(10);
		if(c==27) 	break;

		WindowVIZ.spinOnce(1, true);
	}

	return 0;
}

std::vector<cv::Vec3d> Extract3DPoints(const cv::Mat& oDisparity) {
	std::vector<cv::Vec3d> aResult((size_t)(oDisparity.rows*oDisparity.cols));
	size_t iNumPoints = 0;


	double B = 483.2905;
	double f = 2.500744557379985e+03;

	for (size_t i = 0; i < (size_t)oDisparity.rows; ++i) {
		for (size_t j = 0; j < (size_t)oDisparity.cols; ++j) {
			uchar cDisparity = oDisparity.at<uchar>((int)i, (int)j);
			if (cDisparity > 0) {
				double x = (double)(j) / (double)(oDisparity.rows);
				double y = 1.0-(double)(i) / (double)(oDisparity.rows);
				double z = B*f/(double)(cDisparity);
				aResult[iNumPoints] = cv::Vec3d(y, x, z);
				iNumPoints++;
			}
		}
	}
	aResult.resize(iNumPoints);
	return aResult;
}
