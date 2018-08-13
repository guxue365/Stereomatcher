#include <iostream>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/visualization/cloud_viewer.h>

std::vector<pcl::PointXYZRGB> Extract3DPoints(const cv::Mat& rDisparity, const cv::Mat& rColorImage);


using namespace std;
using namespace cv;

int main() {

	//Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/disparity/img_0.png", IMREAD_GRAYSCALE);
	Mat oImage = imread("disparity.png");
	Mat oColor = imread("color.png");

	imshow("Image", oImage);
	imshow("Color", oColor);

	auto aPoints = Extract3DPoints(oImage, oColor);

	cout<<"Extracted "<<aPoints.size()<<" Points"<<endl;

	//Punktwolke erstellen
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	/*cloud->push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	cloud->push_back(pcl::PointXYZ(1.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(2.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 1.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 2.0, 0.0));*/

	for(auto& rPoint: aPoints) {
		cloud->push_back(rPoint);
		//cout<<"Inserting Point: "<<rPoint.val[0]<<" | "<<rPoint.val[1]<<" | "<<rPoint.val[2]<<endl;
	}

	//Punktwolke laden (siehe oben) oder neue erstellen
	//CloudViewer zur Visualisierung erstellen
	pcl::visualization::PCLVisualizer viewer("Simple Cloud Viewer");

	//Aufruf zur Darstellung der Punktwolke
	viewer.addPointCloud(cloud);
	while (!viewer.wasStopped()) {
		//m�gliche Prozessierungsanweisungen m�glich in einem extra Thread

		waitKey(0);
	}
	return 0;
}

std::vector<pcl::PointXYZRGB> Extract3DPoints(const cv::Mat& rDisparity, const cv::Mat& rColorImage) {
	std::vector<pcl::PointXYZRGB> aResult((size_t)(rDisparity.rows*rDisparity.cols));
	size_t iNumPoints = 0;


	double cx = 1.260414892498634e+03;
	double cx2 = 1.284997418662109e+03;
	double cy = 5.260783408403682e+02;
	double Tx = 483.2905;
	double f = 2.500744557379985e+03;

	for (size_t i = 0; i < (size_t)rDisparity.rows; ++i) {
		for (size_t j = 0; j < (size_t)rDisparity.cols; ++j) {
			uchar cDisparity = rDisparity.at<uchar>((int)i, (int)j);
			if (cDisparity > 0) {
				double x = (double)(j)-cx;
				double y = 1.0-(double)(i)-cy;
				double z = f;
				double w = -(double)(cDisparity)/Tx+(cx-cx2)/Tx;
				x/=w;
				y/=w;
				z/=w;

				cv::Vec3b oColor = rColorImage.at<cv::Vec3b>(i, j);

				pcl::PointXYZRGB oPoint;
				oPoint.x = (float)x;
				oPoint.y = (float)y;
				oPoint.z = (float)z;
				oPoint.r = oColor[2];
				oPoint.g = oColor[1];
				oPoint.b = oColor[0];

				aResult[iNumPoints] = oPoint;
				iNumPoints++;
			}
		}
	}
	aResult.resize(iNumPoints);
	return aResult;
}
