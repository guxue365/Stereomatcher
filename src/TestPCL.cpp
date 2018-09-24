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
	//VideoCapture oStreamDisparity("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/disparity/img_%1d.png");
	//VideoCapture oStreamColor("/home/jung/2018EntwicklungStereoalgorithmus/data/Datensatz/Rec01_ZA1/RectifiedLeft/img_%1d.png");
	VideoCapture oStreamDisparity("E:/sample_images/img_%1d.png");
	VideoCapture oStreamColor("E:/sample_images/img_%1d.png");

	pcl::visualization::PCLVisualizer oViewer ("Simple Cloud Viewer");

	for(int iFrame=0; ; ++iFrame) {
		Mat oFrameDisparity;
		Mat oFrameColor;

		oStreamDisparity>>oFrameDisparity;
		oStreamColor>>oFrameColor;
		

		if(oFrameDisparity.empty() || oFrameColor.empty()) 	break;



		//cvtColor(oFrameDisparity, oFrameDisparity, CV_BGR2GRAY);
		//cvtColor(oFrameColor, oFrameColor, CV_BGR2GRAY);

		oFrameColor *= 3;
		applyColorMap(oFrameColor, oFrameColor, COLORMAP_JET);

		imshow("disp", oFrameDisparity);
		imshow("color", oFrameColor);

		auto aPoints = Extract3DPoints(oFrameDisparity, oFrameColor);

		pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZRGB>());
		for(auto& rPoint: aPoints) {
			pCloud->push_back(rPoint);
		}

		if(iFrame==0) {
			oViewer.addPointCloud(pCloud);
		} else {
			oViewer.updatePointCloud(pCloud);
		}

		oViewer.spinOnce(100, true);


		int c = (char)waitKey(10);
		if(c==27) 	break;
	}

	/*auto aPoints = Extract3DPoints(oImage, oColor);

	cout<<"Extracted "<<aPoints.size()<<" Points"<<endl;

	//Punktwolke erstellen
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
	/*pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	cloud->push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	cloud->push_back(pcl::PointXYZ(1.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(2.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 1.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 2.0, 0.0));*/

	/*for(auto& rPoint: aPoints) {
		cloud->push_back(rPoint);
		//cout<<"Inserting Point: "<<rPoint.val[0]<<" | "<<rPoint.val[1]<<" | "<<rPoint.val[2]<<endl;
	}

	//Punktwolke laden (siehe oben) oder neue erstellen
	//CloudViewer zur Visualisierung erstellen
	pcl::visualization::PCLVisualizer viewer ("Simple Cloud Viewer");
	/*viewer.initCameraParameters();
	int v1(0);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, v1);

	pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
	viewer.addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud1", v1);

	viewer.setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud1");
	viewer.addCoordinateSystem (1.0);*/

	/*viewer.addPointCloud(cloud);


	//Aufruf zur Darstellung der Punktwolke
	//viewer.showCloud(cloud);
	while (!viewer.wasStopped()) {
		//m�gliche Prozessierungsanweisungen m�glich in einem extra Thread
		viewer.spinOnce(100, true);
		waitKey(1);
	}*/
	return 0;
}

std::vector<pcl::PointXYZRGB> Extract3DPoints(const cv::Mat& rDisparity, const cv::Mat& rColorImage) {
	assert(rDisparity.type()==CV_8UC1);
	assert(rColorImage.type()==CV_8UC3);

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
				double x = -((double)(j)-cx);
				double y = (double)(i)-cy;
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
