#include <iostream>
#include <vector>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/moment_of_inertia_estimation.h>



#include <nlohmann/json.hpp>

#include <FileGT.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity);

void onMouse(int event, int x, int y, int, void*) {
	if (event != EVENT_LBUTTONDOWN)		return;

	cout << "Mouse Event at: " << x << " | " << y << endl;
}



int main() {

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());

	cloud->push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(2.0, 0.0, 0.0));
	cloud->push_back(pcl::PointXYZ(2.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 0.0, 2.0));
	cloud->push_back(pcl::PointXYZ(0.0, 2.0, 2.0));
	cloud->push_back(pcl::PointXYZ(2.0, 0.0, 2.0));
	cloud->push_back(pcl::PointXYZ(2.0, 2.0, 2.0));

	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> moe;
	moe.setInputCloud(cloud);
	moe.compute();

	vector<float> eccentricity;
	if (!moe.getEccentricity(eccentricity)) {
		cout << "Error: getEccentricity" << endl;
	}

	Eigen::Vector3f com;
	if (!moe.getMassCenter(com)) {
		cout << "Error: masscenter" << endl;
	}

	vector<float> moi;
	if (!moe.getMomentOfInertia(moi)) {
		cout << "Error: moment of inertia" << endl;
	}

	cout << "Eccentriyti size: " << eccentricity.size() << endl;
	cout << "mass center: " <<endl<< com << endl;
	cout << "moment of inertia size: " << moi.size() << endl;

	for (int i = 0; i < 50; ++i) {
		cout << "ecc | moi: " << eccentricity[i] << " | " << moi[i] << endl;
	}

	return 0;

	FileGT file("gt.json");

	Mat oImage = imread("E:/sample_images/img_0.png", IMREAD_GRAYSCALE);
	auto aPoints3D = Extract3DPoints(oImage);

	cout << "Extracted " << aPoints3D.size() << " Points" << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr oCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints3D) {
		oCloud->push_back(oPoint);
	}


	pcl::search::Search<pcl::PointXYZ>::Ptr oTree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
	pcl::PointCloud <pcl::Normal>::Ptr aNormals(new pcl::PointCloud <pcl::Normal>);
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
	normal_estimator.setSearchMethod(oTree);
	normal_estimator.setInputCloud(oCloud);
	normal_estimator.setKSearch(50);
	normal_estimator.compute(*aNormals);

	pcl::RegionGrowing<pcl::PointXYZ, pcl::Normal> oRegionGrowing;
	oRegionGrowing.setMinClusterSize(200);
	oRegionGrowing.setMaxClusterSize(1000000);
	oRegionGrowing.setSearchMethod(oTree);
	oRegionGrowing.setNumberOfNeighbours(10);
	oRegionGrowing.setInputCloud(oCloud);
	//reg.setIndices (indices);
	oRegionGrowing.setInputNormals(aNormals);
	oRegionGrowing.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
	oRegionGrowing.setCurvatureThreshold(1.0);

	std::vector <pcl::PointIndices> aClusters;
	oRegionGrowing.extract(aClusters);

	cout << "Number of extracted Clusters: " << aClusters.size() << endl;

	pcl::PointCloud <pcl::PointXYZRGB>::Ptr colored_cloud = oRegionGrowing.getColoredCloud();
	pcl::visualization::PCLVisualizer oViewer("Simple Cloud Viewer");
	oViewer.addPointCloud(colored_cloud);
	while (!oViewer.wasStopped())
	{
		oViewer.spinOnce(100, true);
	}

	oImage *= 3;
	applyColorMap(oImage, oImage, COLORMAP_JET);

	imshow("Disp", oImage);

	setMouseCallback("Disp", onMouse);

	

	bool bRunning = true;
	for (int iFrame = 0; bRunning; ++iFrame) {

		int iKeyCode = waitKey(0);

		if (iKeyCode == 27)	break;

		switch (iKeyCode) {
		case 49: {
			cout << "Setting Label 1" << endl;
			break;
		}
		case 50: {
			cout << "Setting Label 2" << endl;
			break;
		}
		}
	}

	return 0;
}

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity) {
	assert(rDisparity.type() == CV_8UC1);

	std::vector<pcl::PointXYZ> aResult;
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
				double w = -(double)(cDisparity) / Tx + (cx - cx2) / Tx;
				x /= w;
				y /= w;
				z /= w;

				pcl::PointXYZ oPoint;
				oPoint.x = (float)x;
				oPoint.y = (float)y;
				oPoint.z = (float)z;

				aResult.push_back(oPoint);
				iNumPoints++;
			}
		}
	}
	
	return aResult;
}