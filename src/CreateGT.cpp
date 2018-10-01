#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

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

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, vector<double>& rDimension, vector<double>& rEccentricity, vector<double>& rPosition);

int main() {

	

	FileGT file("gt.json");

	Mat oImage = imread("E:/sample_images/img_0.png", IMREAD_GRAYSCALE);
	auto aPoints3D = Extract3DPoints(oImage);

	cout << "Extracted " << aPoints3D.size() << " Points" << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr oCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints3D) {
		oCloud->push_back(oPoint);
	}


	/*pcl::search::Search<pcl::PointXYZ>::Ptr oTree = boost::shared_ptr<pcl::search::Search<pcl::PointXYZ> >(new pcl::search::KdTree<pcl::PointXYZ>);
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
	}*/

	vector<double> aDimension;
	vector<double> aEccentricity;
	vector<double> oPosition;
	AnalysePointcloud(oCloud, aDimension, aEccentricity, oPosition);

	cout << "Position: " << oPosition[0] << " | " << oPosition[1] << " | " << oPosition[2] << endl;
	cout << "Dimension: " << aDimension[0] << " | " << aDimension[1] << " | " << aDimension[2] << endl;
	cout << "Eccentricity: " << aEccentricity[0] << " | " << aEccentricity[1] << endl;

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

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, vector<double>& rDimension, vector<double>& rEccentricity, vector<double>& rPosition)
{
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> MoIEstimation;
	MoIEstimation.setInputCloud(pCloud);
	MoIEstimation.compute();

	Eigen::Vector3f oCenterOfMass;
	if (!MoIEstimation.getMassCenter(oCenterOfMass)) {
		cout << "Error: masscenter" << endl;
	}

	pcl::PointXYZ OBBMin;
	pcl::PointXYZ OBBMax;
	pcl::PointXYZ OBBPosition;
	Eigen::Matrix3f OBBRot;
	if (!MoIEstimation.getOBB(OBBMin, OBBMax, OBBPosition, OBBRot)) {
		cout << "Error: getOBB" << endl;
	}

	cout << "OBB min: " << OBBMin << endl;
	cout << "OBB max: " << OBBMax << endl;
	cout << "OBB pos: " << OBBPosition << endl;

	cout << "Dimension: " << OBBMax.x - OBBMin.x << endl;
	cout << "Dimension: " << OBBMax.z - OBBMin.y << endl;
	cout << "Dimension: " << OBBMax.x - OBBMin.z << endl;

	cout << "mass center: " << endl << oCenterOfMass << endl;

	rDimension.resize(3);
	rDimension[0] = OBBMax.x - OBBMin.x;
	rDimension[1] = OBBMax.y - OBBMin.y;
	rDimension[2] = OBBMax.z - OBBMin.z;

	std::sort(rDimension.begin(), rDimension.end(), std::greater<int>());

	rEccentricity.resize(2);
	rEccentricity[0] = rDimension[1] / rDimension[0];
	rEccentricity[1] = rDimension[2] / rDimension[0];

	rPosition.resize(3);
	rPosition[0] = oCenterOfMass.x();
	rPosition[1] = oCenterOfMass.y();
	rPosition[2] = oCenterOfMass.z();
}