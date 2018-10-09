#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/passthrough.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/features/moment_of_inertia_estimation.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>


#include <nlohmann/json.hpp>

#include <FileGT.h>

using namespace std;
using namespace cv;
using json = nlohmann::json;

struct ClusterCenter {
	int iClusterID;
	pcl::PointXYZ oPosition;
};

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity);
std::vector<pcl::PointXYZ> Extract3DPointsOpenCV(const cv::Mat& rDisparity);

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, vector<double>& rDimension, vector<double>& rEccentricity,
		pcl::PointXYZ& rPosition, pcl::PointXYZ& rOBBPosition, pcl::PointXYZ& rOBBMin, pcl::PointXYZ& rOBBMax, Eigen::Matrix3f& rOBBRot);
vector<ClusterCenter> ClusterAndAnalysePointCloud(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloudRaw);


vector<ClusterCenter> aClusterCenter;
Mat oGlobalDisparityImage;
boost::shared_ptr<pcl::visualization::PCLVisualizer> pGlobalViewer;
pcl::PointXYZ oGlobalMousePosition(0.0, 0.0, 0.0);

string type2str(int type) {
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
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
	r += (chans + '0');

	return r;
}

void onMouse(int event, int iMouseX, int iMouseY, int, void*) {
	if (event != EVENT_LBUTTONDOWN)		return;

	cout << "Mouse Event at: " << iMouseX << " | " << iMouseY << endl;
	if(aClusterCenter.size()>0 && !oGlobalDisparityImage.empty()) {
		double cx = 1.260414892498634e+03;
		double cx2 = 1.284997418662109e+03;
		double cy = 5.260783408403682e+02;
		double Tx = 483.2905;
		double f = 2.500744557379985e+03;

		int i = iMouseY;
		int j = iMouseX;
		uchar cDisparity = oGlobalDisparityImage.at<uchar>((int)i, (int)j);
		if (cDisparity > 0) {
			double x = -((double)(j)-cx);
			double y = (double)(i)-cy;
			double z = f;
			double w = -(double)(cDisparity) / Tx + (cx - cx2) / Tx;
			x /= w;
			y /= w;
			z /= w;

			cout << i << " | " << j << " -> " << x << " | " << y << " | " << z << " - Disparity: " << (int)cDisparity << endl;

			oGlobalMousePosition.x = (float)x;
			oGlobalMousePosition.y = (float)y;
			oGlobalMousePosition.z = (float)z;
			pGlobalViewer->updateSphere(oGlobalMousePosition, 100.0, 1.0, 1.0, 1.0, "mouse_sphere");

			for(ClusterCenter& oCenter: aClusterCenter) {
				double dx = oCenter.oPosition.x-x;
				double dy = oCenter.oPosition.y-y;
				double dz = oCenter.oPosition.z-z;

				double dDistance = sqrt(dx*dx+dy*dy+dz*dz);
				cout<<"Distance to Cluster: "<<oCenter.iClusterID<<": "<<dDistance<<endl;
			}

		}
	}
}




int main() {

	FileGT oFileGT("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/gt.json");

	//Mat oImage = imread("E:/sample_images/img_549.png", IMREAD_GRAYSCALE);
	//Mat oImage = imread("/home/jung/2018EntwicklungStereoalgorithmus/sample_images/img_549.png", IMREAD_GRAYSCALE);



	pGlobalViewer = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("3D Viewer"));



	pGlobalViewer->removeAllPointClouds();
	

	cv::VideoCapture oImages("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/postprocess/img_%d.png");
	cv::VideoCapture oImagesColor("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm/preprocess/img_%d_c0.png");

	Mat oFrame;
	Mat oFrameColor;

	for (int iFrame = 0;; ++iFrame) {

		oImages>>oFrame;
		oImagesColor>>oFrameColor;

		if(oFrame.empty() || oFrameColor.empty()) 	break;

		cvtColor(oFrame, oFrame, CV_BGR2GRAY);

		oFrame.copyTo(oGlobalDisparityImage);

		auto aPoints3D = Extract3DPoints(oFrame);

		cout << "Extracted " << aPoints3D.size() << " Points" << endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
		for (auto& oPoint : aPoints3D) {
			pCloud->push_back(oPoint);
		}

		aClusterCenter = ClusterAndAnalysePointCloud(pGlobalViewer, pCloud);

		oFrame *= 3;
		applyColorMap(oFrame, oFrame, COLORMAP_JET);

		imshow("Disp", oFrame);
		imshow("Disp Color", oFrameColor);
		setMouseCallback("Disp", onMouse);

		pGlobalViewer->addSphere(oGlobalMousePosition, 10.0, "mouse_sphere");

		for(;;) {

			int iKeyCode = waitKey(100);

			pGlobalViewer->spinOnce(100, true);

			if (iKeyCode == 27)	break;
			if(iKeyCode==13) 	break;

			switch (iKeyCode) {
				case 49: {
					FrameGT oFrameGT;
					oFrameGT.miFrame = iFrame;
					oFrameGT.miLabel = 1;
					oFrameGT.mdX = oGlobalMousePosition.x;
					oFrameGT.mdY = oGlobalMousePosition.y;
					oFrameGT.mdZ = oGlobalMousePosition.z;

					oFileGT.AddFrameGT(oFrameGT);

					cout << "Setting Label 1" << endl;
					break;
				}
				case 50: {
					FrameGT oFrameGT;
					oFrameGT.miFrame = iFrame;
					oFrameGT.miLabel = 2;
					oFrameGT.mdX = oGlobalMousePosition.x;
					oFrameGT.mdY = oGlobalMousePosition.y;
					oFrameGT.mdZ = oGlobalMousePosition.z;

					oFileGT.AddFrameGT(oFrameGT);

					cout << "Setting Label 2" << endl;
					break;
				}
				case 51: {
					FrameGT oFrameGT;
					oFrameGT.miFrame = iFrame;
					oFrameGT.miLabel = 3;
					oFrameGT.mdX = oGlobalMousePosition.x;
					oFrameGT.mdY = oGlobalMousePosition.y;
					oFrameGT.mdZ = oGlobalMousePosition.z;

					oFileGT.AddFrameGT(oFrameGT);

					cout<<"Setting Label 3"<<endl;
					break;
				}
			}
		}

		pGlobalViewer->removeAllPointClouds();
		pGlobalViewer->removeAllShapes();
	}

	return 0;
}

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity) {
	assert(rDisparity.type() == CV_8UC1);

	std::vector<pcl::PointXYZ> aResult;


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
			}
		}
	}
	
	return aResult;
}

std::vector<pcl::PointXYZ> Extract3DPointsOpenCV(const cv::Mat& rDisparity) {
	assert(rDisparity.type() == CV_8UC1);

	std::vector<pcl::PointXYZ> aResult;

	//double QData[] = { 1.0, 0.0, 0.0, -690.7654724121094, 0.0, 1.0, 0.0, -531.5912857055664, 0.0, 0.0, 0.0, 1597.788948308794, 0.0, 0.0, 0.014676885509634826, -0.6669165936325461 };
	float QData[] = { 1.0, 0.0, 0.0, -1092.9328918457031, 0.0, 1.0, 0.0, -539.3284683227539, 0.0, 0.0, 0.0, 1650.4421002399172, 0.0, 0.0, 0.0020691487390760484, -0.13932939622620283 };

	Mat Q = cv::Mat(4, 4, CV_32F, QData);

	Mat o3DImage;

	reprojectImageTo3D(rDisparity, o3DImage, Q);

	for (int i = 0; i < rDisparity.rows; ++i) {
		for (int j = 0; j < rDisparity.cols; ++j) {
			uchar cDisparity = rDisparity.at<uchar>(i, j);
			if (cDisparity > 0) {
				cv::Vec3f Point3D = o3DImage.at<cv::Vec3f>(i, j);

				pcl::PointXYZ oPoint;
				oPoint.x = -Point3D[0];
				oPoint.y = -Point3D[1];
				oPoint.z = Point3D[2];

				aResult.push_back(oPoint);
			}
		}
	}

	return aResult;
}

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, vector<double>& rDimension, vector<double>& rEccentricity,
		pcl::PointXYZ& rPosition, pcl::PointXYZ& rOBBPosition, pcl::PointXYZ& rOBBMin, pcl::PointXYZ& rOBBMax, Eigen::Matrix3f& rOBBRot)
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

	rDimension.resize(3);
	rDimension[0] = OBBMax.x - OBBMin.x;
	rDimension[1] = OBBMax.y - OBBMin.y;
	rDimension[2] = OBBMax.z - OBBMin.z;

	//std::sort(rDimension.begin(), rDimension.end(), std::greater<int>());

	rEccentricity.resize(2);
	rEccentricity[0] = rDimension[1] / rDimension[0];
	rEccentricity[1] = rDimension[2] / rDimension[0];

	rPosition.x = oCenterOfMass.x();
	rPosition.y = oCenterOfMass.y();
	rPosition.z = oCenterOfMass.z();

	rOBBPosition = OBBPosition;
	rOBBMin = OBBMin;
	rOBBMax = OBBMax;
	rOBBRot = OBBRot;
}


vector<ClusterCenter> ClusterAndAnalysePointCloud(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloudRaw) {
	vector<ClusterCenter> aResult;

	pcl::PointCloud<pcl::PointXYZ>::Ptr oCloud(new pcl::PointCloud<pcl::PointXYZ>());

	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud (pCloudRaw);
	sor.setMeanK (200);
	sor.setStddevMulThresh (0.2);
	sor.filter (*oCloud);

	cout<<"Cloud Size after Filtering: "<<oCloud->width<<endl;

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(oCloud);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (700.0);
	ec.setMinClusterSize (1500);
	ec.setMaxClusterSize (50000);
	ec.setSearchMethod (tree);
	ec.setInputCloud (oCloud);
	ec.extract (cluster_indices);

	cout<<"Extracted "<<cluster_indices.size()<<" Clusters"<<endl;

	int iCloudCount = 1;
	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin (); it != cluster_indices.end (); ++it) {
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZ>);
		for (std::vector<int>::const_iterator pit = it->indices.begin (); pit != it->indices.end (); ++pit) {
			cloud_cluster->points.push_back (oCloud->points[*pit]);
		}
		cloud_cluster->width = cloud_cluster->points.size ();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		cout<<"Cluster "<<iCloudCount<<" size: "<<cloud_cluster->width<<endl;

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_cluster, 256-35*iCloudCount, 35*iCloudCount, 0);

		pViewer->addPointCloud(cloud_cluster, single_color, "cloud_"+std::to_string(iCloudCount));

		vector<double> aDimension;
		vector<double> aEccentricity;
		pcl::PointXYZ oPosition;
		pcl::PointXYZ oOBBPosition;
		pcl::PointXYZ oOBBMin;
		pcl::PointXYZ oOBBMax;
		Eigen::Matrix3f oOBBRot;

		AnalysePointcloud(cloud_cluster, aDimension, aEccentricity, oPosition, oOBBPosition, oOBBMin, oOBBMax, oOBBRot);

		Eigen::Vector3f position (oOBBPosition.x, oOBBPosition.y, oOBBPosition.z);
		Eigen::Quaternionf quat (oOBBRot);
		//pViewer->addCube(position, quat, oOBBMax.x - oOBBMin.x, oOBBMax.y - oOBBMin.y, oOBBMax.z - oOBBMin.z, "OBB_"+std::to_string(iCloudCount));

		pViewer->addSphere(oPosition, 100.0, "sphere_"+std::to_string(iCloudCount));
		cout<<"Sphere: "<<oPosition<<endl;

		cout << "Position: " << oPosition.x << " | " << oPosition.y << " | " << oPosition.z << endl;
		cout << "OBBPosition: " << oOBBPosition.x << " | " << oOBBPosition.y << " | " << oOBBPosition.z << endl;
		cout << "Dimension: " << aDimension[0] << " | " << aDimension[1] << " | " << aDimension[2] << endl;
		cout << "Eccentricity: " << aEccentricity[0] << " | " << aEccentricity[1] << endl;
		cout<<"---------------------------------------------------------------------"<<endl;

		aResult.push_back({iCloudCount, oPosition});

		++iCloudCount;
	}

	return aResult;
}


