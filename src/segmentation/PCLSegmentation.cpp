#include "segmentation/PCLSegmentation.h"

#include <iostream>
#include <vector>

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

using namespace std;
using namespace cv;

struct ClusterCenter {
	int iClusterID;
	cv::Vec3d oPosition;
	cv::Vec3d aDimension;
	cv::Vec2d aEccentricity;
};

std::vector<pcl::PointXYZ> Extract3DPoints(const cv::Mat& rDisparity);
void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, vector<double>& rDimension, vector<double>& rEccentricity,
		pcl::PointXYZ& rPosition, pcl::PointXYZ& rOBBPosition, pcl::PointXYZ& rOBBMin, pcl::PointXYZ& rOBBMax, Eigen::Matrix3f& rOBBRot);
vector<Cluster> ClusterAndAnalysePointCloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloudRaw);


PCLSegmentation::PCLSegmentation() {

}

PCLSegmentation::~PCLSegmentation() {

}

/*cv::Mat PCLSegmentation::Segment(const cv::Mat& rImage) {

	auto aPoints3D = Extract3DPoints(rImage);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints3D) {
		pCloud->push_back(oPoint);
	}

	vector<ClusterCenter> aClusterCenter = ClusterAndAnalysePointCloud(pCloud);

	cout<<"Found "<<aClusterCenter.size()<<" Clusters"<<endl;

	return rImage;
}*/

std::vector<Cluster> PCLSegmentation::Segment(const cv::Mat& rImage) {
	auto aPoints3D = Extract3DPoints(rImage);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints3D) {
		pCloud->push_back(oPoint);
	}

	vector<Cluster> aCluster = ClusterAndAnalysePointCloud(pCloud);

	return aCluster;
}

std::vector<pcl::PointXYZ> PCLSegmentation::Extract3DPoints(const cv::Mat& rDisparity) {
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


vector<Cluster> ClusterAndAnalysePointCloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloudRaw) {
	vector<Cluster> aResult;

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

		/*cout << "Position: " << oPosition.x << " | " << oPosition.y << " | " << oPosition.z << endl;
		cout << "OBBPosition: " << oOBBPosition.x << " | " << oOBBPosition.y << " | " << oOBBPosition.z << endl;
		cout << "Dimension: " << aDimension[0] << " | " << aDimension[1] << " | " << aDimension[2] << endl;
		cout << "Eccentricity: " << aEccentricity[0] << " | " << aEccentricity[1] << endl;
		cout<<"---------------------------------------------------------------------"<<endl;*/

		cv::Vec3d oCVPosition = {oPosition.x, oPosition.y, oPosition.z};
		cv::Vec3d oCVDimension = {aDimension[0], aDimension[1], aDimension[2]};
		cv::Vec2d oCVEccentricity = {aEccentricity[0], aEccentricity[1]};
		aResult.push_back({oCVPosition, oCVDimension, oCVEccentricity});


		++iCloudCount;
	}

	return aResult;
}
