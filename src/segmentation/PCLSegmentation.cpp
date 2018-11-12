#include "segmentation/PCLSegmentation.h"

#include <iostream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <SegmentationHelper.h>

using namespace std;
using namespace cv;

struct ClusterCenter {
	int iClusterID;
	cv::Vec3d oPosition;
	cv::Vec3d aDimension;
	cv::Vec2d aEccentricity;
};


vector<Cluster> ClusterAndAnalysePointCloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloudRaw);


PCLSegmentation::PCLSegmentation() {

}

PCLSegmentation::~PCLSegmentation() {

}


std::vector<Cluster> PCLSegmentation::Segment(const cv::Mat& rImage) {
	auto aPoints3D = Extract3DPoints(rImage);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints3D) {
		pCloud->push_back(oPoint);
	}

	vector<Cluster> aCluster = ClusterAndAnalysePointCloud(pCloud);

	return aCluster;
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


		cv::Vec3d oCVPosition = {oPosition.x, oPosition.y, oPosition.z};
		cv::Vec3d oCVDimension = {aDimension[0], aDimension[1], aDimension[2]};
		cv::Vec2d oCVEccentricity = {aEccentricity[0], aEccentricity[1]};
		aResult.push_back({oCVPosition, oCVDimension, oCVEccentricity});


		++iCloudCount;
	}

	return aResult;
}
