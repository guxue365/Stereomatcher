#include <iostream>

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
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <segmentation/RegionGrowing.h>
#include <segmentation/DBSCAN.h>
#include <segmentation/PCLSegmentation.h>

#include <SegmentationHelper.h>

using namespace std;
using namespace cv;

void ClusterPCL(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame, int iStartName = 0);
void ClusterRegionGrowing(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame);
void ClusterTwoStep(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame);

std::vector<cv::Rect2i> ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions);
std::vector<cv::Rect2i> MergeRegions(const std::vector<cv::Rect2i>& aRegions);

int main() {

	cv::VideoCapture oCapture("/home/jung/2018EntwicklungStereoalgorithmus/Stereomatcher_eclipse/result_bm_scene3/postprocess/img_%d.png");
	if(!oCapture.isOpened()) {
		cout<<"Error opening video"<<endl;
		return -1;
	}

	Mat oFrame;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> pCloudViewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

	for(int iFrame = 0;; ++iFrame) {
		oCapture>>oFrame;

		if(oFrame.empty()) 	break;

		cvtColor(oFrame, oFrame, CV_BGR2GRAY);



		ClusterTwoStep(pCloudViewer, oFrame);

		oFrame *= 3;
		applyColorMap(oFrame, oFrame, COLORMAP_JET);

		imshow("Disp", oFrame);

		for(;;) {
			int iKeyCode = waitKey(100);

			pCloudViewer->spinOnce(100, true);

			if (iKeyCode == 27)	break;
			if(iKeyCode==13) 	break;
		}

		pCloudViewer->removeAllPointClouds();
		pCloudViewer->removeAllShapes();
	}

	return 0;
}

void ClusterPCL(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame, int iStartName) {
	auto aPoints = Extract3DPoints(rFrame);

	pcl::PointCloud<pcl::PointXYZ>::Ptr pCloudRaw(new pcl::PointCloud<pcl::PointXYZ>());
	for (auto& oPoint : aPoints) {
		pCloudRaw->push_back(oPoint);
	}

	cout<<"Cloud Size before Filtering: "<<aPoints.size()<<endl;

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
	ec.setMinClusterSize (1200);
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

		pViewer->addPointCloud(cloud_cluster, single_color, "cloud_"+std::to_string(iCloudCount+iStartName));

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
		pViewer->addCube(position, quat, oOBBMax.x - oOBBMin.x, oOBBMax.y - oOBBMin.y, oOBBMax.z - oOBBMin.z, "OBB_"+std::to_string(iCloudCount+iStartName));

		pViewer->addSphere(oPosition, 100.0, "sphere_"+std::to_string(iCloudCount+iStartName));

		cout << "Position: " << oPosition.x << ", " << oPosition.y << ", " << oPosition.z << endl;
		cout << "OBBPosition: " << oOBBPosition.x << " | " << oOBBPosition.y << " | " << oOBBPosition.z << endl;
		cout << "Dimension: " << aDimension[0] << " | " << aDimension[1] << " | " << aDimension[2] << endl;
		cout << "Eccentricity: " << aEccentricity[0] << " | " << aEccentricity[1] << endl;
		cout<<"---------------------------------------------------------------------"<<endl;

		++iCloudCount;
	}
}

void ClusterRegionGrowing(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame) {
	RegionGrowing oRegionGrowing;
	uchar rNumLabel;
	cv::Mat oRegion = oRegionGrowing.GrowRegion(rFrame, rNumLabel);

	int iNumLabel = (int)rNumLabel;

	cout<<"Num Labels after growing: "<<iNumLabel<<endl<<endl;

	for(int iLabel = 1; iLabel<=iNumLabel; ++iLabel) {
		cout<<"Cloud "<<iLabel<<": "<<endl;
		cv::Mat oDummyMat = cv::Mat::zeros(rFrame.rows, rFrame.cols, CV_8U);

		for(int i=0; i<oRegion.rows; ++i) {
			for(int j=0; j<oRegion.cols; ++j) {
				if((int)(oRegion.at<uchar>(i, j))==iLabel) {
					oDummyMat.at<uchar>(i, j) = rFrame.at<uchar>(i, j);
				}
			}
		}

		auto aPoints = Extract3DPoints(oDummyMat);
		cout<<"Extracted "<<aPoints.size()<<" Points"<<endl;

		pcl::PointCloud<pcl::PointXYZ>::Ptr pCloud(new pcl::PointCloud<pcl::PointXYZ>());
		for (auto& oPoint : aPoints) {
			pCloud->push_back(oPoint);
		}

		pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(pCloud, 256-35*iLabel, 35*iLabel, 0);

		pViewer->addPointCloud(pCloud, single_color, "cloud_"+std::to_string(iLabel));
		cout<<"Adding Cloud "<<"cloud_"+std::to_string(iLabel)<<endl;

		vector<double> aDimension;
		vector<double> aEccentricity;
		pcl::PointXYZ oPosition;
		pcl::PointXYZ oOBBPosition;
		pcl::PointXYZ oOBBMin;
		pcl::PointXYZ oOBBMax;
		Eigen::Matrix3f oOBBRot;

		AnalysePointcloud(pCloud, aDimension, aEccentricity, oPosition, oOBBPosition, oOBBMin, oOBBMax, oOBBRot);

		Eigen::Vector3f position (oOBBPosition.x, oOBBPosition.y, oOBBPosition.z);
		Eigen::Quaternionf quat (oOBBRot);
		pViewer->addCube(position, quat, oOBBMax.x - oOBBMin.x, oOBBMax.y - oOBBMin.y, oOBBMax.z - oOBBMin.z, "OBB_"+std::to_string(iLabel));

		pViewer->addSphere(oPosition, 100.0, "sphere_"+std::to_string(iLabel));

		cout << "Position: " << oPosition.x << ", " << oPosition.y << ", " << oPosition.z << endl;
		cout << "OBBPosition: " << oOBBPosition.x << " | " << oOBBPosition.y << " | " << oOBBPosition.z << endl;
		cout << "Dimension: " << aDimension[0] << " | " << aDimension[1] << " | " << aDimension[2] << endl;
		cout << "Eccentricity: " << aEccentricity[0] << " | " << aEccentricity[1] << endl;
		cout<<"---------------------------------------------------------------------"<<endl;
	}
}

void ClusterTwoStep(boost::shared_ptr<pcl::visualization::PCLVisualizer> pViewer, const cv::Mat& rFrame) {
	RegionGrowing oRegionGrowing;
	PCLSegmentation oPCLSegmentation;

	Mat oImageCoarse;

	double dScaling = 0.025;
	resize(rFrame, oImageCoarse, Size(), dScaling, dScaling, INTER_LINEAR);

	uchar cNumLabel;
	Mat oRegionCoarse = oRegionGrowing.GrowRegion(oImageCoarse, cNumLabel);
	int iNumLabel = (int)cNumLabel;

	vector<Rect2i> aRegionsRaw = ExtractRegions(oRegionCoarse, iNumLabel);

	vector<Rect2i> aRegions = MergeRegions(aRegionsRaw);

	for (int i = 0; i < aRegionsRaw.size(); ++i) {
		cout << "Region " << i << ": " << endl;
		cout << aRegionsRaw[i] << endl;
	}

	cout << "Merged Regions: " << endl;
	for (int i = 0; i < aRegions.size(); ++i) {
		cout << "Region " << i << ": " << endl;
		cout << aRegions[i] << endl;
	}


	for (size_t iRegion = 0; iRegion < aRegions.size(); ++iRegion) {
			//aRegions.at(i).x = (int)((double)(aRegions.at(i).x-1) / (dScaling));
			aRegions[iRegion].x = (int)((double)(aRegions[iRegion].x-1) / (dScaling));
			aRegions[iRegion].y = (int)((double)(aRegions[iRegion].y-1) / (dScaling));
			aRegions[iRegion].width = (int)((double)(aRegions[iRegion].width+2) / (dScaling));
			aRegions[iRegion].height = (int)((double)(aRegions[iRegion].height+2) / (dScaling));
			if (aRegions[iRegion].x < 0)		aRegions[iRegion].x = 0;
			if (aRegions[iRegion].y < 0)		aRegions[iRegion].y = 0;
			if (aRegions[iRegion].width + aRegions[iRegion].x >= rFrame.cols)	aRegions[iRegion].width = rFrame.cols - aRegions[iRegion].x - 1;
			if (aRegions[iRegion].height + aRegions[iRegion].y > rFrame.rows)	aRegions[iRegion].height = rFrame.rows - aRegions[iRegion].y - 1;

			cv::Mat oDummyMat = cv::Mat::zeros(rFrame.rows, rFrame.cols, CV_8U);

			for(int i=aRegions[iRegion].y; i<aRegions[iRegion].y+aRegions[iRegion].height; ++i) {
				for(int j=aRegions[iRegion].x; j<aRegions[iRegion].x+aRegions[iRegion].width; ++j) {
					oDummyMat.at<uchar>(i, j) = rFrame.at<uchar>(i, j);
				}
			}

			ClusterPCL(pViewer, oDummyMat, iRegion*10);
	}

}

std::vector<cv::Rect2i> ExtractRegions(const cv::Mat& rCoarseRegions, int iNumRegions) {
	//cout << "Mat Type: " << type2str(rCoarseRegions.type()) << endl;
	//cout << rCoarseRegions << endl;
	vector<cv::Rect2i> aResult(iNumRegions);

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i] = Rect2i(-1, -1, -1, -1);
	}

	for (int i = 0; i < rCoarseRegions.rows; ++i) {
		for (int j = 0; j < rCoarseRegions.cols; ++j) {
			if (rCoarseRegions.at<uchar>(i, j) > 0) {
				int iRegionIndex = (int)(rCoarseRegions.at<uchar>(i, j))-1;

				Rect2i& rRegion = aResult[iRegionIndex];
				if (rRegion.x == -1) {
					rRegion.x = j;
					rRegion.width = j;
					rRegion.y = i;
					rRegion.height = i;
				}
				else {
					if (j < rRegion.x) {
						rRegion.x = j;
					}
					if (j > rRegion.width) {
						rRegion.width = j;
					}
					if (i < rRegion.y) {
						rRegion.y = i;
					}
					if (i > rRegion.height) {
						rRegion.height = i;
					}
				}
			}
		}
	}

	for (int i = 0; i < aResult.size(); ++i) {
		aResult[i].width = aResult[i].width - aResult[i].x+1;
		aResult[i].height = aResult[i].height - aResult[i].y+1;
	}

	return aResult;
}

std::vector<cv::Rect2i> MergeRegions(const std::vector<cv::Rect2i>& aRegions) {
	vector<cv::Rect2i> aRegionCopy(aRegions);
	vector<cv::Rect2i> aResult;

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		for (size_t j = i + 1; j < aRegionCopy.size(); ++j) {
			if ((aRegionCopy[i] & aRegionCopy[j]).area() > 0) {
				aRegionCopy[i] = (aRegionCopy[i] | aRegionCopy[j]);
				aRegionCopy[j] = cv::Rect2i();
			}
		}
	}

	for (size_t i = 0; i < aRegionCopy.size(); ++i) {
		if (aRegionCopy[i].area() > 0) {
			aResult.push_back(aRegionCopy[i]);
		}
	}

	return aResult;
}
