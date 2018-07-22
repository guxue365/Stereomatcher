#include <iostream>

#include <pcl/visualization/cloud_viewer.h>

using namespace std;

int main() {
	//Punktwolke erstellen
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	cloud->push_back(pcl::PointXYZ(0.0, 0.0, 0.0));
	cloud->push_back(pcl::PointXYZ(1.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(2.0, 2.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 1.0, 0.0));
	cloud->push_back(pcl::PointXYZ(0.0, 2.0, 0.0));
	//Punktwolke laden (siehe oben) oder neue erstellen
	//CloudViewer zur Visualisierung erstellen
	pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
	//Aufruf zur Darstellung der Punktwolke
	viewer.showCloud(cloud);
	while (!viewer.wasStopped()) {
		//mögliche Prozessierungsanweisungen möglich in einem extra Thread
	}
	return 0;
}
