#include <SegmentationHelper.h>

void AnalysePointcloud(boost::shared_ptr<pcl::PointCloud<pcl::PointXYZ> > pCloud, std::vector<double>& rDimension, std::vector<double>& rEccentricity,
		pcl::PointXYZ& rPosition, pcl::PointXYZ& rOBBPosition, pcl::PointXYZ& rOBBMin, pcl::PointXYZ& rOBBMax, Eigen::Matrix3f& rOBBRot)
{
	pcl::MomentOfInertiaEstimation<pcl::PointXYZ> MoIEstimation;
	MoIEstimation.setInputCloud(pCloud);
	MoIEstimation.compute();

	Eigen::Vector3f oCenterOfMass;
	if (!MoIEstimation.getMassCenter(oCenterOfMass)) {
		std::cout << "Error: masscenter" << std::endl;
	}

	pcl::PointXYZ OBBMin;
	pcl::PointXYZ OBBMax;
	pcl::PointXYZ OBBPosition;
	Eigen::Matrix3f OBBRot;
	if (!MoIEstimation.getOBB(OBBMin, OBBMax, OBBPosition, OBBRot)) {
		std::cout << "Error: getOBB" << std::endl;
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
