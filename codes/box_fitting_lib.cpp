
#include "stereo_processing/box_fitting_lib/box_fitting_lib.h"
#include <boost/make_shared.hpp>
#include <pcl/common/common.h>
#include <pcl/common/centroid.h>
#include <pcl/common/transforms.h>
#include <Eigen/Eigenvalues>

namespace mrm
{
namespace stereo_processing
{
BoxFittingLib::BoxFittingLib() : started(false)
{
}

bool BoxFittingLib::start()
{
  if (started)
  {
    return true;
  }
  started = true;
  return true;
}

bool BoxFittingLib::stop()
{
  if (!started)
  {
    return true;
  }
  started = false;
  return true;
}

boost::shared_ptr<std::list<ClusterBox> >
BoxFittingLib::fitBoxes(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& cloud)
{
  auto clusterMap = getClusterIndicesFromPointCloud(cloud);

  auto result = boost::make_shared<std::list<ClusterBox> >();

  for (auto& clusterEntry : *clusterMap)
  {
    auto clusterIndices = clusterEntry.second;

    // for (auto& i: clusterIndices) {
    //   auto &pt = (*cloud)[i];
    //   std::cout << pt.x << " " << pt.y << " " << pt.z << std::endl;
    // }
    // std::cout << std::endl;

    // https://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html
    Eigen::Vector4f pcaCentroid;
    int cnt = pcl::compute3DCentroid(*cloud, clusterIndices, pcaCentroid);
    if (cnt == 0)
    {
      std::cerr << "Could not compute centroid of cluster!" << std::endl;
      continue;
    }

    // std::cout << "Centroid: " << pcaCentroid.x() << " " << pcaCentroid.y() << " " << pcaCentroid.z() << " " <<
    // pcaCentroid.w() << " (" << cnt << ")" << std::endl;

    Eigen::Matrix3f covariance;
    pcl::computeCovarianceMatrixNormalized(*cloud, clusterIndices, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();

    // This line is necessary for proper orientation in some cases. The numbers come out the same without it, but the
    // signs are different and the box doesn't get correctly oriented in some cases.
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    // std::cout << "Eigenvectors" << std::endl << eigenVectorsPCA << std::endl;

    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3, 1>(0, 3) = -1.f * (projectionTransform.block<3, 3>(0, 0) * pcaCentroid.head<3>());

    // std::cout << "Projection transform" << std::endl << projectionTransform << std::endl;

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloudPointsProjected(new pcl::PointCloud<pcl::PointXYZL>);
    pcl::transformPointCloud(*cloud, clusterIndices, *cloudPointsProjected, projectionTransform);

    // std::cout << "Transformed pointcloud" << std::endl;
    // for (auto& pt : *cloudPointsProjected) {
    //   std::cout << pt.x << " " << pt.y << " " << pt.z << std::endl;
    // }
    // std::cout << std::endl;

    // Get the minimum and maximum points of the transformed cloud.
    Eigen::Vector4f minPoint, maxPoint;
    pcl::getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    // std::cout << "Min: " << minPoint.x() << " " << minPoint.y() << " " << minPoint.z() << " " << minPoint.w() <<
    // std::endl; std::cout << "Max: " << maxPoint.x() << " " << maxPoint.y() << " " << maxPoint.z() << " " <<
    // maxPoint.w() << std::endl;

    const Eigen::Vector3f meanDiagonal = 0.5f * (maxPoint.head<3>() + minPoint.head<3>());

    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);  // rotation
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    ClusterBox box;
    box.x = bboxTransform[0];
    box.y = bboxTransform[1];
    box.z = bboxTransform[2];

    box.scale_x = maxPoint.x() - minPoint.x();
    box.scale_y = maxPoint.y() - minPoint.y();
    box.scale_z = maxPoint.z() - minPoint.z();

    box.quart_x = bboxQuaternion.x();
    box.quart_y = bboxQuaternion.y();
    box.quart_z = bboxQuaternion.z();
    box.quart_w = bboxQuaternion.w();

    // std::cout << "Trans: " << box.x << " " << box.y << " " << box.z << " | Scale: " << box.scale_x << " " <<
    // box.scale_y << " "
    //           << box.scale_z << " | Quart: " << box.quart_x << " " << box.quart_y << " " << box.quart_z << " " <<
    //           box.quart_w
    //           << std::endl;
    // std::cout << std::endl;
    // std::cout << std::endl;

    result->push_back(box);
  }

  return result;
}

boost::shared_ptr<ClusterMap>
BoxFittingLib::getClusterIndicesFromPointCloud(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& cloud)
{
  auto clusterMap = boost::make_shared<ClusterMap>();
  int i = 0;
  for (auto& p : *cloud)
  {
    auto clusterIndices = clusterMap->find(p.label);
    if (clusterMap->find(p.label) == clusterMap->end())
    {
      auto ret = clusterMap->insert(std::make_pair(p.label, std::vector<int>()));
      clusterIndices = ret.first;
    }
    clusterIndices->second.push_back(i);
    i++;
  }
  return clusterMap;
}

}  // namespace stereo_processing
}  // namespace mrm
