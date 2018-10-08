#ifndef LIBRARY_INCLUDE_box_fitting_LIB_H_
#define LIBRARY_INCLUDE_box_fitting_LIB_H_

// Includes
#include <iostream>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <map>
#include <list>

// Namespace (Use short but intuitive names, lower case letters, underscored)
namespace mrm
{
namespace stereo_processing
{
using ClusterMap = std::map<int, std::vector<int> >;
typedef struct
{
  PCL_ADD_UNION_POINT4D;
  PCL_ADD_EIGEN_MAPS_POINT4D;

  union
  {
    float scale_data[4];
    struct
    {
      float scale_x;
      float scale_y;
      float scale_z;
    };
  };

  union
  {
    float quart_data[4];
    struct
    {
      float quart_x;
      float quart_y;
      float quart_z;
      float quart_w;
    };
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} ClusterBox;
/**
 * @class MRMROSTemplaceLib
 * @brief this class may contain your library
 */
class BoxFittingLib
{
public:
  /**
   * @brief Parametrized Constructor of the library
   */
  BoxFittingLib();

  bool start();
  bool stop();

  boost::shared_ptr<std::list<ClusterBox> > fitBoxes(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& cloud);

protected:
private:
  bool started;
  boost::shared_ptr<ClusterMap> getClusterIndicesFromPointCloud(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& cloud);
};

}  // namespace stereo_processing
}  // namespace mrm

#endif /* LIBRARY_INCLUDE_box_fitting_H_ */
