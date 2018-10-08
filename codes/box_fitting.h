#ifndef box_fitting_H
#define box_fitting_H

// Includes
#include <cstdint>                 // types
#include <iostream>                // std::cout
#include <mutex>                   // std::mutex
#include <memory>                  // std::shared_ptr
#include "boost/thread/mutex.hpp"  // boost::recursive_mutex
#include "boost/bind.hpp"
#include <math.h>
#include <functional>

// ROS header
#include <ros/ros.h>  // Ros core
#include <sensor_msgs/PointCloud2.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

// Dynamic reconfigure and cfg
#include <dynamic_reconfigure/server.h>
#include "stereo_processing/box_fitting_configConfig.h"

// Library
#include <stereo_processing/box_fitting_lib/box_fitting_lib.h>

namespace stereo_processing
{
class BoxFitting : public nodelet::Nodelet
{
public:
  ~BoxFitting();

  /**
   * @brief Nodelet initialization function
   */
  void onInit();

  /**
   * @brief This function returns the internal running state of the node
   */
  bool isRunning();

  /**
   * @brief This function returns the internal activity state of the node
   */
  bool isActive();

private:
  /**
   * @brief Static parameter initialization
   * @return True if successful
   */
  bool readStaticParams();

  /**
   * @brief Callback function for dynamic reconfigure nodes
   * @param configuration object
   * @param level
   */
  void reconfigureRequest(stereo_processing::box_fitting_configConfig& new_config, uint32_t level);

  /**
   * @brief This starts the node
   * @return True if start was successful
   */
  bool start();

  /**
   * @brief This stops the node
   * @return True if stop was successful
   */
  bool stop();

  /**
   * @brief Set the internal running state
   * @param State to set
   * @return True if set was successful
   */
  bool setRunning(bool running);

  /**
   * @brief Set the internal activity state
   * @param State to set
   * @return True if set was successful
   */
  bool setActive(bool active);

  /**
   * @brief This processes incoming messages of a specific type
   */
  void processInput(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& msg);

  /**
   * Public and private ros node handle
   */
  ros::NodeHandle nh_;
  ros::NodeHandle private_nh_;

  /**
   * Recursive mutex guards the param server to prevent parallel reconfiguration conflicts
   */
  boost::recursive_mutex guard_dyn_param_server_recursive_mutex_;

  /**
   * Node mutex to guard the state variables from parallel access
   */
  std::mutex node_state_mutex_;

  /**
   * Dynamic parameter server object
   */
  std::shared_ptr<dynamic_reconfigure::Server<stereo_processing::box_fitting_configConfig> > dyn_param_server_;

  std::shared_ptr<ros::Subscriber> sub_;
  std::shared_ptr<ros::Publisher> pub_;

  /**
   * Indicates the internal running state of the node
   */
  bool node_running_;

  /**
   * Indicates the state of the activity parameter of the node
   */
  bool node_active_;

  /**
   * Input topic
   */
  std::string topic_in_;

  /**
   * Output topic
   */
  std::string topic_out_;

  mrm::stereo_processing::BoxFittingLib library_;
};

}  // namespace stereo_processing

// watch the capitalization carefully
PLUGINLIB_EXPORT_CLASS(stereo_processing::BoxFitting, nodelet::Nodelet)

#endif  // box_fitting
