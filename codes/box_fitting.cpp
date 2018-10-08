#include <stereo_processing/box_fitting/box_fitting.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/conversions.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl_ros/point_cloud.h>
#include <visualization_msgs/MarkerArray.h>

namespace stereo_processing
{
BoxFitting::~BoxFitting()
{
  // Ensure everything is stopped
  if (this->isRunning())
  {
    this->stop();
  }
}

void BoxFitting::onInit()
{
  // Init member variables
  nh_ = getNodeHandle();
  private_nh_ = getPrivateNodeHandle();
  //	image_transporter_ = std::make_shared<image_transport::ImageTransport>(image_transport::ImageTransport(nh_));
  dyn_param_server_.reset(new dynamic_reconfigure::Server<stereo_processing::box_fitting_configConfig>(
      guard_dyn_param_server_recursive_mutex_, private_nh_));
  node_running_ = false;
  node_active_ = false;

  // Read static parameters
  this->readStaticParams();

  // Bind dynamic_reconfigure callback function (This callback will always be called)
  dyn_param_server_->setCallback(boost::bind(&BoxFitting::reconfigureRequest, this, _1, _2));
}

bool BoxFitting::readStaticParams()
{
  bool rval = true;

  return rval;
}

void BoxFitting::reconfigureRequest(stereo_processing::box_fitting_configConfig& new_config, uint32_t level)
{
  ROS_INFO("Node dynamic reconfigure.");
  (void)level;
  // Check parameter validity
  // -> Nothing to do here at the moment

  // Stop the node if needed
  if (!new_config.Active)  // && this->isRunning())
  {
    if (this->stop())
    {
      this->setActive(false);
    }
    else
    {
      ROS_ERROR("Node could not be stopped.");
      ros::shutdown();
    }
  }

  if (std::strcmp(topic_in_.c_str(), new_config.topic_in.c_str()) != 0)
  {
    topic_in_ = new_config.topic_in;

    sub_ = std::make_shared<ros::Subscriber>(private_nh_.subscribe<pcl::PointCloud<pcl::PointXYZL> >(
        topic_in_, 1, boost::bind(&BoxFitting::processInput, this, _1)));
  }

  if (std::strcmp(topic_out_.c_str(), new_config.topic_out.c_str()) != 0)
  {
    topic_out_ = new_config.topic_out;
    pub_ = std::make_shared<ros::Publisher>(private_nh_.advertise<visualization_msgs::MarkerArray>(topic_out_, 1));
  }

  this->setActive(new_config.Active);

  // Start the node if needed
  if (new_config.Active)  // && !this->isRunning())
  {
    if (!this->start())
    {
      ROS_ERROR("Node could not be started.");
      this->setActive(false);
    }
  }
}

bool BoxFitting::isRunning()
{
  std::unique_lock<std::mutex> lock(node_state_mutex_);
  return node_running_;
}

bool BoxFitting::isActive()
{
  std::unique_lock<std::mutex> lock(node_state_mutex_);
  return node_active_;
}

bool BoxFitting::start()
{
  if (!this->isRunning())  // && this->isActive())
  {
    ROS_INFO("Starting %s", getName().c_str());

    // Start library
    if (!library_.start())
    {
      return false;
    }

    // Subscribe to topic
    sub_ = std::make_shared<ros::Subscriber>(private_nh_.subscribe<pcl::PointCloud<pcl::PointXYZL> >(
        topic_in_, 1, boost::bind(&BoxFitting::processInput, this, _1)));

    pub_ = std::make_shared<ros::Publisher>(private_nh_.advertise<visualization_msgs::MarkerArray>(topic_out_, 1));

    return (this->setRunning(true));
  }

  ROS_DEBUG("Node is running already");
  return true;
}

bool BoxFitting::stop()
{
  if (this->isRunning())
  {
    ROS_INFO("Stopping %s", getName().c_str());

    // Stop library
    if (!library_.stop())
    {
      return false;
    }

    // Unsubscribe from topic
    sub_->shutdown();
    pub_->shutdown();

    return (this->setRunning(false));
  }

  //		ROS_WARN("Node was already stopped");
  return true;
}

bool BoxFitting::setRunning(bool running)
{
  std::unique_lock<std::mutex> lock(node_state_mutex_);
  {
    node_running_ = running;
  }
  return true;
}

bool BoxFitting::setActive(bool active)
{
  std::unique_lock<std::mutex> lock(node_state_mutex_);
  {
    node_active_ = active;
  }
  return true;
}

void BoxFitting::processInput(const pcl::PointCloud<pcl::PointXYZL>::ConstPtr& msg)
{
  if (pub_->getNumSubscribers() == 0)
  {
    return;
  }

  // ros::Time measure_start1 = ros::Time::now();

  auto result = library_.fitBoxes(msg);
  if (result == nullptr)
  {
    return;
  }

  visualization_msgs::MarkerArray markers;

  visualization_msgs::Marker deleteMarker;
  deleteMarker.header.frame_id = msg->header.frame_id;
  deleteMarker.header.stamp = ros::Time::now();
  deleteMarker.action = visualization_msgs::Marker::DELETEALL;
  markers.markers.push_back(deleteMarker);

  int i = 0;
  for (auto& box : *result)
  {
    visualization_msgs::Marker marker;

    marker.header.frame_id = msg->header.frame_id;
    marker.header.stamp = ros::Time::now();
    marker.type = visualization_msgs::Marker::CUBE;
    marker.ns = "cluster_boxes";
    marker.id = i;
    marker.action = visualization_msgs::Marker::ADD;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;
    marker.lifetime = ros::Duration();
    marker.pose.position.x = box.x;
    marker.pose.position.y = box.y;
    marker.pose.position.z = box.z;
    marker.pose.orientation.x = box.quart_x;
    marker.pose.orientation.y = box.quart_y;
    marker.pose.orientation.z = box.quart_z;
    marker.pose.orientation.w = box.quart_w;
    marker.scale.x = box.scale_x;
    marker.scale.y = box.scale_y;
    marker.scale.z = box.scale_z;

    markers.markers.push_back(marker);
    i++;
  }

  pub_->publish(markers);

  // std::cout << "Total: " << ros::Duration(ros::Time::now() - measure_start1).toSec() << std::endl;
}

}  // namespace stereo_processing
