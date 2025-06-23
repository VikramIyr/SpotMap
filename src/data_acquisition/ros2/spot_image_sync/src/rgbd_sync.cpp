#include "rclcpp/rclcpp.hpp"
#include <chrono>
#include <functional>
#include <memory>

#include "message_filters/subscriber.h"
#include "message_filters/synchronizer.h"
#include "message_filters/sync_policies/approximate_time.h"

#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"

#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

using namespace std::chrono_literals;
using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;

class RGBDSyncNode : public rclcpp::Node
{
public:
  RGBDSyncNode()
  : Node("rgbd_sync_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    rclcpp::QoS qos_profile(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default));
    qos_profile
      .reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE)
      .history(RMW_QOS_POLICY_HISTORY_KEEP_LAST)
      .keep_last(10)
      .durability(RMW_QOS_POLICY_DURABILITY_VOLATILE);

    // Subscribers
    rgb_sub_.subscribe(this, "/camera/hand/image", qos_profile.get_rmw_qos_profile());
    depth_sub_.subscribe(this, "/depth/hand/image", qos_profile.get_rmw_qos_profile());
    rgb_info_sub_.subscribe(this, "/camera/hand/camera_info", qos_profile.get_rmw_qos_profile());
    depth_info_sub_.subscribe(this, "/depth/hand/camera_info", qos_profile.get_rmw_qos_profile());

    // Publishers
    rgb_pub_   = this->create_publisher<sensor_msgs::msg::Image>("rgb_synced/image", qos_profile);
    depth_pub_ = this->create_publisher<sensor_msgs::msg::Image>("depth_synced/image", qos_profile);
    rgb_info_pub_   = this->create_publisher<sensor_msgs::msg::CameraInfo>("rgb_synced/camera_info", qos_profile);
    depth_info_pub_ = this->create_publisher<sensor_msgs::msg::CameraInfo>("depth_synced/camera_info", qos_profile);
    pose_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/camera_pose", 10);

    // Synchronizer parameters
    uint32_t queue_size = this->declare_parameter("queue_size", 10);
    double max_interval = this->declare_parameter("max_interval", 0.03);
    double age_penalty = this->declare_parameter("age_penalty", 1.0);

    sync_ = std::make_shared<Sync>(SyncPolicy(queue_size), rgb_sub_, depth_sub_, rgb_info_sub_, depth_info_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval));
    sync_->setAgePenalty(age_penalty);
    sync_->registerCallback(std::bind(&RGBDSyncNode::syncCallback, this, _1, _2, _3, _4));
  }

private:
  void syncCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr &rgb,
    const sensor_msgs::msg::Image::ConstSharedPtr &depth,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info_rgb_,
    const sensor_msgs::msg::CameraInfo::ConstSharedPtr &info_depth_)
  {
    auto now = this->get_clock()->now();


    auto rgb_out = *rgb;              rgb_out.header.stamp = now;
    auto depth_out = *depth;          depth_out.header.stamp = now;
    auto info_rgb = *info_rgb_;       info_rgb.header.stamp = now;
    auto info_depth = *info_depth_;   info_depth.header.stamp = now;

    rgb_pub_->publish(rgb_out);
    depth_pub_->publish(depth_out);
    rgb_info_pub_->publish(info_rgb);
    depth_info_pub_->publish(info_depth);

    // Publish camera pose
    try {
      auto tf = tf_buffer_.lookupTransform("vision", "hand_color_image_sensor", tf2::TimePointZero);
      tf.header.stamp = now;
      pose_pub_->publish(tf);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
    }
  }

  using SyncPolicy = message_filters::sync_policies::ApproximateTime<
                       sensor_msgs::msg::Image,
                       sensor_msgs::msg::Image,
                       sensor_msgs::msg::CameraInfo,
                       sensor_msgs::msg::CameraInfo>;
  using Sync = message_filters::Synchronizer<SyncPolicy>;

  message_filters::Subscriber<sensor_msgs::msg::Image> rgb_sub_, depth_sub_;
  message_filters::Subscriber<sensor_msgs::msg::CameraInfo> rgb_info_sub_, depth_info_sub_;
  std::shared_ptr<Sync> sync_;

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr rgb_pub_, depth_pub_;
  rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr rgb_info_pub_, depth_info_pub_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr pose_pub_;

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RGBDSyncNode>());
  rclcpp::shutdown();
  return 0;
}
