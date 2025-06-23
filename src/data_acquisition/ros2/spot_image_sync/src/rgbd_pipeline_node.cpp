#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <image_geometry/pinhole_camera_model.h>
#include <depth_image_proc/depth_traits.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>

using sensor_msgs::msg::Image;
using sensor_msgs::msg::CameraInfo;
using geometry_msgs::msg::TransformStamped;

class RGBDPipelineNode : public rclcpp::Node
{
public:
  RGBDPipelineNode()
  : Node("rgbd_pipeline_node"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10)).reliable().durability_volatile();

    rgb_sub_.subscribe(this, "/camera/hand/image", qos.get_rmw_qos_profile());
    rgb_info_sub_.subscribe(this, "/camera/hand/camera_info", qos.get_rmw_qos_profile());
    depth_sub_.subscribe(this, "/depth/hand/image", qos.get_rmw_qos_profile());
    depth_info_sub_.subscribe(this, "/depth/hand/camera_info", qos.get_rmw_qos_profile());

    rgb_pub_       = create_publisher<Image>("/rgbd_pipeline/rgb/image", qos);
    rgb_info_pub_  = create_publisher<CameraInfo>("/rgbd_pipeline/rgb/camera_info", qos);
    depth_pub_     = create_publisher<Image>("/rgbd_pipeline/depth_registered/image", qos);
    depth_info_pub_= create_publisher<CameraInfo>("/rgbd_pipeline/depth_registered/camera_info", qos);
    pose_pub_      = create_publisher<TransformStamped>("/camera_pose", 10);

    // Synchronizer parameters
    uint32_t queue_size = this->declare_parameter("queue_size", 10);
    double max_interval = this->declare_parameter("max_interval", 0.03);
    double age_penalty = this->declare_parameter("age_penalty", 1.0);

    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
      Image, CameraInfo, Image, CameraInfo>;
    sync_.reset(new message_filters::Synchronizer<SyncPolicy>(SyncPolicy(queue_size),
      rgb_sub_, rgb_info_sub_, depth_sub_, depth_info_sub_));
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(max_interval));
    sync_->setAgePenalty(age_penalty);
    sync_->registerCallback(
      std::bind(&RGBDPipelineNode::callback, this, std::placeholders::_1,
                std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
  }

private:
  void callback(
    const Image::ConstSharedPtr & rgb_msg,
    const CameraInfo::ConstSharedPtr & rgb_info,
    const Image::ConstSharedPtr & depth_msg,
    const CameraInfo::ConstSharedPtr & depth_info)
  {
    auto now = this->get_clock()->now();

    cv_bridge::CvImageConstPtr cv_depth;
    try {
      cv_depth = cv_bridge::toCvShare(depth_msg);
    } catch (cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    double scale_x = double(rgb_info->width) / depth_msg->width;
    double scale_y = double(rgb_info->height) / depth_msg->height;
    cv::Mat depth_resized;
    cv::resize(cv_depth->image, depth_resized, cv::Size(), scale_x, scale_y, cv::INTER_NEAREST);

    CameraInfo depth_info_scaled = *depth_info;
    depth_info_scaled.header.stamp = now;
    depth_info_scaled.width  = rgb_info->width;
    depth_info_scaled.height = rgb_info->height;
    depth_info_scaled.k[0] *= scale_x;
    depth_info_scaled.k[2] *= scale_x;
    depth_info_scaled.k[4] *= scale_y;
    depth_info_scaled.k[5] *= scale_y;
    depth_info_scaled.p[0] *= scale_x;
    depth_info_scaled.p[2] *= scale_x;
    depth_info_scaled.p[3] *= scale_x;
    depth_info_scaled.p[5] *= scale_y;
    depth_info_scaled.p[6] *= scale_y;
    depth_info_scaled.roi.x_offset = int(depth_info_scaled.roi.x_offset * scale_x);
    depth_info_scaled.roi.y_offset = int(depth_info_scaled.roi.y_offset * scale_y);
    depth_info_scaled.roi.width    = int(depth_info_scaled.roi.width * scale_x);
    depth_info_scaled.roi.height   = int(depth_info_scaled.roi.height * scale_y);

    image_geometry::PinholeCameraModel depth_model, rgb_model;
    depth_model.fromCameraInfo(depth_info_scaled);
    rgb_model.fromCameraInfo(*rgb_info);

    Eigen::Affine3d depth_to_rgb;
    try {
      auto tf = tf_buffer_.lookupTransform(
        rgb_info->header.frame_id,
        depth_info->header.frame_id,
        tf2::TimePointZero);
      depth_to_rgb = tf2::transformToEigen(tf);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "TF lookup failed: %s", ex.what());
      return;
    }

    Image reg_msg;
    reg_msg.header.stamp = now;
    reg_msg.header.frame_id = rgb_info->header.frame_id;
    reg_msg.height = rgb_info->height;
    reg_msg.width  = rgb_info->width;
    reg_msg.encoding = depth_msg->encoding;
    reg_msg.step = sizeof(uint16_t) * reg_msg.width;
    reg_msg.data.assign(reg_msg.step * reg_msg.height, 0);

    // Cache width/height as signed ints to avoid signed/unsigned comparison warnings
    int reg_width  = static_cast<int>(reg_msg.width);
    int reg_height = static_cast<int>(reg_msg.height);

    const auto & di = depth_model;
    const auto & ri = rgb_model;
    double inv_dx = 1.0/di.fx();
    double inv_dy = 1.0/di.fy();
    for (int v = 0; v < depth_resized.rows; ++v) {
      for (int u = 0; u < depth_resized.cols; ++u) {
        uint16_t raw = depth_resized.at<uint16_t>(v,u);
        if (!depth_image_proc::DepthTraits<uint16_t>::valid(raw)) continue;
        double z = depth_image_proc::DepthTraits<uint16_t>::toMeters(raw);
        Eigen::Vector4d pt_d;
        pt_d << ((u - di.cx())*z - di.Tx()) * inv_dx,
                ((v - di.cy())*z - di.Ty()) * inv_dy,
                z, 1.0;
        auto pt_rgb = depth_to_rgb * pt_d;
        double iz = 1.0 / pt_rgb.z();
        int u2 = int((ri.fx()*pt_rgb.x() + ri.Tx()) * iz + ri.cx() + 0.5);
        int v2 = int((ri.fy()*pt_rgb.y() + ri.Ty()) * iz + ri.cy() + 0.5);
        if (u2 < 0 || u2 >= reg_width || v2 < 0 || v2 >= reg_height) continue;
        uint16_t d2 = depth_image_proc::DepthTraits<uint16_t>::fromMeters(pt_rgb.z());
        auto & out = *reinterpret_cast<uint16_t*>(reg_msg.data.data() + v2*reg_msg.step + u2*sizeof(uint16_t));
        if (!depth_image_proc::DepthTraits<uint16_t>::valid(out) || out > d2) out = d2;
      }
    }

    auto rgb_out = *rgb_msg;
    rgb_out.header.stamp = now;
    rgb_pub_->publish(rgb_out);
    auto ri_out = *rgb_info;
    ri_out.header.stamp = now;
    rgb_info_pub_->publish(ri_out);
    depth_pub_->publish(reg_msg);
    depth_info_pub_->publish(depth_info_scaled);

    try {
      auto tf_cam = tf_buffer_.lookupTransform(
        "vision", "hand_color_image_sensor", tf2::TimePointZero);
      tf_cam.header.stamp = now;
      pose_pub_->publish(tf_cam);
    } catch (const tf2::TransformException & ex) {
      RCLCPP_WARN(get_logger(), "TF lookup failed: %s", ex.what());
    }
  }

  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  message_filters::Subscriber<Image>      rgb_sub_, depth_sub_;
  message_filters::Subscriber<CameraInfo> rgb_info_sub_, depth_info_sub_;
  std::shared_ptr<message_filters::Synchronizer<
    message_filters::sync_policies::ApproximateTime<Image, CameraInfo, Image, CameraInfo>>> sync_;
  rclcpp::Publisher<Image>::SharedPtr       rgb_pub_, depth_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr  rgb_info_pub_, depth_info_pub_;
  rclcpp::Publisher<TransformStamped>::SharedPtr pose_pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<RGBDPipelineNode>());
  rclcpp::shutdown();
  return 0;
}
