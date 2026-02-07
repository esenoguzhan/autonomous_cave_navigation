#include <rclcpp/rclcpp.hpp>

#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/point_stamped.hpp>
#include <std_msgs/msg/float32_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/int16.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <sensor_msgs/image_encodings.hpp>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

#include <cmath>
#include <vector>
#include <string>
#include <sstream>

class LanternDetector : public rclcpp::Node
{
public:
  LanternDetector()
  : Node("lantern_detector"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    lantern_count_(0)
  {
    this->declare_parameter<double>("distance_threshold", 100.0);
    this->declare_parameter<double>("drone_lantern_dist", 30.0);
    this->declare_parameter<std::string>("camera_frame", "camera");
    this->declare_parameter<std::string>("world_frame", "world");

    distance_threshold_ = this->get_parameter("distance_threshold").as_double();
    drone_lantern_dist_ = this->get_parameter("drone_lantern_dist").as_double();
    camera_frame_ = this->get_parameter("camera_frame").as_string();
    world_frame_ = this->get_parameter("world_frame").as_string();

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/current_state_est", rclcpp::QoS(1),
      std::bind(&LanternDetector::odomCallback, this, std::placeholders::_1));

    semantic_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/Quadrotor/Sensors/SemanticCamera/image_raw", rclcpp::QoS(10),
      std::bind(&LanternDetector::semanticCallback, this, std::placeholders::_1));

    depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/realsense/depth/image", rclcpp::QoS(10),
      std::bind(&LanternDetector::depthCallback, this, std::placeholders::_1));

    camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
      "/realsense/depth/camera_info", rclcpp::QoS(10),
      std::bind(&LanternDetector::cameraInfoCallback, this, std::placeholders::_1));

    // ---- Publishers ----
    lantern_positions_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>(
      "/lantern_positions", rclcpp::QoS(10));

    lantern_text_pub_ = this->create_publisher<std_msgs::msg::String>(
      "/detected_lanterns", rclcpp::QoS(10));

    lantern_count_pub_ = this->create_publisher<std_msgs::msg::Int16>(
      "/num_lanterns", rclcpp::QoS(10));

    RCLCPP_INFO(this->get_logger(), "LanternDetector ROS2 node started.");
  }

private:
  // Subscribers
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr semantic_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  // Publishers
  rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr lantern_positions_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr lantern_text_pub_;
  rclcpp::Publisher<std_msgs::msg::Int16>::SharedPtr lantern_count_pub_;

  // TF2
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // Depth + intrinsics
  cv::Mat depth_image_;
  sensor_msgs::msg::CameraInfo camera_info_;
  bool has_camera_info_{false};

  // Drone position
  geometry_msgs::msg::Point drone_position_;

  // Known lanterns
  std::vector<geometry_msgs::msg::Point> known_lantern_positions_;
  int lantern_count_;

  // Thresholds
  double distance_threshold_{100.0};
  double drone_lantern_dist_{30.0};

  // Frames
  std::string camera_frame_{"camera"};
  std::string world_frame_{"world"};

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    drone_position_ = msg->pose.pose.position;
  }

  void semanticCallback(const sensor_msgs::msg::Image::SharedPtr semantic_msg)
  {
    // Her callback'te yeni array üret (ROS1 kodunda birikiyordu)
    std_msgs::msg::Float32MultiArray new_lanterns;

    // Convert to CV MONO8
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(semantic_msg, sensor_msgs::image_encodings::MONO8);
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Semantic cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat semantic_img = cv_ptr->image;
    if (cv::countNonZero(semantic_img) == 0) {
      return;
    }

    // Connected components
    cv::Mat labels, stats, centroids;
    int num_components = cv::connectedComponentsWithStats(
      semantic_img, labels, stats, centroids, 8, CV_32S);

    for (int label_idx = 1; label_idx < num_components; ++label_idx) {
      double cx = centroids.at<double>(label_idx, 0);
      double cy = centroids.at<double>(label_idx, 1);

      int px = static_cast<int>(cx);
      int py = static_cast<int>(cy);

      geometry_msgs::msg::Point cam_point = pixelTo3D(px, py);

      if (!depth_image_.empty() &&
          px >= 0 && px < depth_image_.cols &&
          py >= 0 && py < depth_image_.rows)
      {
        uint16_t depth_mm = depth_image_.at<uint16_t>(py, px);
        if (depth_mm == 0) {
          RCLCPP_INFO(this->get_logger(),
                      "Detected lantern pixel but depth=0 => skipping.");
          continue;
        }

        RCLCPP_INFO(this->get_logger(),
                    "Camera coords=(%.3f,%.3f,%.3f), depth=%u mm",
                    cam_point.x, cam_point.y, cam_point.z, depth_mm);
      }

      geometry_msgs::msg::PointStamped world_point = transformToWorldFrame(cam_point);

      double dist_drone_lantern = distanceBetween(drone_position_, world_point.point);
      RCLCPP_INFO(this->get_logger(),
                  "Distance between drone and Lantern is %.2f", dist_drone_lantern);

      if (dist_drone_lantern > drone_lantern_dist_) {
        continue;
      }

      if (isNewLantern(world_point.point)) {
        lantern_count_++;
        known_lantern_positions_.push_back(world_point.point);

        RCLCPP_INFO(this->get_logger(),
                    "Lantern %d detected at (%.2f,%.2f,%.2f), dist=%.2f m",
                    lantern_count_,
                    world_point.point.x, world_point.point.y, world_point.point.z,
                    dist_drone_lantern);

        new_lanterns.data.push_back(static_cast<float>(world_point.point.x));
        new_lanterns.data.push_back(static_cast<float>(world_point.point.y));
        new_lanterns.data.push_back(static_cast<float>(world_point.point.z));

        // Text publish
        std_msgs::msg::String text_msg;
        std::stringstream ss;
        ss << "Lantern " << lantern_count_
           << " at (" << world_point.point.x
           << ", " << world_point.point.y
           << ", " << world_point.point.z
           << "), droneDist=" << dist_drone_lantern << "m";
        text_msg.data = ss.str();
        lantern_text_pub_->publish(text_msg);

        // Count publish
        std_msgs::msg::Int16 count_msg;
        count_msg.data = static_cast<int16_t>(lantern_count_);
        lantern_count_pub_->publish(count_msg);
      }
    }

    if (!new_lanterns.data.empty()) {
      lantern_positions_pub_->publish(new_lanterns);
    }
  }

  geometry_msgs::msg::Point pixelTo3D(int px, int py)
  {
    geometry_msgs::msg::Point pt;

    if (depth_image_.empty() || !has_camera_info_ || camera_info_.k.empty()) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                           "Depth image or camera info not ready.");
      return pt; // (0,0,0)
    }

    // bounds check
    if (px < 0 || py < 0 || px >= depth_image_.cols || py >= depth_image_.rows) {
      return pt;
    }

    uint16_t depth_raw = depth_image_.at<uint16_t>(py, px); // mm
    float depth_m = depth_raw * 0.001f;

    float fx = static_cast<float>(camera_info_.k[0]);
    float fy = static_cast<float>(camera_info_.k[4]);
    float cx = static_cast<float>(camera_info_.k[2]);
    float cy = static_cast<float>(camera_info_.k[5]);

    pt.x = (static_cast<float>(px) - cx) * depth_m / fx;
    pt.y = (static_cast<float>(py) - cy) * depth_m / fy;
    pt.z = depth_m;
    return pt;
  }

  geometry_msgs::msg::PointStamped transformToWorldFrame(const geometry_msgs::msg::Point &cam_point)
  {
    geometry_msgs::msg::PointStamped camera_pt, world_pt;
    camera_pt.header.frame_id = camera_frame_;
    camera_pt.header.stamp = this->now();
    camera_pt.point = cam_point;

    try {

      world_pt = tf_buffer_.transform(camera_pt, world_frame_, tf2::durationFromSec(0.2));
      RCLCPP_INFO(this->get_logger(),
                  "World coords=(%.3f,%.3f,%.3f)",
                  world_pt.point.x, world_pt.point.y, world_pt.point.z);
      return world_pt;
    } catch (const tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "transformToWorldFrame error: %s", ex.what());
      return camera_pt;
    }
  }

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr depth_msg)
  {
    try {
      auto cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
      depth_image_ = cv_ptr->image;
    } catch (const cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "Depth cv_bridge exception: %s", e.what());
    }
  }

  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr info_msg)
  {
    camera_info_ = *info_msg;
    has_camera_info_ = true;
  }

  bool isNewLantern(const geometry_msgs::msg::Point &candidate)
  {
    for (const auto &known : known_lantern_positions_) {
      double d = distanceBetween(known, candidate);
      if (d < distance_threshold_) {
        return false;
      }
    }
    return true;
  }

  double distanceBetween(const geometry_msgs::msg::Point &p1,
                         const geometry_msgs::msg::Point &p2)
  {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    double dz = p1.z - p2.z;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LanternDetector>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
