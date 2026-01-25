#include <memory>
#include <string>
#include <vector>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <std_msgs/msg/empty.hpp>
#include <std_srvs/srv/empty.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <Eigen/Dense>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

// MAV trajectory generation includes
#include <mav_trajectory_generation/polynomial_optimization_nonlinear.h>
#include <mav_trajectory_generation/trajectory.h>
#include <mav_trajectory_generation/trajectory_sampling.h>
#include <mav_msgs/eigen_mav_msgs.hpp>
#include <mav_msgs/conversions.hpp>

using namespace std::chrono_literals;

class TrajectoryPlannerNode : public rclcpp::Node {
private:
  // ROS communication
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr current_state_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_state_pub_;
  rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr trajectory_complete_pub_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr start_navigation_srv_;
  rclcpp::TimerBase::SharedPtr sampling_timer_;
  
  // Current UAV state
  Eigen::Vector3d current_position_;
  Eigen::Vector3d current_velocity_;
  double current_yaw_;
  bool received_current_state_;
  
  // Trajectory parameters
  double max_v_;
  double max_a_;
  double sampling_dt_;
  std::vector<double> waypoints_x_;
  std::vector<double> waypoints_y_;
  std::vector<double> waypoints_z_;
  
  // Trajectory state
  mav_trajectory_generation::Trajectory trajectory_;
  double current_sample_time_;
  bool trajectory_active_;
  bool trajectory_generated_;

public:
  TrajectoryPlannerNode()
  : Node("trajectory_planner_node"),
    current_position_(0, 0, 0),
    current_velocity_(0, 0, 0),
    current_yaw_(0.0),
    received_current_state_(false),
    max_v_(5.0),
    max_a_(2.0),
    sampling_dt_(0.02),
    current_sample_time_(0.0),
    trajectory_active_(false),
    trajectory_generated_(false)
  {
    // Declare parameters
    this->declare_parameter<double>("max_v", 5.0);
    this->declare_parameter<double>("max_a", 2.0);
    this->declare_parameter<double>("sampling_dt", 0.02);
    this->declare_parameter<std::vector<double>>("waypoints.x", std::vector<double>());
    this->declare_parameter<std::vector<double>>("waypoints.y", std::vector<double>());
    this->declare_parameter<std::vector<double>>("waypoints.z", std::vector<double>());
    this->declare_parameter<std::string>("current_state_topic", "current_state_est");
    this->declare_parameter<std::string>("desired_state_topic", "desired_state");
    this->declare_parameter<std::string>("trajectory_complete_topic", "trajectory_complete");
    this->declare_parameter<std::string>("start_navigation_service", "start_navigation");
    this->declare_parameter<int>("subscriber_queue_size", 10);
    this->declare_parameter<int>("publisher_queue_size", 10);
    
    // Load parameters
    if (!loadParameters()) {
      RCLCPP_FATAL(this->get_logger(), 
        "Failed to load required parameters. Node cannot start.");
      rclcpp::shutdown();
      return;
    }
    
    // Get topic and service names
    std::string current_state_topic = this->get_parameter("current_state_topic").as_string();
    std::string desired_state_topic = this->get_parameter("desired_state_topic").as_string();
    std::string trajectory_complete_topic = this->get_parameter("trajectory_complete_topic").as_string();
    std::string start_navigation_service = this->get_parameter("start_navigation_service").as_string();
    int sub_queue_size = this->get_parameter("subscriber_queue_size").as_int();
    int pub_queue_size = this->get_parameter("publisher_queue_size").as_int();
    
    // Initialize subscriber
    current_state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      current_state_topic, sub_queue_size,
      std::bind(&TrajectoryPlannerNode::onCurrentState, this, std::placeholders::_1));
    
    // Initialize publishers
    desired_state_pub_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
      desired_state_topic, pub_queue_size);
    
    trajectory_complete_pub_ = this->create_publisher<std_msgs::msg::Empty>(
      trajectory_complete_topic, pub_queue_size);
    
    // Initialize service
    start_navigation_srv_ = this->create_service<std_srvs::srv::Empty>(
      start_navigation_service,
      std::bind(&TrajectoryPlannerNode::onStartNavigation, this, 
                std::placeholders::_1, std::placeholders::_2));
    
    // Log successful initialization
    RCLCPP_INFO(this->get_logger(), "╔════════════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║     Trajectory Planner Node Initialized Successfully      ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Trajectory Parameters:");
    RCLCPP_INFO(this->get_logger(), "  Max Velocity: %.2f m/s", max_v_);
    RCLCPP_INFO(this->get_logger(), "  Max Acceleration: %.2f m/s^2", max_a_);
    RCLCPP_INFO(this->get_logger(), "  Sampling Rate: %.0f Hz", 1.0 / sampling_dt_);
    RCLCPP_INFO(this->get_logger(), "  Waypoints: %zu points", waypoints_x_.size());
    RCLCPP_INFO(this->get_logger(), "Topics:");
    RCLCPP_INFO(this->get_logger(), "  Subscribing: %s", current_state_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Publishing:  %s", desired_state_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Complete:    %s", trajectory_complete_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "Service:");
    RCLCPP_INFO(this->get_logger(), "  Start:       %s", start_navigation_service.c_str());
  }

private:
  bool loadParameters() {
    try {
      max_v_ = this->get_parameter("max_v").as_double();
      max_a_ = this->get_parameter("max_a").as_double();
      sampling_dt_ = this->get_parameter("sampling_dt").as_double();
      waypoints_x_ = this->get_parameter("waypoints.x").as_double_array();
      waypoints_y_ = this->get_parameter("waypoints.y").as_double_array();
      waypoints_z_ = this->get_parameter("waypoints.z").as_double_array();
      
      // Validate waypoints
      if (waypoints_x_.empty() || waypoints_y_.empty() || waypoints_z_.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Waypoints are empty!");
        return false;
      }
      
      if (waypoints_x_.size() != waypoints_y_.size() || 
          waypoints_x_.size() != waypoints_z_.size()) {
        RCLCPP_ERROR(this->get_logger(), 
          "Waypoint vectors have different sizes (x: %zu, y: %zu, z: %zu)",
          waypoints_x_.size(), waypoints_y_.size(), waypoints_z_.size());
        return false;
      }
      
      // Validate trajectory parameters
      if (max_v_ <= 0 || max_a_ <= 0 || sampling_dt_ <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Invalid trajectory parameters!");
        return false;
      }
      
      return true;
      
    } catch (const std::exception& e) {
      RCLCPP_FATAL(this->get_logger(), "Error loading parameters: %s", e.what());
      return false;
    }
  }
  
  void onCurrentState(const nav_msgs::msg::Odometry::SharedPtr msg) {
    // Extract current position
    current_position_ << msg->pose.pose.position.x,
                         msg->pose.pose.position.y,
                         msg->pose.pose.position.z;
    
    // Extract current velocity
    current_velocity_ << msg->twist.twist.linear.x,
                         msg->twist.twist.linear.y,
                         msg->twist.twist.linear.z;
    
    // Extract current yaw
    current_yaw_ = tf2::getYaw(msg->pose.pose.orientation);
    
    received_current_state_ = true;
  }
  
  void onStartNavigation(
    const std::shared_ptr<std_srvs::srv::Empty::Request> /*request*/,
    std::shared_ptr<std_srvs::srv::Empty::Response> /*response*/)
  {
    RCLCPP_INFO(this->get_logger(), "Received start_navigation request");
    
    if (!received_current_state_) {
      RCLCPP_WARN(this->get_logger(), 
        "Cannot start navigation: No current state received yet!");
      return;
    }
    
    if (trajectory_active_) {
      RCLCPP_WARN(this->get_logger(), 
        "Trajectory already active, ignoring request");
      return;
    }
    
    // Generate trajectory from current position through waypoints
    if (!generateTrajectory()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to generate trajectory!");
      return;
    }
    
    // Start trajectory sampling
    trajectory_active_ = true;
    trajectory_generated_ = true;
    current_sample_time_ = 0.0;
    
    // Create sampling timer
    sampling_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(sampling_dt_),
      std::bind(&TrajectoryPlannerNode::samplingTimerCallback, this));
    
    RCLCPP_INFO(this->get_logger(), 
      "Started trajectory execution. Duration: %.2f seconds",
      trajectory_.getMaxTime());
  }
  
  bool generateTrajectory() {
    const int dimension = 3;  // 3D trajectory (x, y, z)
    const int derivative_to_optimize = mav_trajectory_generation::derivative_order::SNAP;
    
    // Create vertex vector
    mav_trajectory_generation::Vertex::Vector vertices;
    
    // Start vertex: current position with current velocity
    mav_trajectory_generation::Vertex start(dimension);
    start.makeStartOrEnd(current_position_, derivative_to_optimize);
    start.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY, current_velocity_);
    vertices.push_back(start);
    
    // Skip first waypoint if it's too close to current position (< 5m horizontal distance)
    // This prevents overshoot when starting trajectory
    size_t start_idx = 0;
    if (!waypoints_x_.empty()) {
      Eigen::Vector3d first_waypoint(waypoints_x_[0], waypoints_y_[0], waypoints_z_[0]);
      Eigen::Vector2d current_pos_xy(current_position_.x(), current_position_.y());
      Eigen::Vector2d first_wp_xy(first_waypoint.x(), first_waypoint.y());
      double horizontal_dist = (first_wp_xy - current_pos_xy).norm();
      
      if (horizontal_dist < 5.0) {
        RCLCPP_INFO(this->get_logger(), 
          "Skipping first waypoint (too close: %.2fm), starting from waypoint 2", 
          horizontal_dist);
        start_idx = 1;
      }
    }
    
    // Intermediate waypoints (skip first if too close)
    for (size_t i = start_idx; i < waypoints_x_.size() - 1; ++i) {
      mav_trajectory_generation::Vertex waypoint(dimension);
      Eigen::Vector3d pos(waypoints_x_[i], waypoints_y_[i], waypoints_z_[i]);
      waypoint.addConstraint(mav_trajectory_generation::derivative_order::POSITION, pos);
      vertices.push_back(waypoint);
      
      RCLCPP_DEBUG(this->get_logger(), "Waypoint %zu: [%.2f, %.2f, %.2f]",
        i, pos.x(), pos.y(), pos.z());
    }
    
    // End vertex: last waypoint with zero velocity
    Eigen::Vector3d goal_pos(waypoints_x_.back(), waypoints_y_.back(), waypoints_z_.back());
    mav_trajectory_generation::Vertex end(dimension);
    end.makeStartOrEnd(goal_pos, derivative_to_optimize);
    end.addConstraint(mav_trajectory_generation::derivative_order::VELOCITY, Eigen::Vector3d::Zero());
    vertices.push_back(end);
    
    RCLCPP_INFO(this->get_logger(), "Planning trajectory:");
    RCLCPP_INFO(this->get_logger(), "  Start: [%.2f, %.2f, %.2f]",
      current_position_.x(), current_position_.y(), current_position_.z());
    RCLCPP_INFO(this->get_logger(), "  Goal:  [%.2f, %.2f, %.2f]",
      goal_pos.x(), goal_pos.y(), goal_pos.z());
    
    // Estimate segment times
    std::vector<double> segment_times;
    segment_times = mav_trajectory_generation::estimateSegmentTimes(vertices, max_v_, max_a_);
    
    // Set up polynomial optimization
    mav_trajectory_generation::NonlinearOptimizationParameters parameters;
    const int N = 10;  // Polynomial order
    mav_trajectory_generation::PolynomialOptimizationNonLinear<N> opt(dimension, parameters);
    opt.setupFromVertices(vertices, segment_times, derivative_to_optimize);
    
    // Add velocity and acceleration constraints
    opt.addMaximumMagnitudeConstraint(mav_trajectory_generation::derivative_order::VELOCITY, max_v_);
    opt.addMaximumMagnitudeConstraint(mav_trajectory_generation::derivative_order::ACCELERATION, max_a_);
    
    // Optimize
    opt.optimize();
    
    // Get the trajectory
    opt.getTrajectory(&trajectory_);
    
    RCLCPP_INFO(this->get_logger(), "Trajectory generated successfully");
    RCLCPP_INFO(this->get_logger(), "  Duration: %.2f seconds", trajectory_.getMaxTime());
    RCLCPP_INFO(this->get_logger(), "  Segments: %zu", trajectory_.segments().size());
    
    return true;
  }
  
  void samplingTimerCallback() {
    if (!trajectory_active_ || !trajectory_generated_) {
      return;
    }
    
    if (current_sample_time_ <= trajectory_.getMaxTime()) {
      // Sample trajectory at current time
      mav_msgs::EigenTrajectoryPoint trajectory_point;
      bool success = mav_trajectory_generation::sampleTrajectoryAtTime(
        trajectory_, current_sample_time_, &trajectory_point);
      
      if (!success) {
        RCLCPP_WARN(this->get_logger(), 
          "Failed to sample trajectory at time %.2f", current_sample_time_);
        return;
      }
      
      // Publish desired state
      publishDesiredState(trajectory_point);
      
      // Update sample time
      current_sample_time_ += sampling_dt_;
      
      // Log progress periodically
      if (static_cast<int>(current_sample_time_ / sampling_dt_) % 50 == 0) {
        double progress = (current_sample_time_ / trajectory_.getMaxTime()) * 100.0;
        RCLCPP_INFO(this->get_logger(), 
          "Trajectory progress: %.1f%% | Position: [%.2f, %.2f, %.2f]",
          progress, trajectory_point.position_W.x(), 
          trajectory_point.position_W.y(), trajectory_point.position_W.z());
      }
    } else {
      // Trajectory completed
      RCLCPP_INFO(this->get_logger(), "Trajectory execution completed!");
      
      // Stop the timer
      if (sampling_timer_) {
        sampling_timer_->cancel();
      }
      trajectory_active_ = false;
      
      // Publish trajectory complete message
      auto complete_msg = std_msgs::msg::Empty();
      trajectory_complete_pub_->publish(complete_msg);
      
      RCLCPP_INFO(this->get_logger(), "Published trajectory_complete signal");
    }
  }
  
  void publishDesiredState(const mav_msgs::EigenTrajectoryPoint& trajectory_point) {
    auto msg = trajectory_msgs::msg::MultiDOFJointTrajectoryPoint();
    
    // Position and orientation
    geometry_msgs::msg::Transform transform;
    transform.translation.x = trajectory_point.position_W.x();
    transform.translation.y = trajectory_point.position_W.y();
    transform.translation.z = trajectory_point.position_W.z();
    
    // Use trajectory orientation or default to current yaw
    transform.rotation.x = trajectory_point.orientation_W_B.x();
    transform.rotation.y = trajectory_point.orientation_W_B.y();
    transform.rotation.z = trajectory_point.orientation_W_B.z();
    transform.rotation.w = trajectory_point.orientation_W_B.w();
    
    msg.transforms.push_back(transform);
    
    // Velocity
    geometry_msgs::msg::Twist twist;
    twist.linear.x = trajectory_point.velocity_W.x();
    twist.linear.y = trajectory_point.velocity_W.y();
    twist.linear.z = trajectory_point.velocity_W.z();
    twist.angular.x = trajectory_point.angular_velocity_W.x();
    twist.angular.y = trajectory_point.angular_velocity_W.y();
    twist.angular.z = trajectory_point.angular_velocity_W.z();
    msg.velocities.push_back(twist);
    
    // Acceleration
    geometry_msgs::msg::Twist accel;
    accel.linear.x = trajectory_point.acceleration_W.x();
    accel.linear.y = trajectory_point.acceleration_W.y();
    accel.linear.z = trajectory_point.acceleration_W.z();
    accel.angular.x = trajectory_point.angular_acceleration_W.x();
    accel.angular.y = trajectory_point.angular_acceleration_W.y();
    accel.angular.z = trajectory_point.angular_acceleration_W.z();
    msg.accelerations.push_back(accel);
    
    desired_state_pub_->publish(msg);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryPlannerNode>());
  rclcpp::shutdown();
  return 0;
}
