#include <memory>
#include <string>
#include <chrono>

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include <Eigen/Dense>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace std::chrono_literals;

// Mission states
enum class MissionState {
  IDLE,
  TAKEOFF,
  FINISHED
};

// Helper function to convert state to string
std::string stateToString(MissionState state) {
  switch(state) {
    case MissionState::IDLE: return "IDLE";
    case MissionState::TAKEOFF: return "TAKEOFF";
    case MissionState::FINISHED: return "FINISHED";
    default: return "UNKNOWN";
  }
}

class MissionControlNode : public rclcpp::Node {
private:
  // State machine
  MissionState current_state_;
  
  // ROS communication
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr current_state_sub_;
  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_state_pub_;
  rclcpp::TimerBase::SharedPtr state_machine_timer_;
  
  // Current UAV state
  Eigen::Vector3d current_position_;
  Eigen::Vector3d current_velocity_;
  double current_yaw_;
  double desired_yaw_;
  bool received_current_state_;
  
  // Mission parameters
  double takeoff_altitude_;
  double position_tolerance_;
  double velocity_tolerance_;
  double state_machine_hz_;
  double idle_duration_;
  
  // Timing
  rclcpp::Time state_start_time_;
  rclcpp::Time idle_start_time_;

public:
  MissionControlNode()
  : Node("mission_control_node"),
    current_state_(MissionState::IDLE),
    current_position_(0, 0, 0),
    current_velocity_(0, 0, 0),
    current_yaw_(0.0),
    desired_yaw_(0.0),
    received_current_state_(false)
  {
    // Declare all parameters as REQUIRED (no default values for critical params)
    this->declare_parameter<double>("takeoff_altitude");
    this->declare_parameter<double>("position_tolerance", 0.3);
    this->declare_parameter<double>("velocity_tolerance", 0.1);
    this->declare_parameter<double>("state_machine_hz", 10.0);
    this->declare_parameter<double>("idle_duration", 3.0);
    this->declare_parameter<std::string>("current_state_topic", "current_state_est");
    this->declare_parameter<std::string>("desired_state_topic", "desired_state");
    this->declare_parameter<int>("subscriber_queue_size", 10);
    this->declare_parameter<int>("publisher_queue_size", 10);
    
    // Load parameters
    if (!loadParameters()) {
      RCLCPP_FATAL(this->get_logger(), 
        "Failed to load required parameters. Node cannot start.");
      rclcpp::shutdown();
      return;
    }
    
    // Get topic names and queue sizes
    std::string current_state_topic = this->get_parameter("current_state_topic").as_string();
    std::string desired_state_topic = this->get_parameter("desired_state_topic").as_string();
    int sub_queue_size = this->get_parameter("subscriber_queue_size").as_int();
    int pub_queue_size = this->get_parameter("publisher_queue_size").as_int();
    
    // Initialize subscribers
    current_state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      current_state_topic, sub_queue_size,
      std::bind(&MissionControlNode::onCurrentState, this, std::placeholders::_1));
    
    // Initialize publishers
    desired_state_pub_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
      desired_state_topic, pub_queue_size);
    
    // Initialize state machine timer
    state_machine_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / state_machine_hz_),
      std::bind(&MissionControlNode::stateMachineLoop, this));
    
    // Initialize timing
    state_start_time_ = this->now();
    idle_start_time_ = this->now();
    
    // Log successful initialization
    RCLCPP_INFO(this->get_logger(), "╔════════════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║       Mission Control Node Initialized Successfully       ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Initial State: %s", stateToString(current_state_).c_str());
    RCLCPP_INFO(this->get_logger(), "State Machine Frequency: %.1f Hz", state_machine_hz_);
    RCLCPP_INFO(this->get_logger(), "Mission Parameters:");
    RCLCPP_INFO(this->get_logger(), "  Takeoff Altitude: %.2f m", takeoff_altitude_);
    RCLCPP_INFO(this->get_logger(), "  Position Tolerance: %.2f m", position_tolerance_);
    RCLCPP_INFO(this->get_logger(), "  Velocity Tolerance: %.2f m/s", velocity_tolerance_);
    RCLCPP_INFO(this->get_logger(), "  IDLE Duration: %.1f s", idle_duration_);
    RCLCPP_INFO(this->get_logger(), "Topics:");
    RCLCPP_INFO(this->get_logger(), "  Subscribing: %s", current_state_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Publishing:  %s", desired_state_topic.c_str());
  }

private:
  bool loadParameters() {
    try {
      // Load required takeoff altitude
      takeoff_altitude_ = this->get_parameter("takeoff_altitude").as_double();
      
      // Load optional parameters
      position_tolerance_ = this->get_parameter("position_tolerance").as_double();
      velocity_tolerance_ = this->get_parameter("velocity_tolerance").as_double();
      state_machine_hz_ = this->get_parameter("state_machine_hz").as_double();
      idle_duration_ = this->get_parameter("idle_duration").as_double();
      
      // Validate parameters
      if (takeoff_altitude_ <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Takeoff altitude must be positive!");
        return false;
      }
      
      if (position_tolerance_ <= 0 || velocity_tolerance_ <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Tolerances must be positive!");
        return false;
      }
      
      if (state_machine_hz_ <= 0 || state_machine_hz_ > 1000) {
        RCLCPP_ERROR(this->get_logger(), "Invalid state machine frequency: %.1f Hz", state_machine_hz_);
        return false;
      }
      
      return true;
      
    } catch (const rclcpp::exceptions::ParameterNotDeclaredException& e) {
      RCLCPP_FATAL(this->get_logger(), "Parameter not declared: %s", e.what());
      return false;
    } catch (const rclcpp::ParameterTypeException& e) {
      RCLCPP_FATAL(this->get_logger(), "Parameter type mismatch: %s", e.what());
      return false;
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
    
    // Set desired yaw to 0.0 on first state reception (or use current yaw if you want to maintain initial orientation)
    if (!received_current_state_) {
      desired_yaw_ = 0.0;  // Fixed yaw for stability
    }
    
    received_current_state_ = true;
  }
  
  void publishDesiredState(const Eigen::Vector3d& position, 
                          const Eigen::Vector3d& velocity = Eigen::Vector3d::Zero(),
                          const Eigen::Vector3d& acceleration = Eigen::Vector3d::Zero(),
                          double yaw = 0.0) {
    auto msg = trajectory_msgs::msg::MultiDOFJointTrajectoryPoint();
    
    // Position
    geometry_msgs::msg::Transform transform;
    transform.translation.x = position.x();
    transform.translation.y = position.y();
    transform.translation.z = position.z();
    
    // Yaw to quaternion
    tf2::Quaternion q;
    q.setRPY(0, 0, yaw);
    transform.rotation.x = q.x();
    transform.rotation.y = q.y();
    transform.rotation.z = q.z();
    transform.rotation.w = q.w();
    
    msg.transforms.push_back(transform);
    
    // Velocity
    geometry_msgs::msg::Twist twist;
    twist.linear.x = velocity.x();
    twist.linear.y = velocity.y();
    twist.linear.z = velocity.z();
    msg.velocities.push_back(twist);
    
    // Acceleration
    geometry_msgs::msg::Twist accel;
    accel.linear.x = acceleration.x();
    accel.linear.y = acceleration.y();
    accel.linear.z = acceleration.z();
    msg.accelerations.push_back(accel);
    
    desired_state_pub_->publish(msg);
  }
  
  bool isAtPosition(const Eigen::Vector3d& target_position, double tolerance = -1.0) {
    if (tolerance < 0) tolerance = position_tolerance_;
    double distance = (current_position_ - target_position).norm();
    return distance < tolerance;
  }
  
  bool isVelocityLow(double tolerance = -1.0) {
    if (tolerance < 0) tolerance = velocity_tolerance_;
    double speed = current_velocity_.norm();
    return speed < tolerance;
  }
  
  void transitionToState(MissionState new_state) {
    if (current_state_ != new_state) {
      RCLCPP_INFO(this->get_logger(), "State transition: %s → %s", 
                  stateToString(current_state_).c_str(),
                  stateToString(new_state).c_str());
      current_state_ = new_state;
      state_start_time_ = this->now();
    }
  }
  
  void stateMachineLoop() {
    // Don't process until we receive current state
    if (!received_current_state_) {
      return;
    }
    
    // State machine logic
    switch (current_state_) {
      case MissionState::IDLE:
        handleIdleState();
        break;
        
      case MissionState::TAKEOFF:
        handleTakeoffState();
        break;
        
      case MissionState::FINISHED:
        handleFinishedState();
        break;
        
      default:
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                            "State %s not yet implemented", 
                            stateToString(current_state_).c_str());
        break;
    }
  }
  
  void handleIdleState() {
    // Publish current position to keep drone stable with fixed yaw
    publishDesiredState(current_position_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), desired_yaw_);
    
    // Wait for idle duration before transitioning to takeoff
    auto elapsed = (this->now() - idle_start_time_).seconds();
    
    if (elapsed >= idle_duration_) {
      RCLCPP_INFO(this->get_logger(), 
                  "IDLE duration complete (%.1f s). Starting takeoff sequence...", 
                  elapsed);
      transitionToState(MissionState::TAKEOFF);
    } else {
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "IDLE: Waiting... (%.1f / %.1f s)", 
                          elapsed, idle_duration_);
    }
  }
  
  void handleTakeoffState() {
    // Target position: maintain current x, y, but climb to takeoff altitude
    Eigen::Vector3d target_position(current_position_.x(), 
                                     current_position_.y(), 
                                     takeoff_altitude_);
    
    // Publish desired state with fixed yaw for stability
    publishDesiredState(target_position, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), desired_yaw_);
    
    // Check if we've reached the target altitude
    if (isAtPosition(target_position) && isVelocityLow()) {
      RCLCPP_INFO(this->get_logger(), 
                  "Takeoff complete! Reached altitude: %.2f m", 
                  current_position_.z());
      RCLCPP_INFO(this->get_logger(), "Mission complete - transitioning to FINISHED state");
      transitionToState(MissionState::FINISHED);
    } else {
      double distance_to_target = (current_position_ - target_position).norm();
      RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                          "TAKEOFF: Current altitude: %.2f m | Target: %.2f m | Distance: %.2f m | Yaw: %.2f°",
                          current_position_.z(), takeoff_altitude_, distance_to_target, current_yaw_ * 180.0 / M_PI);
    }
  }
  
  void handleFinishedState() {
    // Hold current position with fixed yaw
    publishDesiredState(current_position_, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), desired_yaw_);
    
    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                        "Mission FINISHED. Holding position at [%.2f, %.2f, %.2f] | Yaw: %.2f°",
                        current_position_.x(), current_position_.y(), current_position_.z(), 
                        current_yaw_ * 180.0 / M_PI);
  }
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MissionControlNode>());
  rclcpp::shutdown();
  return 0;
}
