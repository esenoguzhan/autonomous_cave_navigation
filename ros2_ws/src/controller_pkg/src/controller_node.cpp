#include <cmath>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include <mav_msgs/msg/actuators.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>

#include <Eigen/Dense>
#include <tf2/utils.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#define PI M_PI

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//  PART 0 |  Autonomous Systems - Fall 2025  - Lab 2 coding assignment
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//
//  In this code, we ask you to implement a geometric controller for a
//  simulated UAV, following the publication:
//
//  [1] Lee, Taeyoung, Melvin Leoky, N. Harris McClamroch. "Geometric tracking
//      control of a quadrotor UAV on SE (3)." Decision and Control (CDC),
//      49th IEEE Conference on. IEEE, 2010
//
//  We use variable names as close as possible to the conventions found in the
//  paper, however, we have slightly different conventions for the aerodynamic
//  coefficients of the propellers (refer to the lecture notes for these).
//  Additionally, watch out for the different conventions on reference frames
//  (see Lab Handout for more details).
//
//  Eigen is a C++ library for linear algebra that will help you significantly 
//  with the implementation. Check the reference page to learn the basics:
//
//  https://eigen.tuxfamily.org/dox/group__QuickRefPage.html
//
// ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
//                                 end part 0
// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class ControllerNode : public rclcpp::Node {
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  //  PART 1 |  Declare ROS callback handlers
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //
  // In this section, you need to declare:
  //   1. two subscribers (for the desired and current UAVStates)
  //   2. one publisher (for the propeller speeds)
  //   3. a timer for your main control loop
  //
  // ~~~~ begin solution

  // Subscribers for desired and current UAV states
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_state_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr current_state_sub_;
  
  // Publisher for propeller speeds
  rclcpp::Publisher<mav_msgs::msg::Actuators>::SharedPtr rotor_pub_;
  
  // Timer for main control loop
  rclcpp::TimerBase::SharedPtr control_timer_;

  // ~~~~ end solution
  // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
  //                                 end part 1
  // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  // Controller parameters
  double kx, kv, kr, komega; // controller gains - [1] eq (15), (16)

  // Physical constants (we will set them below)
  double m;              // mass of the UAV
  double g;              // gravity acceleration
  double d;              // distance from the center of propellers to the c.o.m.
  double cf,             // Propeller lift coefficient
         cd;             // Propeller drag coefficient
  Eigen::Matrix3d J;     // Inertia Matrix
  Eigen::Vector3d e3;    // [0,0,1]
  Eigen::MatrixXd F2W;   // Wrench-rotor speeds map

  // Controller internals (you will have to set them below)
  // Current state
  Eigen::Vector3d x;     // current position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d v;     // current velocity of the UAV's c.o.m. in the world frame
  Eigen::Matrix3d R;     // current orientation of the UAV
  Eigen::Vector3d omega; // current angular velocity of the UAV's c.o.m. in the *body* frame

  // Desired state
  Eigen::Vector3d xd;    // desired position of the UAV's c.o.m. in the world frame
  Eigen::Vector3d vd;    // desired velocity of the UAV's c.o.m. in the world frame
  Eigen::Vector3d ad;    // desired acceleration of the UAV's c.o.m. in the world frame
  double yawd;           // desired yaw angle

  double hz;             // frequency of the main control loop
  
  bool received_desired_state_;  // flag to check if we've received desired state

  static Eigen::Vector3d Vee(const Eigen::Matrix3d& in){
    Eigen::Vector3d out;
    out << in(2,1), in(0,2), in(1,0);
    return out;
  }

  static double signed_sqrt(double val){
    return val>0?sqrt(val):-sqrt(-val);
  }

public:
  ControllerNode()
  : rclcpp::Node("controller_node"),
    e3(0,0,1),
    F2W(4,4),
    received_desired_state_(false)
  {
    // Declare all parameters as REQUIRED (no default values)
    this->declare_parameter<double>("kx");
    this->declare_parameter<double>("kv");
    this->declare_parameter<double>("kr");
    this->declare_parameter<double>("komega");
    this->declare_parameter<double>("control_loop_hz");
    this->declare_parameter<double>("mass");
    this->declare_parameter<double>("gravity");
    this->declare_parameter<double>("arm_length");
    this->declare_parameter<double>("lift_coefficient");
    this->declare_parameter<double>("drag_coefficient");
    this->declare_parameter<double>("inertia_xx");
    this->declare_parameter<double>("inertia_yy");
    this->declare_parameter<double>("inertia_zz");
    this->declare_parameter<std::string>("desired_state_topic");
    this->declare_parameter<std::string>("current_state_topic");
    this->declare_parameter<std::string>("rotor_speed_cmd_topic");
    this->declare_parameter<int>("subscriber_queue_size");
    this->declare_parameter<int>("publisher_queue_size");

    // Load controller gains from parameters
    if (!loadParameters()) {
      RCLCPP_FATAL(this->get_logger(), 
        "Failed to load required parameters. Node cannot start.");
      rclcpp::shutdown();
      return;
    }

    // Get queue sizes and topic names
    int sub_queue_size = this->get_parameter("subscriber_queue_size").as_int();
    int pub_queue_size = this->get_parameter("publisher_queue_size").as_int();
    std::string desired_state_topic = this->get_parameter("desired_state_topic").as_string();
    std::string current_state_topic = this->get_parameter("current_state_topic").as_string();
    std::string rotor_cmd_topic = this->get_parameter("rotor_speed_cmd_topic").as_string();

    // Initialize subscribers
    desired_state_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
      desired_state_topic, sub_queue_size, 
      std::bind(&ControllerNode::onDesiredState, this, std::placeholders::_1));
    
    current_state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      current_state_topic, sub_queue_size,
      std::bind(&ControllerNode::onCurrentState, this, std::placeholders::_1));
    
    // Initialize publisher
    rotor_pub_ = this->create_publisher<mav_msgs::msg::Actuators>(rotor_cmd_topic, pub_queue_size);
    
    // Initialize timer for control loop
    control_timer_ = this->create_wall_timer(
      std::chrono::duration<double>(1.0 / hz),
      std::bind(&ControllerNode::controlLoop, this));

    // Log successful initialization
    RCLCPP_INFO(this->get_logger(), "╔════════════════════════════════════════════════════════════╗");
    RCLCPP_INFO(this->get_logger(), "║       Controller Node Initialized Successfully            ║");
    RCLCPP_INFO(this->get_logger(), "╚════════════════════════════════════════════════════════════╝");
    RCLCPP_INFO(this->get_logger(), "Control Loop Frequency: %.1f Hz", hz);
    RCLCPP_INFO(this->get_logger(), "Controller Gains:");
    RCLCPP_INFO(this->get_logger(), "  kx=%.2f, kv=%.2f, kr=%.2f, komega=%.2f", kx, kv, kr, komega);
    RCLCPP_INFO(this->get_logger(), "Physical Properties:");
    RCLCPP_INFO(this->get_logger(), "  mass=%.2f kg, gravity=%.2f m/s^2, arm_length=%.2f m", m, g, d);
    RCLCPP_INFO(this->get_logger(), "Topics:");
    RCLCPP_INFO(this->get_logger(), "  Subscribing: %s", desired_state_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Subscribing: %s", current_state_topic.c_str());
    RCLCPP_INFO(this->get_logger(), "  Publishing:  %s", rotor_cmd_topic.c_str());
  }

private:
  bool loadParameters() {
    try {
      // Load controller gains
      kx = this->get_parameter("kx").as_double();
      kv = this->get_parameter("kv").as_double();
      kr = this->get_parameter("kr").as_double();
      komega = this->get_parameter("komega").as_double();
      
      // Load control loop frequency
      hz = this->get_parameter("control_loop_hz").as_double();
      
      // Load physical constants
      m = this->get_parameter("mass").as_double();
      g = this->get_parameter("gravity").as_double();
      d = this->get_parameter("arm_length").as_double();
      cf = this->get_parameter("lift_coefficient").as_double();
      cd = this->get_parameter("drag_coefficient").as_double();
      
      // Load inertia matrix
      double Jxx = this->get_parameter("inertia_xx").as_double();
      double Jyy = this->get_parameter("inertia_yy").as_double();
      double Jzz = this->get_parameter("inertia_zz").as_double();
      J << Jxx, 0.0, 0.0,
           0.0, Jyy, 0.0,
           0.0, 0.0, Jzz;

      // Validate parameters
      if (kx <= 0 || kv <= 0 || kr <= 0 || komega <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Controller gains must be positive!");
        return false;
      }
      
      if (hz <= 0 || hz > 10000) {
        RCLCPP_ERROR(this->get_logger(), "Invalid control loop frequency: %.1f Hz", hz);
        return false;
      }
      
      if (m <= 0 || g <= 0 || d <= 0 || cf <= 0 || cd <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Physical constants must be positive!");
        return false;
      }
      
      if (Jxx <= 0 || Jyy <= 0 || Jzz <= 0) {
        RCLCPP_ERROR(this->get_logger(), "Inertia values must be positive!");
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

public:

  void onDesiredState(const trajectory_msgs::msg::MultiDOFJointTrajectoryPoint::SharedPtr des_state_msg){
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  PART 3 | Objective: fill in xd, vd, ad, yawd
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //
      // 3.1 Get the desired position, velocity and acceleration from the in-
      //     coming ROS message and fill in the class member variables xd, vd
      //     and ad accordingly. You can ignore the angular acceleration.
      //
      // Hint: use "v << vx, vy, vz;" to fill in a vector with Eigen.
      //
      // ~~~~ begin solution
      
      // 3.1 Extract position, velocity and acceleration
      xd << des_state_msg->transforms[0].translation.x,
            des_state_msg->transforms[0].translation.y,
            des_state_msg->transforms[0].translation.z;
      
      vd << des_state_msg->velocities[0].linear.x,
            des_state_msg->velocities[0].linear.y,
            des_state_msg->velocities[0].linear.z;
      
      ad << des_state_msg->accelerations[0].linear.x,
            des_state_msg->accelerations[0].linear.y,
            des_state_msg->accelerations[0].linear.z;

      // ~~~~ end solution
      //
      // 3.2 Extract the yaw component from the quaternion in the incoming ROS
      //     message and store in the yawd class member variable
      //
      //  Hints:
      //    - use tf2::getYaw(des_state_msg->transforms[0].rotation)
      //
      // ~~~~ begin solution
      
      // 3.2 Extract yaw from quaternion
      yawd = tf2::getYaw(des_state_msg->transforms[0].rotation);
      
      // ~~~~ end solution
      
      // Mark that we've received desired state
      received_desired_state_ = true;
  }

  void onCurrentState(const nav_msgs::msg::Odometry::SharedPtr cur_state_msg){
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //  PART 4 | Objective: fill in x, v, R and omega
      // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      //
      // Get the current position and velocity from the incoming ROS message and
      // fill in the class member variables x, v, R and omega accordingly.
      //
      //  CAVEAT: cur_state.twist.twist.angular is in the world frame, while omega
      //          needs to be in the body frame!
      //
      // ~~~~ begin solution
      
      // Extract current position
      x << cur_state_msg->pose.pose.position.x,
           cur_state_msg->pose.pose.position.y,
           cur_state_msg->pose.pose.position.z;
      
      // Extract current velocity
      v << cur_state_msg->twist.twist.linear.x,
           cur_state_msg->twist.twist.linear.y,
           cur_state_msg->twist.twist.linear.z;
      
      // Extract current orientation (convert quaternion to rotation matrix)
      Eigen::Quaterniond q(cur_state_msg->pose.pose.orientation.w,
                           cur_state_msg->pose.pose.orientation.x,
                           cur_state_msg->pose.pose.orientation.y,
                           cur_state_msg->pose.pose.orientation.z);
      R = q.toRotationMatrix();
      
      // Extract angular velocity and convert from world frame to body frame
      Eigen::Vector3d omega_world;
      omega_world << cur_state_msg->twist.twist.angular.x,
                     cur_state_msg->twist.twist.angular.y,
                     cur_state_msg->twist.twist.angular.z;
      omega = R.transpose() * omega_world;

      // ~~~~ end solution
  }

  void controlLoop(){
    // Don't publish control commands until we've received a desired state
    if (!received_desired_state_) {
      return;
    }
    
    Eigen::Vector3d ex, ev, er, eomega;

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 5 | Objective: Implement the controller!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // 5.1 Compute position and velocity errors. Objective: fill in ex, ev.
    //  Hint: [1], eq. (6), (7)
    //
    // ~~~~ begin solution
    
    // 5.1 Compute position and velocity errors - [1] eq. (6), (7)
    ex = x - xd;      // Position error: current - desired
    ev = v - vd;      // Velocity error: current - desired
    
    // ~~~~ end solution

    // 5.2 Compute the Rd matrix.
    //
    //  Hint: break it down in 3 parts:
    //    - b3d vector = z-body axis of the quadrotor, [1] eq. (12)
    //    - check out [1] fig. (3) for the remaining axes [use cross product]
    //    - assemble the Rd matrix, eigen offers: "MATRIX << col1, col2, col3"
    //
    //  CAVEATS:
    //    - Compare the reference frames in the Lab handout with Fig. 1 in the
    //      paper. The z-axes are flipped, which affects the signs of:
    //         i) the gravity term and
    //        ii) the overall sign (in front of the fraction) in equation (12)
    //            of the paper
    //    - remember to normalize your axes!
    //
    // ~~~~ begin solution
    
    // 5.2 Compute the desired rotation matrix Rd - [1] eq. (12)
    
    // Step 1: Compute b3d (desired z-body axis) - [1] eq. (12)
    // Note: Z-axes are flipped compared to paper, so we use +g instead of -g
    Eigen::Vector3d b3d = (-kx * ex - kv * ev + m * g * e3 + m * ad);
    b3d.normalize();  // Normalize to unit vector
    
    // Step 2: Compute b1d (desired x-body axis) using yaw
    Eigen::Vector3d b1c(cos(yawd), sin(yawd), 0);  // Direction in yaw
    Eigen::Vector3d b1d = b1c.cross(b3d);
    b1d.normalize();  // Normalize to unit vector
    
    // Step 3: Compute b2d (desired y-body axis) using right-hand rule
    Eigen::Vector3d b2d = b3d.cross(b1d);
    // b2d is automatically normalized since b3d and b1d are orthonormal
    
    // Step 4: Assemble the desired rotation matrix
    Eigen::Matrix3d Rd;
    Rd << b1d, b2d, b3d;  // [b1d | b2d | b3d] as columns

    // ~~~~ end solution
    //
    // 5.3 Compute the orientation error (er) and the rotation-rate error (eomega)
    //  Hints:
    //     - [1] eq. (10) and (11)
    //     - you can use the Vee() static method implemented above
    //
    //  CAVEATS: feel free to ignore the second addend in eq (11), since it
    //        requires numerical differentiation of Rd and it has negligible
    //        effects on the closed-loop dynamics.
    //
    // ~~~~ begin solution
    
    // 5.3 Compute orientation and angular velocity errors - [1] eq. (10), (11)
    
    // Orientation error - [1] eq. (10)
    Eigen::Matrix3d eR_matrix = 0.5 * (Rd.transpose() * R - R.transpose() * Rd);
    er = Vee(eR_matrix);
    
    // Angular velocity error - [1] eq. (11) (simplified, ignoring Omega_d terms)
    eomega = -omega;  // Assuming desired angular velocity is zero
    
    // ~~~~ end solution
    //
    // 5.4 Compute the desired wrench (force + torques) to control the UAV.
    //  Hints:
    //     - [1] eq. (15), (16)
    //
    // CAVEATS:
    //    - Compare the reference frames in the Lab handout with Fig. 1 in the
    //      paper. The z-axes are flipped, which affects the signs of:
    //         i) the gravity term
    //        ii) the overall sign (in front of the bracket) in equation (15)
    //            of the paper
    //
    //    - feel free to ignore all the terms involving \Omega_d and its time
    //      derivative as they are of the second order and have negligible
    //      effects on the closed-loop dynamics.
    //
    // ~~~~ begin solution
    
    // 5.4 Compute desired wrench (force + torques) - [1] eq. (15), (16)
    
    // Compute desired total thrust force - [1] eq. (15)
    // Note: Z-axis flipped, so we use +mg instead of -mg, and + sign instead of -
    double f = (-kx * ex - kv * ev + m * g * e3 + m * ad).dot(R * e3);
    
    // Compute desired torques - [1] eq. (16) (simplified, ignoring Omega_d terms)
    Eigen::Vector3d M = -kr * er - komega * eomega + omega.cross(J * omega);
    
    // Assemble wrench vector [force; torques]
    Eigen::Vector4d wrench;
    wrench << f, M(0), M(1), M(2);
    
    // ~~~~ end solution
    //
    // 5.5 Recover the rotor speeds from the wrench computed above
    //
    //  Hints:
    //     - [1] eq. (1)
    //
    // CAVEATs:
    //     - we have different conventions for the arodynamic coefficients,
    //       Namely: C_{\tau f} = c_d / c_f
    //               (LHS paper [1], RHS our conventions [lecture notes])
    //
    //     - Compare the reference frames in the Lab handout with Fig. 1 in the
    //       paper. In the paper [1], the x-body axis [b1] is aligned with a
    //       quadrotor arm, whereas for us, it is 45° from it (i.e., "halfway"
    //       between b1 and b2). To resolve this, check out equation 6.9 in the
    //       lecture notes!
    //
    //     - The thrust forces are **in absolute value** proportional to the
    //       square of the propeller speeds. Negative propeller speeds - although
    //       uncommon - should be a possible outcome of the controller when
    //       appropriate. Note that this is the case in unity but not in real
    //       life, where propellers are aerodynamically optimized to spin in one
    //       direction!
    //
    // ~~~~ begin solution
    
    // 5.5 Recover rotor speeds from wrench - [1] eq. (1) with frame adjustments
    
    // Setup wrench-to-rotor mapping matrix F2W according to our frame convention
    // Note: 45° rotation from paper's frame convention (equation 6.9 in lecture notes)
    double sqrt2_over_2 = sqrt(2.0) / 2.0;
    double tau_f = cd / cf;  // Drag-to-lift ratio (our convention)
    
    // F2W << cf,        cf,        cf,        cf,
    //        cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2, cf*d*sqrt2_over_2,
    //       -cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2,  cf*d*sqrt2_over_2, cf*d*sqrt2_over_2,
    //       -cd,        cd,       -cd,        cd;

    F2W << cf,        cf,        cf,        cf,
           cf*d*sqrt2_over_2, cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2,
          -cf*d*sqrt2_over_2, cf*d*sqrt2_over_2,  cf*d*sqrt2_over_2, -cf*d*sqrt2_over_2,
          cd,        -cd,        cd,        -cd;
    
    // Solve for squared rotor speeds: F2W * w^2 = wrench
    Eigen::Vector4d w_squared = F2W.colPivHouseholderQr().solve(wrench);
    
    // Extract individual rotor speeds (with sign preservation)
    double w1 = signed_sqrt(w_squared(0));
    double w2 = signed_sqrt(w_squared(1));
    double w3 = signed_sqrt(w_squared(2));
    double w4 = signed_sqrt(w_squared(3));
    
    // ~~~~ end solution
    //
    // 5.6 Populate and publish the control message
    //
    // Hint: do not forget that the propeller speeds are signed (maybe you want
    // to use signed_sqrt function).
    //
    // ~~~~ begin solution
    
    // 5.6 Create and publish control message with computed rotor speeds
    mav_msgs::msg::Actuators cmd;
    cmd.angular_velocities.resize(4);
    cmd.angular_velocities[0] = w1;  // Rotor 1 speed
    cmd.angular_velocities[1] = w2;  // Rotor 2 speed  
    cmd.angular_velocities[2] = w3;  // Rotor 3 speed
    cmd.angular_velocities[3] = w4;  // Rotor 4 speed
    
    // Publish the control command
    rotor_pub_->publish(cmd);
    
    // ~~~~ end solution

    // Example publish skeleton (keep after you compute rotor speeds):
    // mav_msgs::msg::Actuators cmd;
    // cmd.angular_velocities.resize(4);
    // cmd.angular_velocities[0] = /* w1 */;
    // cmd.angular_velocities[1] = /* w2 */;
    // cmd.angular_velocities[2] = /* w3 */;
    // cmd.angular_velocities[3] = /* w4 */;
    // motor_pub_->publish(cmd);
  }
};

int main(int argc, char** argv){
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<ControllerNode>());
  rclcpp::shutdown();
  return 0;
}
