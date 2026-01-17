#include <cmath>
#include <memory>
#include <string>
#include <iostream>

#include <rclcpp/rclcpp.hpp>

#include <mav_msgs/msg/actuators.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>


#include <Eigen/Dense>
#include <tf2/utils.h>

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
  //
   // ~~~~ begin solution
  //
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_state_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr current_state_sub_;
  rclcpp::Publisher<mav_msgs::msg::Actuators>::SharedPtr motor_pub_;
  rclcpp::TimerBase::SharedPtr control_timer_;
  //
  // ~~~~ end solution
  //
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
  
  bool got_current_{false};
  bool got_desired_{false};


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
  hz(1000.0)
{
  x.setZero();
  v.setZero();
  omega.setZero();
  xd.setZero();
  vd.setZero();
  ad.setZero();
  R.setIdentity();
  yawd = 0.0;


    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 2 |  Initialize ROS callback handlers
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~  ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //
    // In this section, you need to initialize your handlers from part 1.
    // Specifically:
    //  - bind controllerNode::onDesiredState() to the topic "desired_state"
    //  - bind controllerNode::onCurrentState() to the topic "current_state"
    //  - bind controllerNode::controlLoop() to the created timer, at frequency
    //    given by the "hz" variable
    //
    // Hints:
    //  - read the lab handout to find the message type
    //
    // ~~~~ begin solution
    //
        // ~~~~ begin solution
    //
    // Subscriber for desired state
    desired_state_sub_ = this->create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
      "desired_state", rclcpp::QoS(10),
      std::bind(&ControllerNode::onDesiredState, this, std::placeholders::_1));
    
    // Subscriber for current state
    current_state_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "current_state", rclcpp::QoS(10),
      std::bind(&ControllerNode::onCurrentState, this, std::placeholders::_1));
    
    // Publisher for rotor speed commands
    motor_pub_ = this->create_publisher<mav_msgs::msg::Actuators>(
      "rotor_speed_cmds", rclcpp::QoS(10));
    
    // Timer for control loop (hz frequency)
    control_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(static_cast<int>(1000.0 / hz)),
      std::bind(&ControllerNode::controlLoop, this));
    //
    // ~~~~ end solution
    //
    // ~~~~ end solution
    //
    // ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
    //                                 end part 2
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 6 [NOTE: save this for last] |  Tune your gains!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Controller gains (move to parameters.yaml for convenience if you like)
    kx = 1.8;
    kv = 1.45;             //    **** FIDDLE WITH THESE! ***
    kr = 6.5;
    komega = 0.8;

    // Initialize constants
    m = 1.0;
    cd = 1e-5;
    cf = 1e-3;
    g = 9.81;
    d = 0.3;
    J << 1.0,0.0,0.0,
         0.0,1.0,0.0,
         0.0,0.0,1.0;

    RCLCPP_INFO(this->get_logger(), "controller_node ready (hz=%.1f)", hz);
  }

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
      //
      double xdx = des_state_msg->transforms[0].translation.x;
      double xdy = des_state_msg->transforms[0].translation.y;
      double xdz = des_state_msg->transforms[0].translation.z;
      xd << xdx, xdy, xdz;

      double vdx = des_state_msg->velocities[0].linear.x;
      double vdy = des_state_msg->velocities[0].linear.y;
      double vdz = des_state_msg->velocities[0].linear.z;
      vd << vdx, vdy, vdz;

      double adx = des_state_msg->accelerations[0].linear.x;
      double ady = des_state_msg->accelerations[0].linear.y;
      double adz = des_state_msg->accelerations[0].linear.z;
      ad << adx, ady, adz;
      //
      // ~~~~ end solution
      //
      // 3.2 Extract the yaw component from the quaternion in the incoming ROS
      //     message and store in the yawd class member variable
      //
      //  Hints:
      //    - use tf2::getYaw(des_state_msg->transforms[0].rotation)
      yawd = tf2::getYaw(des_state_msg->transforms[0].rotation);
      //
      // ~~~~ begin solution
      //
      //     **** FILL IN HERE ***
      // ~~~~ end solution
      got_desired_ = true;

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
      //
      //     **** FILL IN HERE ***
      double xx = cur_state_msg->pose.pose.position.x;
      double yy = cur_state_msg->pose.pose.position.y;
      double zz = cur_state_msg->pose.pose.position.z;
      x << xx, yy, zz;

      double vx = cur_state_msg->twist.twist.linear.x;
      double vy = cur_state_msg->twist.twist.linear.y;
      double vz = cur_state_msg->twist.twist.linear.z;
      v << vx, vy, vz;

      // From quaternion to rotation matrix
      Eigen::Quaterniond q(
        cur_state_msg->pose.pose.orientation.w,
        cur_state_msg->pose.pose.orientation.x,
        cur_state_msg->pose.pose.orientation.y,
        cur_state_msg->pose.pose.orientation.z
      );
      R = q.toRotationMatrix();  // Converts quaternion to 3x3 rotation matrix

      Eigen::Vector3d omega_world;

      double wx = cur_state_msg->twist.twist.angular.x;
      double wy = cur_state_msg->twist.twist.angular.y;
      double wz = cur_state_msg->twist.twist.angular.z;
      omega_world << wx, wy, wz;

      omega = R.transpose() * omega_world;


      




      //
      // ~~~~ end solution
      got_current_ = true;

  }

  void controlLoop(){
  
    Eigen::Vector3d ex, ev, er, eomega;
    if (!got_current_ || !got_desired_) {
      RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, 
    "Waiting for state/trajectory messages... Got Current: %d, Got Desired: %d", 
    got_current_, got_desired_);
      return;
    }


    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //  PART 5 | Objective: Implement the controller!
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //
    // 5.1 Compute position and velocity errors. Objective: fill in ex, ev.
    //  Hint: [1], eq. (6), (7)
    //
    // ~~~~ begin solution
    //
    ex = x - xd;
    ev = v - vd;
    //
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
    //
    Eigen::Vector3d b3d;
    Eigen::Vector3d b2d;
    Eigen::Vector3d b1d;
    Eigen::Vector3d b1d_projected;
    Eigen::Matrix3d Rd; 

    b3d = (-kx * ex - kv *ev + m*g*e3 + m*ad).normalized();
    b1d << cos(yawd), sin(yawd), 0;
    b2d = b3d.cross(b1d).normalized();
    b1d_projected = b2d.cross(b3d).normalized();
    Rd << b1d_projected, b2d, b3d;

    




   
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
    //
    er = 0.5 * Vee(Rd.transpose() * R - R.transpose() * Rd);
    eomega = omega;

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
    "DEBUG Errors | |ex|=%.4f |ev|=%.4f |er|=%.4f |eomega|=%.4f",
    ex.norm(), ev.norm(), er.norm(), eomega.norm());
    //
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
    //
    Eigen::Vector3d f_world;
    Eigen::Vector3d M;

    f_world = -kx * ex - kv * ev + m*g*e3 + m*ad;
    double f = f_world.dot(R.col(2));

    M = -kr * er - komega * eomega + omega.cross(J*omega);
  
    //
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
    //
    Eigen::Vector4d M1, M2, M3, f_row;
    f_row << 1,1,1,1;
    M1 << d*sin(M_PI/4), d*cos(M_PI/4), -d * sin(M_PI/4), -d * cos(M_PI/4);
    M2 << -d*cos(M_PI/4), d*sin(M_PI/4), d * cos(M_PI/4), -d * sin(M_PI/4);
    M3 << cd/cf , -cd/cf , cd/cf , -cd/cf ;
    F2W << f_row.transpose(), 
           M1.transpose(), 
           M2.transpose(), 
           M3.transpose();

    Eigen::Vector4d wrench;
    wrench << f,      // Element 0: force (scalar from part 5.4)
            M(0),    // Element 1: torque x
            M(1),    // Element 2: torque y
            M(2);    // Element 3: torque z
    
    Eigen::Vector4d motor_thrusts = F2W.inverse() * wrench;

    
    Eigen::Vector4d w_squared = motor_thrusts / cf;

    // Extract rotor speeds (with sign)
    double w1 = signed_sqrt(w_squared(0));
    double w2 = signed_sqrt(w_squared(1));
    double w3 = signed_sqrt(w_squared(2));
    double w4 = signed_sqrt(w_squared(3));

    
    //
    // ~~~~ end solution
    //
    // 5.6 Populate and publish the control message
    //
    // Hint: do not forget that the propeller speeds are signed (maybe you want
    // to use signed_sqrt function).
    //
    // ~~~~ begin solution
    //
    // ~~~~ begin solution
//
    mav_msgs::msg::Actuators cmd;
    cmd.angular_velocities.resize(4);
    cmd.angular_velocities[0] = w1;
    cmd.angular_velocities[1] = w2;
    cmd.angular_velocities[2] = w3;
    cmd.angular_velocities[3] = w4;
    motor_pub_->publish(cmd);
//
// ~~~~ end solution
    //
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
