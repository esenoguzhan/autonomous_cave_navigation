#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include "trajectory_planner/srv/execute_trajectory.hpp"

using namespace std::chrono_literals;

class TrajectoryGenerator : public rclcpp::Node {
public:
    TrajectoryGenerator() : Node("trajectory_generator") {
        path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "rrt_path", 1, std::bind(&TrajectoryGenerator::pathCallback, this, std::placeholders::_1));

        odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "current_state_est", 10, std::bind(&TrajectoryGenerator::odomCallback, this, std::placeholders::_1));

        desired_pub_   = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>("desired_state", 10);
        complete_pub_  = this->create_publisher<std_msgs::msg::Empty>("trajectory_complete", 10);
        traj_path_pub_ = this->create_publisher<nav_msgs::msg::Path>("planned_trajectory", 10);

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1, std::bind(&TrajectoryGenerator::stateCallback, this, std::placeholders::_1));

        start_navigation_service_ = this->create_service<trajectory_planner::srv::ExecuteTrajectory>(
            "start_navigation", std::bind(&TrajectoryGenerator::startNavigationCallback, this,
                                          std::placeholders::_1, std::placeholders::_2));

        timer_ = this->create_wall_timer(50ms, std::bind(&TrajectoryGenerator::timerCallback, this)); // 20Hz

        RCLCPP_INFO(this->get_logger(), "Trajectory Generator Initialized (Quintic, Continuity-Aware)");
    }

private:
    // ── Subscribers & Publishers ──────────────────────────────────────────────
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr     path_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr   state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_pub_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr       complete_pub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr        traj_path_pub_;
    rclcpp::Service<trajectory_planner::srv::ExecuteTrajectory>::SharedPtr start_navigation_service_;
    rclcpp::TimerBase::SharedPtr timer_;

    // ── Trajectory state ─────────────────────────────────────────────────────
    std::string stm_state_  = "IDLE";
    bool has_coeffs_        = false;
    bool finished_pub_      = false;
    rclcpp::Time start_time_;
    double total_time_      = 0.0;

    // Quintic coefficients per segment (6 elements each)
    std::vector<double>         segTimes_;
    std::vector<double>         cumTimes_;
    std::vector<Eigen::VectorXd> cxs_, cys_, czs_;

    // Last goal endpoint (for debounce)
    Eigen::Vector3d last_goal_ = Eigen::Vector3d::Zero();
    bool goal_set_ = false;

    // ── Odometry (current velocity & acceleration) ───────────────────────────
    Eigen::Vector3d current_velocity_     = Eigen::Vector3d::Zero();
    Eigen::Vector3d current_acceleration_ = Eigen::Vector3d::Zero();
    Eigen::Vector3d last_velocity_        = Eigen::Vector3d::Zero();
    rclcpp::Time    last_odom_time_;
    bool odom_received_ = false;

    // =========================================================================
    // Odometry callback – estimates acceleration via finite difference
    // =========================================================================
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        Eigen::Vector3d new_vel(
            msg->twist.twist.linear.x,
            msg->twist.twist.linear.y,
            msg->twist.twist.linear.z);

        if (odom_received_) {
            double dt = (this->now() - last_odom_time_).seconds();
            if (dt > 1e-3)
                current_acceleration_ = (new_vel - current_velocity_) / dt;
        }
        current_velocity_ = new_vel;
        last_odom_time_   = this->now();
        odom_received_    = true;
    }

    void stateCallback(const std_msgs::msg::String::SharedPtr msg) {
        stm_state_ = msg->data;
    }

    // =========================================================================
    // Service: start_navigation  (NAVIGATE_TO_CAVE phase)
    // =========================================================================
    void startNavigationCallback(
        const std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Request>  request,
        std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Response> response)
    {
        if (request->waypoints.poses.size() < 2) {
            response->success = false;
            response->message = "Need at least 2 waypoints";
            return;
        }

        nav_msgs::msg::Path path;
        path.poses.resize(request->waypoints.poses.size());
        for (size_t i = 0; i < request->waypoints.poses.size(); ++i)
            path.poses[i].pose = request->waypoints.poses[i];

        buildTrajectory(path);

        response->success = true;
        response->message = "Trajectory started (Quintic Spline, Continuity-Aware)";
    }

    // =========================================================================
    // RRT path callback – debounced: only rebuild if goal changed > 2m
    // =========================================================================
    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.size() < 2) return;
        if (stm_state_ != "EXPLORE_CAVE" && stm_state_ != "EXPLORE") return;

        // Debounce: ignore if active trajectory and goal hasn't moved much
        const auto& last_pose = msg->poses.back().pose.position;
        Eigen::Vector3d new_goal(last_pose.x, last_pose.y, last_pose.z);

        if (goal_set_ && has_coeffs_ && !finished_pub_) {
            double dist = (new_goal - last_goal_).norm();
            if (dist < 2.0) {
                RCLCPP_DEBUG(this->get_logger(),
                    "Goal unchanged (%.2fm) – keeping active trajectory.", dist);
                return;
            }
            RCLCPP_INFO(this->get_logger(),
                "Goal shifted %.2fm – rebuilding trajectory.", dist);
        }

        last_goal_ = new_goal;
        goal_set_  = true;
        buildTrajectory(*msg);
    }

    // =========================================================================
    // Quintic polynomial: solve for c[0..5] given boundary conditions
    //   p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
    //   boundary: pos, vel, acc at t=0 and t=T
    // =========================================================================
    Eigen::VectorXd computeQuinticCoeffs(
        double p0, double v0, double a0,
        double pT, double vT, double aT,
        double T)
    {
        Eigen::VectorXd c(6);
        c[0] = p0;
        c[1] = v0;
        c[2] = a0 / 2.0;

        double T2 = T*T, T3 = T2*T, T4 = T3*T, T5 = T4*T;

        Eigen::Matrix3d A;
        A <<   T3,    T4,    T5,
             3*T2,  4*T3,  5*T4,
              6*T, 12*T2, 20*T3;

        Eigen::Vector3d b;
        b(0) = pT - p0 - v0*T - (a0/2.0)*T2;
        b(1) = vT - v0 - a0*T;
        b(2) = aT - a0;

        Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
        c[3] = x(0); c[4] = x(1); c[5] = x(2);
        return c;
    }

    // Evaluate quintic (or its derivatives) at time t
    double evalQuintic(const Eigen::VectorXd& c, double t, int order) {
        switch (order) {
            case 0:
                return c[0] + c[1]*t + c[2]*t*t + c[3]*t*t*t
                     + c[4]*t*t*t*t + c[5]*t*t*t*t*t;
            case 1:
                return c[1] + 2*c[2]*t + 3*c[3]*t*t
                     + 4*c[4]*t*t*t + 5*c[5]*t*t*t*t;
            case 2:
                return 2*c[2] + 6*c[3]*t + 12*c[4]*t*t + 20*c[5]*t*t*t;
            default:
                return 0.0;
        }
    }

    // =========================================================================
    // Build quintic spline through RRT/service waypoints.
    //   – First segment: initial vel/acc from live odometry
    //   – Intermediate nodes: central-difference vel, 2nd-difference acc
    //   – Last node: zero vel and acc (stop)
    // =========================================================================
    void buildTrajectory(const nav_msgs::msg::Path& path) {
        size_t N = path.poses.size();
        cxs_.clear(); cys_.clear(); czs_.clear();
        segTimes_.clear(); cumTimes_.clear();

        std::vector<geometry_msgs::msg::Point> pts(N);
        for (size_t i = 0; i < N; ++i) pts[i] = path.poses[i].pose.position;

        // Segment durations: dist / 0.5 (conservative), minimum 1s
        double total = 0.0;
        cumTimes_.push_back(0.0);
        for (size_t i = 0; i < N-1; ++i) {
            double dist = std::hypot(
                std::hypot(pts[i+1].x - pts[i].x, pts[i+1].y - pts[i].y),
                pts[i+1].z - pts[i].z);
            double T = std::max(1.0, dist * 0.5);
            segTimes_.push_back(T);
            total += T;
            cumTimes_.push_back(total);
        }
        total_time_ = total;

        // ── Velocity at each node (central difference) ──────────────────────
        std::vector<double> vx(N,0), vy(N,0), vz(N,0);
        for (size_t i = 1; i < N-1; ++i) {
            double dt = segTimes_[i-1] + segTimes_[i];
            vx[i] = (pts[i+1].x - pts[i-1].x) / dt;
            vy[i] = (pts[i+1].y - pts[i-1].y) / dt;
            vz[i] = (pts[i+1].z - pts[i-1].z) / dt;
        }

        // ── Acceleration at each node (second central difference) ────────────
        std::vector<double> ax(N,0), ay(N,0), az(N,0);
        for (size_t i = 1; i < N-1; ++i) {
            double dt_prev = segTimes_[i-1];
            double dt_next = segTimes_[i];
            double dt_avg  = (dt_prev + dt_next) / 2.0;
            ax[i] = (vx[i+1 < N ? i+1 : i] - vx[i > 0 ? i-1 : i]) / (2.0 * dt_avg);
            ay[i] = (vy[i+1 < N ? i+1 : i] - vy[i > 0 ? i-1 : i]) / (2.0 * dt_avg);
            az[i] = (vz[i+1 < N ? i+1 : i] - vz[i > 0 ? i-1 : i]) / (2.0 * dt_avg);
        }

        // ── Override first node with live odometry ───────────────────────────
        if (odom_received_) {
            vx[0] = current_velocity_.x();
            vy[0] = current_velocity_.y();
            vz[0] = current_velocity_.z();
            ax[0] = current_acceleration_.x();
            ay[0] = current_acceleration_.y();
            az[0] = current_acceleration_.z();
            RCLCPP_INFO(this->get_logger(),
                "Initial conditions from odometry: v=(%.2f,%.2f,%.2f) a=(%.2f,%.2f,%.2f)",
                vx[0], vy[0], vz[0], ax[0], ay[0], az[0]);
        }

        // ── Build quintic segments ───────────────────────────────────────────
        for (size_t i = 0; i < N-1; ++i) {
            cxs_.push_back(computeQuinticCoeffs(
                pts[i].x, vx[i], ax[i], pts[i+1].x, vx[i+1], ax[i+1], segTimes_[i]));
            cys_.push_back(computeQuinticCoeffs(
                pts[i].y, vy[i], ay[i], pts[i+1].y, vy[i+1], ay[i+1], segTimes_[i]));
            czs_.push_back(computeQuinticCoeffs(
                pts[i].z, vz[i], az[i], pts[i+1].z, vz[i+1], az[i+1], segTimes_[i]));
        }

        has_coeffs_   = true;
        finished_pub_ = false;
        start_time_   = this->now();
        RCLCPP_INFO(this->get_logger(),
            "Quintic trajectory built: %zu segments, %.2fs total", N-1, total_time_);
        publishTrajectoryPath();
    }

    // =========================================================================
    // Visualisation: dense-sample trajectory and publish as nav_msgs/Path
    // =========================================================================
    void publishTrajectoryPath() {
        nav_msgs::msg::Path path;
        path.header.frame_id = "world";
        path.header.stamp    = this->now();

        constexpr double dt = 0.1;
        for (size_t i = 0; i < cxs_.size(); ++i) {
            double T = segTimes_[i];
            for (double t = 0.0; t <= T + 1e-6; t += dt) {
                geometry_msgs::msg::PoseStamped ps;
                ps.header.frame_id   = "world";
                ps.header.stamp      = this->now();
                ps.pose.position.x   = evalQuintic(cxs_[i], t, 0);
                ps.pose.position.y   = evalQuintic(cys_[i], t, 0);
                ps.pose.position.z   = evalQuintic(czs_[i], t, 0);
                ps.pose.orientation.w = 1.0;
                path.poses.push_back(ps);
            }
        }
        traj_path_pub_->publish(path);
    }

    // =========================================================================
    // 20 Hz execution timer
    // =========================================================================
    void timerCallback() {
        if (!has_coeffs_) return;

        double t_elapsed = (this->now() - start_time_).seconds();

        size_t seg_idx = 0;
        bool   finished = false;

        if (t_elapsed >= total_time_) {
            seg_idx  = segTimes_.size() - 1;
            t_elapsed = total_time_;
            finished  = true;
        } else {
            for (size_t i = 0; i < cumTimes_.size()-1; ++i) {
                if (t_elapsed >= cumTimes_[i] && t_elapsed < cumTimes_[i+1]) {
                    seg_idx = i;
                    break;
                }
            }
        }

        double t_seg = t_elapsed - cumTimes_[seg_idx];
        if (t_seg > segTimes_[seg_idx]) t_seg = segTimes_[seg_idx];

        double px = evalQuintic(cxs_[seg_idx], t_seg, 0);
        double py = evalQuintic(cys_[seg_idx], t_seg, 0);
        double pz = evalQuintic(czs_[seg_idx], t_seg, 0);

        double vx = evalQuintic(cxs_[seg_idx], t_seg, 1);
        double vy = evalQuintic(cys_[seg_idx], t_seg, 1);
        double vz = evalQuintic(czs_[seg_idx], t_seg, 1);

        double ax = evalQuintic(cxs_[seg_idx], t_seg, 2);
        double ay = evalQuintic(cys_[seg_idx], t_seg, 2);
        double az = evalQuintic(czs_[seg_idx], t_seg, 2);

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint msg;
        msg.transforms.resize(1);
        msg.velocities.resize(1);
        msg.accelerations.resize(1);

        msg.transforms[0].translation.x = px;
        msg.transforms[0].translation.y = py;
        msg.transforms[0].translation.z = pz;

        // Yaw: follow velocity direction; freeze near zero speed
        double speed_xy = std::sqrt(vx*vx + vy*vy);
        static double last_yaw = 0.0;
        if (speed_xy > 0.1) last_yaw = std::atan2(vy, vx);
        tf2::Quaternion q;
        q.setRPY(0, 0, last_yaw);
        msg.transforms[0].rotation = tf2::toMsg(q);

        msg.velocities[0].linear.x = vx;
        msg.velocities[0].linear.y = vy;
        msg.velocities[0].linear.z = vz;

        msg.accelerations[0].linear.x = ax;
        msg.accelerations[0].linear.y = ay;
        msg.accelerations[0].linear.z = az;

        desired_pub_->publish(msg);

        if (finished && !finished_pub_) {
            complete_pub_->publish(std_msgs::msg::Empty());
            finished_pub_ = true;
            RCLCPP_INFO(this->get_logger(), "Trajectory complete.");
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryGenerator>());
    rclcpp::shutdown();
    return 0;
}
