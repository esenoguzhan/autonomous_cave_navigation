#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
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

        desired_pub_ = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>("desired_state", 10);
        complete_pub_ = this->create_publisher<std_msgs::msg::Empty>("trajectory_complete", 10);

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1, std::bind(&TrajectoryGenerator::stateCallback, this, std::placeholders::_1));

        start_navigation_service_ = this->create_service<trajectory_planner::srv::ExecuteTrajectory>(
            "start_navigation", std::bind(&TrajectoryGenerator::startNavigationCallback, this, std::placeholders::_1, std::placeholders::_2));

        timer_ = this->create_wall_timer(50ms, std::bind(&TrajectoryGenerator::timerCallback, this)); // 20Hz

        RCLCPP_INFO(this->get_logger(), "Trajectory Generator Initialized");
    }

private:
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_pub_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr complete_pub_; // Added publisher
    rclcpp::Service<trajectory_planner::srv::ExecuteTrajectory>::SharedPtr start_navigation_service_;
    rclcpp::TimerBase::SharedPtr timer_;

    std::string stm_state_ = "IDLE";
    bool has_coeffs_ = false;
    rclcpp::Time start_time_;
    double total_time_ = 0.0;
    bool finished_pub_ = false; // Flag to ensure single publication
    
    std::vector<double> segTimes_;
    std::vector<double> cumTimes_;
    std::vector<Eigen::VectorXd> cxs_, cys_, czs_;

    void stateCallback(const std_msgs::msg::String::SharedPtr msg) {
        stm_state_ = msg->data;
    }

    void startNavigationCallback(const std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Request> request,
                                 std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Response> response) {
        if (request->waypoints.poses.size() < 2) {
             response->success = false;
             response->message = "Need at least 2 waypoints";
             return;
        }

        nav_msgs::msg::Path path;
        path.poses.resize(request->waypoints.poses.size());
        for(size_t i=0; i<request->waypoints.poses.size(); ++i) {
             path.poses[i].pose = request->waypoints.poses[i];
        }

        buildTrajectory(path);
        
        response->success = true;
        response->message = "Trajectory started (Cubic Spline)";
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        if (msg->poses.size() < 2) return;
        // Only accept RRT path updates in EXPLORE state
        if (stm_state_ == "EXPLORE_CAVE" || stm_state_ == "EXPLORE") {
            buildTrajectory(*msg);
        }
    }

    Eigen::VectorXd computeCubicCoeffs(double p0, double pT, double T, double v0, double vT) {
        Eigen::VectorXd coeffs(4);
        coeffs[0] = p0;
        coeffs[1] = v0;
        double A = pT - p0 - v0 * T;
        double B = vT - v0;
        coeffs[3] = (B * T - 2.0 * A) / (std::pow(T, 3));
        coeffs[2] = (A - coeffs[3] * std::pow(T, 3)) / (std::pow(T, 2));
        return coeffs;
    }

    double evalCubic(const Eigen::VectorXd &coeffs, double t, int order) {
        if (order == 0) return coeffs[0] + coeffs[1]*t + coeffs[2]*t*t + coeffs[3]*t*t*t;
        if (order == 1) return coeffs[1] + 2*coeffs[2]*t + 3*coeffs[3]*t*t;
        if (order == 2) return 2*coeffs[2] + 6*coeffs[3]*t;
        return 0.0;
    }

    void buildTrajectory(const nav_msgs::msg::Path &path) {
        size_t N = path.poses.size();
        cxs_.clear(); cys_.clear(); czs_.clear();
        segTimes_.clear(); cumTimes_.clear();
        
        std::vector<geometry_msgs::msg::Point> points(N);
        for(size_t i=0; i<N; ++i) points[i] = path.poses[i].pose.position;

        double total = 0;
        cumTimes_.push_back(0);
        for(size_t i=0; i<N-1; ++i) {
            double dist = std::hypot(points[i+1].x - points[i].x, points[i+1].y - points[i].y);
            dist = std::hypot(dist, points[i+1].z - points[i].z); 
            double T = std::max(1.0, dist * 0.5); // Slower for safety
            segTimes_.push_back(T);
            total += T;
            cumTimes_.push_back(total);
        }
        total_time_ = total;

        std::vector<double> vx(N, 0), vy(N, 0), vz(N, 0);
        for(size_t i=1; i<N-1; ++i) {
             vx[i] = (points[i+1].x - points[i-1].x) / (segTimes_[i-1] + segTimes_[i]);
             vy[i] = (points[i+1].y - points[i-1].y) / (segTimes_[i-1] + segTimes_[i]);
             vz[i] = (points[i+1].z - points[i-1].z) / (segTimes_[i-1] + segTimes_[i]);
        }

        for(size_t i=0; i<N-1; ++i) {
            cxs_.push_back(computeCubicCoeffs(points[i].x, points[i+1].x, segTimes_[i], vx[i], vx[i+1]));
            cys_.push_back(computeCubicCoeffs(points[i].y, points[i+1].y, segTimes_[i], vy[i], vy[i+1]));
            czs_.push_back(computeCubicCoeffs(points[i].z, points[i+1].z, segTimes_[i], vz[i], vz[i+1]));
        }

        has_coeffs_ = true;
        finished_pub_ = false;
        start_time_ = this->now();
        RCLCPP_INFO(this->get_logger(), "Trajectory Built. Duration: %.2f", total_time_);
    }

    void timerCallback() {
        if (!has_coeffs_) return;

        double t_elapsed = (this->now() - start_time_).seconds();
        
        size_t seg_idx = 0;
        bool finished = false;
        if (t_elapsed >= total_time_) {
            seg_idx = segTimes_.size() - 1;
            t_elapsed = total_time_; // Clamp to end
            finished = true;
        } else {
            for(size_t i=0; i<cumTimes_.size()-1; ++i) {
                if (t_elapsed >= cumTimes_[i] && t_elapsed < cumTimes_[i+1]) {
                    seg_idx = i;
                    break;
                }
            }
        }

        double t_seg = t_elapsed - cumTimes_[seg_idx];
        if (t_seg > segTimes_[seg_idx]) t_seg = segTimes_[seg_idx];

        double px = evalCubic(cxs_[seg_idx], t_seg, 0);
        double py = evalCubic(cys_[seg_idx], t_seg, 0);
        double pz = evalCubic(czs_[seg_idx], t_seg, 0);

        double vx = evalCubic(cxs_[seg_idx], t_seg, 1);
        double vy = evalCubic(cys_[seg_idx], t_seg, 1);
        double vz = evalCubic(czs_[seg_idx], t_seg, 1);
        
        double ax = evalCubic(cxs_[seg_idx], t_seg, 2);
        double ay = evalCubic(cys_[seg_idx], t_seg, 2);
        double az = evalCubic(czs_[seg_idx], t_seg, 2);

        trajectory_msgs::msg::MultiDOFJointTrajectoryPoint msg;
        msg.transforms.resize(1);
        msg.velocities.resize(1);
        msg.accelerations.resize(1);

        msg.transforms[0].translation.x = px;
        msg.transforms[0].translation.y = py;
        msg.transforms[0].translation.z = pz;

        double yaw = std::atan2(vy, vx);
        tf2::Quaternion q;
        q.setRPY(0, 0, yaw);
        msg.transforms[0].rotation = tf2::toMsg(q);

        msg.velocities[0].linear.x = vx;
        msg.velocities[0].linear.y = vy;
        msg.velocities[0].linear.z = vz;

        msg.accelerations[0].linear.x = ax;
        msg.accelerations[0].linear.y = ay;
        msg.accelerations[0].linear.z = az;

        desired_pub_->publish(msg);

        if (finished && !finished_pub_) {
             std_msgs::msg::Empty msg_empty;
             complete_pub_->publish(msg_empty);
             finished_pub_ = true;
        }
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<TrajectoryGenerator>());
    rclcpp::shutdown();
    return 0;
}
