#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/srv/get_plan.hpp>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/geometric/SimpleSetup.h>
#include <ompl/geometric/planners/rrt/RRTstar.h>
#include <ompl/base/ScopedState.h>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vector>
#include <memory>

namespace ob = ompl::base;
namespace og = ompl::geometric;

class RRTPlanner : public rclcpp::Node {
public:
    RRTPlanner() : Node("rrt_planner") {
        this->declare_parameter("step_size_factor", 0.5);
        this->declare_parameter("bias", 0.05);
        this->declare_parameter("timeout", 1.0);
        this->declare_parameter("rrt_frequency", 1.0);

        step_size_factor_ = this->get_parameter("step_size_factor").as_double();
        bias_ = this->get_parameter("bias").as_double();
        timeout_ = this->get_parameter("timeout").as_double();
        rrt_frequency_ = this->get_parameter("rrt_frequency").as_double();

        // Use transient_local QoS to match the octomap_server publisher
        auto octomap_qos = rclcpp::QoS(1).transient_local().reliable();
        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", octomap_qos, std::bind(&RRTPlanner::octomapCallback, this, std::placeholders::_1));
        
        start_pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "pose_est", 1, std::bind(&RRTPlanner::startPoseCallback, this, std::placeholders::_1));

        goal_point_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "frontier_goal", 1, std::bind(&RRTPlanner::goalCallback, this, std::placeholders::_1));

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1, std::bind(&RRTPlanner::stateCallback, this, std::placeholders::_1));

        path_pub_ = this->create_publisher<nav_msgs::msg::Path>("rrt_path", 1);

        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / rrt_frequency_), 
            std::bind(&RRTPlanner::timerCallback, this));
        
        RCLCPP_INFO(this->get_logger(), "RRT Planner Initialized");
    }

    ~RRTPlanner() {
        if (map_) delete map_;
    }

private:
    double step_size_factor_;
    double bias_;
    double timeout_;
    double rrt_frequency_;

    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr start_pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_point_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    octomap::OcTree* map_ = nullptr;
    geometry_msgs::msg::PoseStamped start_pose_;
    geometry_msgs::msg::Point goal_point_;
    bool start_received_ = false;
    bool goal_received_ = false;
    bool octomap_received_ = false;
    bool active_ = false;
    double map_resolution_ = 0.5;

    void stateCallback(const std_msgs::msg::String::SharedPtr msg) {
        if (msg->data == "EXPLORE_CAVE" || msg->data == "EXPLORE") {
             active_ = true;
        } else {
             active_ = false;
        }
    }

    void startPoseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        start_pose_ = *msg;
        start_received_ = true;
    }

    void goalCallback(const geometry_msgs::msg::Point::SharedPtr msg) {
        goal_point_ = *msg;
        goal_received_ = true;
    }

    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        if (map_) delete map_;
        octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
        map_ = dynamic_cast<octomap::OcTree*>(tree);
        if (map_) {
            map_resolution_ = map_->getResolution();
            octomap_received_ = true;
        }
    }

    void timerCallback() {
        if (active_ && start_received_ && goal_received_ && octomap_received_) {
            nav_msgs::msg::Path path;
            if (planPath(path)) {
                 path_pub_->publish(path);
            }
        }
    }

    bool isStateValid(const ob::State *state) {
        if (!map_) return false;
        const auto *state3D = state->as<ob::RealVectorStateSpace::StateType>();
        double x = state3D->values[0];
        double y = state3D->values[1];
        double z = state3D->values[2];
        octomap::OcTreeNode *node = map_->search(x, y, z);
        if (node == nullptr) return true; 
        return node->getOccupancy() <= 0.7;
    }

    bool planPath(nav_msgs::msg::Path& out_path) {
        if (!map_) return false;

        octomap::point3d start_pt(start_pose_.pose.position.x, start_pose_.pose.position.y, start_pose_.pose.position.z);
        octomap::point3d goal_pt(goal_point_.x, goal_point_.y, goal_point_.z);

        octomap::point3d hit;
        double dist = (goal_pt - start_pt).norm();
        if (!map_->castRay(start_pt, goal_pt - start_pt, hit, true, dist)) {
            // No obstacle, returns straight line
            out_path.header.frame_id = "world";
            out_path.header.stamp = this->now();
            geometry_msgs::msg::PoseStamped p1, p2;
            p1.pose.position = start_pose_.pose.position;
            p2.pose.position = goal_point_;
            out_path.poses.push_back(p1);
            out_path.poses.push_back(p2);
            return true;
        }

        auto space = std::make_shared<ob::RealVectorStateSpace>(3);
        ob::RealVectorBounds bounds(3);
        double x_min, y_min, z_min, x_max, y_max, z_max;
        map_->getMetricMin(x_min, y_min, z_min);
        map_->getMetricMax(x_max, y_max, z_max);
        
        bounds.setLow(0, x_min); bounds.setHigh(0, x_max);
        bounds.setLow(1, y_min); bounds.setHigh(1, y_max);
        bounds.setLow(2, z_min); bounds.setHigh(2, z_max);
        space->setBounds(bounds);

        og::SimpleSetup ss(space);
        ss.setStateValidityChecker([this](const ob::State *state) { return isStateValid(state); });

        ob::ScopedState<> start(space), goal(space);
        start[0] = start_pose_.pose.position.x;
        start[1] = start_pose_.pose.position.y;
        start[2] = start_pose_.pose.position.z;
        goal[0] = goal_point_.x;
        goal[1] = goal_point_.y;
        goal[2] = goal_point_.z;

        ss.setStartAndGoalStates(start, goal);

        auto planner = std::make_shared<og::RRTstar>(ss.getSpaceInformation());
        planner->setRange(step_size_factor_ * map_resolution_);
        planner->setGoalBias(bias_);
        ss.setPlanner(planner);

        ob::PlannerStatus solved = ss.solve(timeout_);
        
        if (solved) {
             ss.simplifySolution(); 
             og::PathGeometric& path = ss.getSolutionPath();
             
             out_path.header.frame_id = "world";
             out_path.header.stamp = this->now();
             
             for (auto state : path.getStates()) {
                 const auto *s = state->as<ob::RealVectorStateSpace::StateType>();
                 geometry_msgs::msg::PoseStamped pose;
                 pose.pose.position.x = s->values[0];
                 pose.pose.position.y = s->values[1];
                 pose.pose.position.z = s->values[2];
                 out_path.poses.push_back(pose);
             }
             return true;
        }

        return false;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RRTPlanner>());
    rclcpp::shutdown();
    return 0;
}
