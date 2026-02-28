#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory_point.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/empty.hpp>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <Eigen/Dense>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include "trajectory_planner/srv/execute_trajectory.hpp"

using namespace std::chrono_literals;

/* ==========================================================================
 *  SAMPLING-BASED TRAJECTORY GENERATION
 * ==========================================================================
 *
 *  Instead of the classic "plan path → smooth trajectory" two-stage
 *  pipeline (RRT + cubic spline), this node works directly in
 *  TRAJECTORY SPACE:
 *
 *    1. Given a goal, sample N candidate 5th-order polynomial
 *       trajectories with varied durations and lateral offsets.
 *    2. Densely collision-check each trajectory against the OctoMap.
 *    3. Score feasible trajectories (time, smoothness, clearance).
 *    4. Execute the best one.
 *
 *  If no single-segment trajectory is feasible (e.g., obstacle in the
 *  way), the node recursively splits the problem by sampling
 *  intermediate via-points and building a multi-segment polynomial
 *  trajectory.
 * ========================================================================== */

// ============================================================================
// 5th-order (quintic) polynomial coefficients
//   p(t) = c0 + c1*t + c2*t² + c3*t³ + c4*t⁴ + c5*t⁵
//   boundary conditions: pos, vel, acc at t=0 and t=T
// ============================================================================
struct QuinticPoly {
    double c[6] = {};

    double pos(double t) const {
        return c[0] + c[1]*t + c[2]*t*t + c[3]*t*t*t + c[4]*t*t*t*t + c[5]*t*t*t*t*t;
    }
    double vel(double t) const {
        return c[1] + 2*c[2]*t + 3*c[3]*t*t + 4*c[4]*t*t*t + 5*c[5]*t*t*t*t;
    }
    double acc(double t) const {
        return 2*c[2] + 6*c[3]*t + 12*c[4]*t*t + 20*c[5]*t*t*t;
    }

    static QuinticPoly solve(double p0, double v0, double a0,
                             double pT, double vT, double aT, double T) {
        QuinticPoly q;
        q.c[0] = p0;
        q.c[1] = v0;
        q.c[2] = a0 / 2.0;

        double T2 = T * T;
        double T3 = T2 * T;
        double T4 = T3 * T;
        double T5 = T4 * T;

        // Solve the 3x3 system for c3, c4, c5
        Eigen::Matrix3d A;
        A <<   T3,    T4,    T5,
             3*T2,  4*T3,  5*T4,
             6*T,  12*T2, 20*T3;

        Eigen::Vector3d b;
        b(0) = pT  - p0 - v0*T - (a0/2.0)*T2;
        b(1) = vT  - v0 - a0*T;
        b(2) = aT  - a0;

        Eigen::Vector3d x = A.colPivHouseholderQr().solve(b);
        q.c[3] = x(0);
        q.c[4] = x(1);
        q.c[5] = x(2);
        return q;
    }
};

// One segment of a trajectory (x, y, z quintics over duration T)
struct TrajSegment {
    QuinticPoly px, py, pz;
    double T;  // duration of this segment
};

// Full trajectory = list of segments
struct Trajectory {
    std::vector<TrajSegment> segments;
    double total_time = 0.0;
    double score      = 0.0;  // lower is better

    void computeTotalTime() {
        total_time = 0.0;
        for (auto& s : segments) total_time += s.T;
    }
};

// ============================================================================
// SamplingPlannerNode
// ============================================================================
class SamplingPlannerNode : public rclcpp::Node {
public:
    SamplingPlannerNode()
    : Node("sampling_planner_node"),
      rng_(std::random_device{}())
    {
        // ---- Parameters ----
        this->declare_parameter("num_samples",        30);
        this->declare_parameter("min_duration_factor", 0.6);
        this->declare_parameter("max_duration_factor", 2.5);
        this->declare_parameter("lateral_spread",      3.0);
        this->declare_parameter("max_recursion_depth",  3);
        this->declare_parameter("safety_radius",       0.5);
        this->declare_parameter("trajectory_speed",    2.0);
        this->declare_parameter("collision_check_dt",  0.2);
        this->declare_parameter("planning_frequency",  1.0);

        num_samples_        = this->get_parameter("num_samples").as_int();
        min_dur_factor_     = this->get_parameter("min_duration_factor").as_double();
        max_dur_factor_     = this->get_parameter("max_duration_factor").as_double();
        lateral_spread_     = this->get_parameter("lateral_spread").as_double();
        max_recursion_      = this->get_parameter("max_recursion_depth").as_int();
        safety_radius_      = this->get_parameter("safety_radius").as_double();
        traj_speed_         = this->get_parameter("trajectory_speed").as_double();
        collision_check_dt_ = this->get_parameter("collision_check_dt").as_double();
        planning_freq_      = this->get_parameter("planning_frequency").as_double();

        // ---- Subscribers ----
        auto octomap_qos = rclcpp::QoS(1).transient_local().reliable();
        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", octomap_qos,
            std::bind(&SamplingPlannerNode::octomapCallback, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "pose_est", 1,
            std::bind(&SamplingPlannerNode::poseCallback, this, std::placeholders::_1));

        goal_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
            "frontier_goal", 1,
            std::bind(&SamplingPlannerNode::goalCallback, this, std::placeholders::_1));

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1,
            std::bind(&SamplingPlannerNode::stateCallback, this, std::placeholders::_1));

        // ---- Publishers ----
        desired_pub_  = this->create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>(
            "desired_state", 10);
        complete_pub_ = this->create_publisher<std_msgs::msg::Empty>(
            "trajectory_complete", 10);

        // ---- Service ----
        start_nav_srv_ = this->create_service<trajectory_planner::srv::ExecuteTrajectory>(
            "start_navigation",
            std::bind(&SamplingPlannerNode::startNavigationCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // ---- Timers ----
        traj_timer_ = this->create_wall_timer(
            50ms, std::bind(&SamplingPlannerNode::trajTimerCallback, this));

        plan_timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / planning_freq_),
            std::bind(&SamplingPlannerNode::planTimerCallback, this));

        RCLCPP_INFO(this->get_logger(),
            "╔════════════════════════════════════════════════════════════╗");
        RCLCPP_INFO(this->get_logger(),
            "║  Sampling-Based Trajectory Generator Initialized         ║");
        RCLCPP_INFO(this->get_logger(),
            "╚════════════════════════════════════════════════════════════╝");
        RCLCPP_INFO(this->get_logger(),
            "  samples=%d  lat_spread=%.1f  safety_r=%.2f  speed=%.1f",
            num_samples_, lateral_spread_, safety_radius_, traj_speed_);
    }

    ~SamplingPlannerNode() {
        if (octree_) delete octree_;
    }

private:
    // ---- Parameters ----
    int    num_samples_;
    double min_dur_factor_;
    double max_dur_factor_;
    double lateral_spread_;
    int    max_recursion_;
    double safety_radius_;
    double traj_speed_;
    double collision_check_dt_;
    double planning_freq_;

    // ---- ROS interfaces ----
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr goal_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectoryPoint>::SharedPtr desired_pub_;
    rclcpp::Publisher<std_msgs::msg::Empty>::SharedPtr complete_pub_;
    rclcpp::Service<trajectory_planner::srv::ExecuteTrajectory>::SharedPtr start_nav_srv_;
    rclcpp::TimerBase::SharedPtr traj_timer_;
    rclcpp::TimerBase::SharedPtr plan_timer_;

    // ---- State ----
    octomap::OcTree* octree_ = nullptr;
    geometry_msgs::msg::PoseStamped current_pose_;
    geometry_msgs::msg::Point       current_goal_;
    bool pose_received_   = false;
    bool goal_received_   = false;
    bool octree_received_ = false;
    bool active_          = false;   // stm_mode == EXPLORE_CAVE
    bool goal_changed_    = false;   // true when frontier goal shifted >2m mid-trajectory

    // ---- Active trajectory ----
    Trajectory active_traj_;
    bool   has_trajectory_ = false;
    bool   finished_pub_   = false;
    rclcpp::Time traj_start_time_;

    // ---- RNG ----
    std::mt19937 rng_;

    // ========================================================================
    // Callbacks
    // ========================================================================

    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        if (octree_) delete octree_;
        auto* tree = octomap_msgs::msgToMap(*msg);
        octree_ = dynamic_cast<octomap::OcTree*>(tree);
        if (octree_) octree_received_ = true;
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        current_pose_ = *msg;
        pose_received_ = true;
    }

    void goalCallback(const geometry_msgs::msg::Point::SharedPtr msg) {
        if (!active_) return;
        double dx = msg->x - current_goal_.x;
        double dy = msg->y - current_goal_.y;
        double dz = msg->z - current_goal_.z;
        if (!goal_received_ || std::sqrt(dx*dx + dy*dy + dz*dz) > 2.0) {
            current_goal_ = *msg;
            goal_received_ = true;
            goal_changed_  = true;
            RCLCPP_INFO(this->get_logger(),
                "New frontier goal: [%.2f, %.2f, %.2f]", msg->x, msg->y, msg->z);
        }
    }

    void stateCallback(const std_msgs::msg::String::SharedPtr msg) {
        active_ = (msg->data == "EXPLORE_CAVE" || msg->data == "EXPLORE");
    }

    // ========================================================================
    // Service: start_navigation (NAVIGATE_TO_CAVE phase)
    // ========================================================================

    void startNavigationCallback(
        const std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Request> request,
        std::shared_ptr<trajectory_planner::srv::ExecuteTrajectory::Response> response)
    {
        if (request->waypoints.poses.size() < 2) {
            response->success = false;
            response->message = "Need at least 2 waypoints";
            return;
        }

        // For service-provided waypoints, build a multi-segment quintic
        // trajectory through all via-points with zero vel/acc at endpoints.
        Trajectory traj;
        auto& poses = request->waypoints.poses;
        for (size_t i = 0; i < poses.size() - 1; ++i) {
            Eigen::Vector3d p0(poses[i].position.x,   poses[i].position.y,   poses[i].position.z);
            Eigen::Vector3d pT(poses[i+1].position.x, poses[i+1].position.y, poses[i+1].position.z);
            double dist = (pT - p0).norm();
            double T = std::max(1.0, dist / traj_speed_);

            // Intermediate via-points: non-zero velocity matching direction
            Eigen::Vector3d v0 = Eigen::Vector3d::Zero();
            Eigen::Vector3d vT = Eigen::Vector3d::Zero();
            if (i > 0) {
                Eigen::Vector3d pp(poses[i-1].position.x, poses[i-1].position.y, poses[i-1].position.z);
                v0 = (pT - pp).normalized() * traj_speed_ * 0.5;
            }
            if (i + 2 < poses.size()) {
                Eigen::Vector3d pn(poses[i+2].position.x, poses[i+2].position.y, poses[i+2].position.z);
                vT = (pn - p0).normalized() * traj_speed_ * 0.5;
            }

            TrajSegment seg;
            seg.T  = T;
            seg.px = QuinticPoly::solve(p0.x(), v0.x(), 0, pT.x(), vT.x(), 0, T);
            seg.py = QuinticPoly::solve(p0.y(), v0.y(), 0, pT.y(), vT.y(), 0, T);
            seg.pz = QuinticPoly::solve(p0.z(), v0.z(), 0, pT.z(), vT.z(), 0, T);
            traj.segments.push_back(seg);
        }
        traj.computeTotalTime();

        active_traj_ = traj;
        has_trajectory_ = true;
        finished_pub_ = false;
        traj_start_time_ = this->now();

        response->success = true;
        response->message = "Sampling-based trajectory started";
        RCLCPP_INFO(this->get_logger(),
            "start_navigation: %zu segments, total=%.2fs", traj.segments.size(), traj.total_time);
    }

    // ========================================================================
    // Collision checking helpers
    // ========================================================================

    bool isPointCollisionFree(double x, double y, double z) const {
        if (!octree_) return false;
        auto* node = octree_->search(x, y, z);
        if (node && octree_->isNodeOccupied(node)) return false;

        // Safety margin: check axis-aligned neighbors
        if (safety_radius_ > 0.0) {
            double r = safety_radius_;
            double offsets[6][3] = {
                {r,0,0}, {-r,0,0}, {0,r,0}, {0,-r,0}, {0,0,r}, {0,0,-r}
            };
            for (auto& off : offsets) {
                auto* n = octree_->search(x + off[0], y + off[1], z + off[2]);
                if (n && octree_->isNodeOccupied(n)) return false;
            }
        }
        return true;
    }

    // Check entire trajectory for collisions by sampling at collision_check_dt_
    bool isTrajectoryFeasible(const Trajectory& traj) const {
        double t = 0.0;
        for (size_t si = 0; si < traj.segments.size(); ++si) {
            const auto& seg = traj.segments[si];
            double seg_t = 0.0;
            while (seg_t <= seg.T) {
                double x = seg.px.pos(seg_t);
                double y = seg.py.pos(seg_t);
                double z = seg.pz.pos(seg_t);
                if (!isPointCollisionFree(x, y, z)) return false;
                seg_t += collision_check_dt_;
            }
        }
        return true;
    }

    // Compute minimum clearance along trajectory (for scoring)
    double minClearance(const Trajectory& traj) const {
        double min_clear = std::numeric_limits<double>::max();
        for (const auto& seg : traj.segments) {
            double seg_t = 0.0;
            while (seg_t <= seg.T) {
                double x = seg.px.pos(seg_t);
                double y = seg.py.pos(seg_t);
                double z = seg.pz.pos(seg_t);
                // Probe clearance: find first occupied in radial directions
                for (double r = 0.1; r <= 5.0; r += 0.3) {
                    bool any_occ = false;
                    double offs[6][3] = {{r,0,0},{-r,0,0},{0,r,0},{0,-r,0},{0,0,r},{0,0,-r}};
                    for (auto& off : offs) {
                        auto* n = octree_->search(x + off[0], y + off[1], z + off[2]);
                        if (n && octree_->isNodeOccupied(n)) { any_occ = true; break; }
                    }
                    if (any_occ) { min_clear = std::min(min_clear, r); break; }
                }
                seg_t += collision_check_dt_ * 2.0;  // coarser for scoring
            }
        }
        return min_clear;
    }

    // ========================================================================
    // Core: Sample candidate trajectories and pick the best feasible one
    // ========================================================================

    bool sampleTrajectories(const Eigen::Vector3d& p0, const Eigen::Vector3d& v0,
                            const Eigen::Vector3d& pGoal,
                            Trajectory& best_traj, int depth = 0)
    {
        double dist = (pGoal - p0).norm();
        if (dist < 0.5) return false;

        double nominal_T = dist / traj_speed_;

        // Build a local coordinate frame: x_dir = toward goal, y_dir and z_dir = perpendicular
        Eigen::Vector3d x_dir = (pGoal - p0).normalized();
        Eigen::Vector3d arbitrary = (std::abs(x_dir.z()) < 0.9) ?
            Eigen::Vector3d(0, 0, 1) : Eigen::Vector3d(1, 0, 0);
        Eigen::Vector3d y_dir = x_dir.cross(arbitrary).normalized();
        Eigen::Vector3d z_dir = x_dir.cross(y_dir).normalized();

        std::uniform_real_distribution<double> dur_dist(
            nominal_T * min_dur_factor_, nominal_T * max_dur_factor_);
        std::uniform_real_distribution<double> lat_dist(
            -lateral_spread_, lateral_spread_);
        std::uniform_real_distribution<double> z_dist(
            -lateral_spread_ * 0.5, lateral_spread_ * 0.5);

        std::vector<Trajectory> candidates;
        candidates.reserve(num_samples_);

        for (int i = 0; i < num_samples_; ++i) {
            double T = dur_dist(rng_);
            T = std::max(T, 0.5);

            // Sample a goal variation (lateral + vertical offset from the true goal)
            Eigen::Vector3d goal_offset = Eigen::Vector3d::Zero();
            if (i > 0) {  // first sample: go exactly to goal
                goal_offset = y_dir * lat_dist(rng_) + z_dir * z_dist(rng_);
            }
            Eigen::Vector3d pEnd = pGoal + goal_offset;

            // Build quintic from (p0, v0, a0=0) to (pEnd, v_end=0, a_end=0)
            TrajSegment seg;
            seg.T  = T;
            seg.px = QuinticPoly::solve(p0.x(), v0.x(), 0, pEnd.x(), 0, 0, T);
            seg.py = QuinticPoly::solve(p0.y(), v0.y(), 0, pEnd.y(), 0, 0, T);
            seg.pz = QuinticPoly::solve(p0.z(), v0.z(), 0, pEnd.z(), 0, 0, T);

            Trajectory traj;
            traj.segments.push_back(seg);
            traj.computeTotalTime();

            // Feasibility check
            if (isTrajectoryFeasible(traj)) {
                // Score: prefer shorter time + larger clearance + closer to actual goal
                double clearance = minClearance(traj);
                double goal_dev  = goal_offset.norm();
                traj.score = T - 0.5 * clearance + 0.3 * goal_dev;
                candidates.push_back(traj);
            }
        }

        // If we found feasible trajectories, pick the best
        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end(),
                [](const Trajectory& a, const Trajectory& b) { return a.score < b.score; });
            best_traj = candidates[0];
            RCLCPP_INFO(this->get_logger(),
                "Found %zu/%d feasible single-segment trajectories (best score=%.2f, T=%.2f)",
                candidates.size(), num_samples_, best_traj.score, best_traj.total_time);
            return true;
        }

        // No single-segment trajectory worked → try splitting via intermediate waypoints
        if (depth >= max_recursion_) {
            RCLCPP_WARN(this->get_logger(),
                "Max recursion depth %d reached, cannot find feasible trajectory.", depth);
            return false;
        }

        RCLCPP_INFO(this->get_logger(),
            "No direct trajectory feasible. Splitting (depth=%d)...", depth);

        // Sample intermediate via-points near the midpoint
        Eigen::Vector3d midpoint = (p0 + pGoal) * 0.5;
        int mid_samples = num_samples_ / 2;
        std::uniform_real_distribution<double> mid_lat(-lateral_spread_ * 1.5, lateral_spread_ * 1.5);

        for (int i = 0; i < mid_samples; ++i) {
            Eigen::Vector3d via = midpoint;
            if (i > 0) {
                via += y_dir * mid_lat(rng_) + z_dir * mid_lat(rng_) * 0.5;
            }

            // Check via-point not in collision
            if (!isPointCollisionFree(via.x(), via.y(), via.z())) continue;

            // Try first half: p0 → via
            Trajectory traj_first;
            if (!sampleTrajectories(p0, v0, via, traj_first, depth + 1)) continue;

            // Get velocity at end of first half for continuity
            auto& last_seg = traj_first.segments.back();
            double T1 = last_seg.T;
            Eigen::Vector3d v_via(last_seg.px.vel(T1), last_seg.py.vel(T1), last_seg.pz.vel(T1));

            // Try second half: via → pGoal
            Trajectory traj_second;
            if (!sampleTrajectories(via, v_via, pGoal, traj_second, depth + 1)) continue;

            // Concatenate
            Trajectory combined;
            combined.segments.insert(combined.segments.end(),
                traj_first.segments.begin(), traj_first.segments.end());
            combined.segments.insert(combined.segments.end(),
                traj_second.segments.begin(), traj_second.segments.end());
            combined.computeTotalTime();
            combined.score = traj_first.score + traj_second.score;

            best_traj = combined;
            RCLCPP_INFO(this->get_logger(),
                "Multi-segment trajectory found! %zu segments, total=%.2fs",
                combined.segments.size(), combined.total_time);
            return true;
        }

        return false;
    }

    // ========================================================================
    // Planning timer: periodically plan if exploring
    // ========================================================================

    void planTimerCallback() {
        if (!active_ || !pose_received_ || !goal_received_ || !octree_received_) return;
        if (has_trajectory_ && !finished_pub_ && !goal_changed_) return;  // already executing

        Eigen::Vector3d p0(
            current_pose_.pose.position.x,
            current_pose_.pose.position.y,
            current_pose_.pose.position.z);
        Eigen::Vector3d pGoal(current_goal_.x, current_goal_.y, current_goal_.z);

        if ((pGoal - p0).norm() < 1.0) {
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 3000,
                "Already near goal (dist=%.2fm).", (pGoal - p0).norm());
            return;
        }

        // Current velocity is roughly zero if hovering; could be improved
        // with actual velocity feedback
        Eigen::Vector3d v0 = Eigen::Vector3d::Zero();

        Trajectory best;
        if (sampleTrajectories(p0, v0, pGoal, best)) {
            active_traj_ = best;
            has_trajectory_ = true;
            finished_pub_ = false;
            goal_changed_ = false;
            traj_start_time_ = this->now();
            RCLCPP_INFO(this->get_logger(),
                "Executing trajectory: %zu segments, %.2fs total",
                best.segments.size(), best.total_time);
        } else {
            RCLCPP_WARN(this->get_logger(),
                "Failed to find any feasible trajectory to goal.");
        }
    }

    // ========================================================================
    // Trajectory execution timer (20 Hz)
    // ========================================================================

    void trajTimerCallback() {
        if (!has_trajectory_) return;

        double t_elapsed = (this->now() - traj_start_time_).seconds();

        // Determine which segment and local time
        double cum_time = 0.0;
        size_t seg_idx = 0;
        double t_local = 0.0;
        bool finished = false;

        if (t_elapsed >= active_traj_.total_time) {
            seg_idx = active_traj_.segments.size() - 1;
            t_local = active_traj_.segments[seg_idx].T;
            finished = true;
        } else {
            for (size_t i = 0; i < active_traj_.segments.size(); ++i) {
                if (t_elapsed < cum_time + active_traj_.segments[i].T) {
                    seg_idx = i;
                    t_local = t_elapsed - cum_time;
                    break;
                }
                cum_time += active_traj_.segments[i].T;
            }
        }

        const auto& seg = active_traj_.segments[seg_idx];

        // Evaluate position, velocity, acceleration from quintic polynomials
        double px = seg.px.pos(t_local);
        double py = seg.py.pos(t_local);
        double pz = seg.pz.pos(t_local);

        double vx = seg.px.vel(t_local);
        double vy = seg.py.vel(t_local);
        double vz = seg.pz.vel(t_local);

        double ax = seg.px.acc(t_local);
        double ay = seg.py.acc(t_local);
        double az = seg.pz.acc(t_local);

        // Build desired_state message
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

        // Trajectory complete
        if (finished && !finished_pub_) {
            complete_pub_->publish(std_msgs::msg::Empty());
            finished_pub_ = true;
            has_trajectory_ = false;
            RCLCPP_INFO(this->get_logger(), "Trajectory complete.");
        }
    }
};

// ============================================================================
int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<SamplingPlannerNode>());
    rclcpp::shutdown();
    return 0;
}
