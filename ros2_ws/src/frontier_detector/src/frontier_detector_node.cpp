#include <rclcpp/rclcpp.hpp>
#include <rclcpp/qos.hpp>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <algorithm>
#include <cmath>
#include <memory>
#include <random>
#include <vector>

using namespace std::chrono_literals;

struct Point3D {
    double x, y, z;
};

struct Frontier {
    Point3D coordinates;
    int neighborcount = 0;
    double score = 0.0;
    bool isReachable = false;
};

class FrontierDetector : public rclcpp::Node {
public:
    FrontierDetector() : Node("frontier_detector") {
        // Parameters - same as previous year's frontier_detector.yaml
        this->declare_parameter("neighborcount_threshold", 100);
        this->declare_parameter("bandwidth", 17.0);
        this->declare_parameter("k_distance", 1.0);
        this->declare_parameter("k_neighborcount", 0.1);
        this->declare_parameter("k_yaw", 55.0);
        this->declare_parameter("distance_limit", 600.0);
        this->declare_parameter("publish_goal_frequency", 2.0);
        this->declare_parameter("occ_neighbor_threshold", 1);
        // Hard cap on frontier points fed to mean-shift (random subsample)
        this->declare_parameter("max_frontiers", 3000);
        // Only keep clustered frontiers within [min, max] distance from drone.
        this->declare_parameter("min_frontier_distance", 10.0);
        this->declare_parameter("max_frontier_distance", 60.0);

        neighborcount_threshold_ = this->get_parameter("neighborcount_threshold").as_int();
        bandwidth_                = this->get_parameter("bandwidth").as_double();
        k_distance_               = this->get_parameter("k_distance").as_double();
        k_neighborcount_          = this->get_parameter("k_neighborcount").as_double();
        k_yaw_                    = this->get_parameter("k_yaw").as_double();
        distance_limit_           = this->get_parameter("distance_limit").as_double();
        publish_goal_frequency_   = this->get_parameter("publish_goal_frequency").as_double();
        occ_neighbor_threshold_   = this->get_parameter("occ_neighbor_threshold").as_int();
        max_frontiers_            = this->get_parameter("max_frontiers").as_int();
        min_frontier_distance_    = this->get_parameter("min_frontier_distance").as_double();
        max_frontier_distance_    = this->get_parameter("max_frontier_distance").as_double();

        // Use transient_local QoS to match octomap_server publisher
        auto octomap_qos = rclcpp::QoS(1).transient_local().reliable();
        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", octomap_qos,
            std::bind(&FrontierDetector::parseOctomap, this, std::placeholders::_1));

        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "pose_est", 1,
            std::bind(&FrontierDetector::currentPosition, this, std::placeholders::_1));

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1,
            std::bind(&FrontierDetector::onStateStm, this, std::placeholders::_1));

        frontier_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("frontiers", 1);
        goal_pub_     = this->create_publisher<geometry_msgs::msg::PoseStamped>("frontier_goal_pose", 1);
        goal_point_pub_ = this->create_publisher<geometry_msgs::msg::Point>("frontier_goal", 1);

        // OctoMap reset service client
        octomap_reset_client_ = this->create_client<std_srvs::srv::Empty>("octomap_server/reset");

        // Service for start_exploration (kept for mission_control compatibility, state controls logic)
        start_exploration_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "start_exploration",
            std::bind(&FrontierDetector::startExplorationCallback, this,
                      std::placeholders::_1, std::placeholders::_2));

        // Timer for publishing goal at fixed frequency
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_goal_frequency_),
            std::bind(&FrontierDetector::publishGoal, this));

        // Cave entrance coordinates (same as previous year)
        cave_entry_point_ = {-321.0, 10.0, 15.0};

        RCLCPP_INFO(this->get_logger(), "Frontier Detector Initialized");
    }

private:
    // Parameters
    int    neighborcount_threshold_;
    double bandwidth_;
    double k_distance_;
    double k_neighborcount_;
    double k_yaw_;
    double distance_limit_;
    double publish_goal_frequency_;
    int    occ_neighbor_threshold_;
    int    max_frontiers_;
    double min_frontier_distance_;
    double max_frontier_distance_;

    // ROS interfaces
    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr  octomap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr       state_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr      goal_point_pub_;
    rclcpp::Client<std_srvs::srv::Empty>::SharedPtr              octomap_reset_client_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr           start_exploration_srv_;
    rclcpp::TimerBase::SharedPtr timer_;

    // State
    Point3D     curr_drone_position_ = {0.0, 0.0, 0.0};
    double      drone_yaw_           = 0.0;
    std::string statemachine_state_  = "IDLE";
    bool        octomap_reset_done_  = false;
    float       octomap_res_         = 0.5f;
    // True between resetOctomap() call and the first post-reset message.
    // While set, parseOctomap skips stale pre-reset messages.
    bool        waiting_for_reset_   = false;

    // Current best goal (set in selectFrontier, published by timer)
    geometry_msgs::msg::PoseStamped goal_message_;
    geometry_msgs::msg::Point       goal_point_;
    bool goal_available_ = false;

    Point3D cave_entry_point_;

    // =========================================================================
    // State callback — triggers OctoMap reset exactly once on EXPLORE entry
    // (mirrors prev year's onStateStm)
    // =========================================================================
    void onStateStm(const std_msgs::msg::String::SharedPtr msg) {
        std::string prev = statemachine_state_;
        statemachine_state_ = msg->data;

        bool entering_explore = (statemachine_state_ == "EXPLORE_CAVE" ||
                                 statemachine_state_ == "EXPLORE");
        bool was_not_exploring = (prev != "EXPLORE_CAVE" && prev != "EXPLORE");

        if (entering_explore && was_not_exploring && !octomap_reset_done_) {
            RCLCPP_INFO(this->get_logger(),
                "Reached cave entry. Resetting OctoMap and starting cave exploration.");
            resetOctomap();
            octomap_reset_done_ = true;
        }
    }

    // =========================================================================
    // start_exploration service stub (mission_control compatibility)
    // =========================================================================
    void startExplorationCallback(
        const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
        std::shared_ptr<std_srvs::srv::Trigger::Response> res)
    {
        res->success = true;
        res->message = "Frontier Detector: exploration controlled by state machine";
        RCLCPP_INFO(this->get_logger(), "start_exploration service called (state machine controls logic).");
    }

    // =========================================================================
    // Pose callback
    // =========================================================================
    void currentPosition(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        curr_drone_position_.x = msg->pose.position.x;
        curr_drone_position_.y = msg->pose.position.y;
        curr_drone_position_.z = msg->pose.position.z;

        tf2::Quaternion quat(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w);
        tf2::Matrix3x3 mat(quat);
        double roll, pitch;
        mat.getRPY(roll, pitch, drone_yaw_);
    }

    // =========================================================================
    // OctoMap reset (calls octomap_server/reset service)
    // =========================================================================
    void resetOctomap() {
        if (!octomap_reset_client_->wait_for_service(1s)) {
            RCLCPP_WARN(this->get_logger(), "OctoMap reset service not available, skipping.");
            return;
        }
        // Block processing of stale pre-reset OctoMap messages
        waiting_for_reset_ = true;
        auto request = std::make_shared<std_srvs::srv::Empty::Request>();
        octomap_reset_client_->async_send_request(request);
        RCLCPP_INFO(this->get_logger(), "OctoMap reset requested. Skipping stale messages until reset.");
    }

    // =========================================================================
    // Helpers
    // =========================================================================
    double euclideanDistance(const Point3D& p1, const Point3D& p2) {
        double dx = p1.x - p2.x;
        double dy = p1.y - p2.y;
        double dz = p1.z - p2.z;
        return std::sqrt(dx*dx + dy*dy + dz*dz);
    }

    double gaussianKernel(double distance) {
        return std::exp(-(distance * distance) / (2.0 * bandwidth_ * bandwidth_));
    }

    // =========================================================================
    // Mean shift clustering — O(n²), faithful port of previous year
    // =========================================================================
    std::vector<Point3D> meanShiftClustering(
        const std::vector<Frontier>& points, double convergenceThreshold)
    {
        RCLCPP_INFO(this->get_logger(), "MeanShiftClustering starting (%zu points)", points.size());

        std::vector<Point3D> shiftedPoints;
        shiftedPoints.reserve(points.size());
        for (const auto& f : points) {
            shiftedPoints.push_back(f.coordinates);
        }

        bool converged = false;
        while (!converged) {
            converged = true;
            for (size_t i = 0; i < shiftedPoints.size(); ++i) {
                Point3D original = shiftedPoints[i];
                Point3D shifted  = {0.0, 0.0, 0.0};
                double  totalW   = 0.0;

                for (const auto& p : points) {
                    double dist   = euclideanDistance(original, p.coordinates);
                    double weight = gaussianKernel(dist);
                    shifted.x += p.coordinates.x * weight;
                    shifted.y += p.coordinates.y * weight;
                    shifted.z += p.coordinates.z * weight;
                    totalW    += weight;
                }

                if (totalW > 0.0) {
                    shifted.x /= totalW;
                    shifted.y /= totalW;
                    shifted.z /= totalW;
                }

                if (euclideanDistance(original, shifted) > convergenceThreshold) {
                    shiftedPoints[i] = shifted;
                    converged = false;
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Clustering done: %zu cluster points", shiftedPoints.size());
        return shiftedPoints;
    }

    // =========================================================================
    // Score function — identical to previous year
    // =========================================================================
    double getScore(const Frontier& frontier) {
        double yaw_to_frontier = std::atan2(
            frontier.coordinates.y - curr_drone_position_.y,
            frontier.coordinates.x - curr_drone_position_.x);
        double yaw_score = std::abs(yaw_to_frontier - drone_yaw_);
        if (yaw_score > M_PI) {
            yaw_score = 2.0 * M_PI - yaw_score;
        }
        return -k_distance_    * euclideanDistance(frontier.coordinates, curr_drone_position_)
               + k_neighborcount_ * static_cast<double>(frontier.neighborcount)
               - k_yaw_          * yaw_score;
    }

    // =========================================================================
    // Timer callback — publish goal only when exploring
    // (mirrors previous year's publish_goal)
    // =========================================================================
    void publishGoal() {
        if (statemachine_state_ != "EXPLORE_CAVE" && statemachine_state_ != "EXPLORE") {
            return;
        }
        if (!goal_available_) return;

        goal_pub_->publish(goal_message_);
        goal_point_pub_->publish(goal_point_);
        double gd = std::sqrt(
            std::pow(goal_point_.x - curr_drone_position_.x, 2) +
            std::pow(goal_point_.y - curr_drone_position_.y, 2) +
            std::pow(goal_point_.z - curr_drone_position_.z, 2));
        RCLCPP_INFO(this->get_logger(),
            "GOAL_PUB | goal=[%.1f,%.1f,%.1f] drone=[%.1f,%.1f,%.1f] dist=%.1fm",
            goal_point_.x, goal_point_.y, goal_point_.z,
            curr_drone_position_.x, curr_drone_position_.y, curr_drone_position_.z, gd);
    }

    // =========================================================================
    // Set goal message — mirrors previous year's set_goal_message
    // =========================================================================
    void setGoalMessage(const Frontier& best_frontier) {
        goal_message_.header.frame_id   = "world";
        goal_message_.header.stamp      = this->now();
        goal_message_.pose.position.x   = best_frontier.coordinates.x;
        goal_message_.pose.position.y   = best_frontier.coordinates.y;
        goal_message_.pose.position.z   = best_frontier.coordinates.z;

        double goal_yaw = std::atan2(
            best_frontier.coordinates.y - curr_drone_position_.y,
            best_frontier.coordinates.x - curr_drone_position_.x);
        tf2::Quaternion q;
        q.setRPY(0.0, 0.0, goal_yaw);
        goal_message_.pose.orientation.x = q.x();
        goal_message_.pose.orientation.y = q.y();
        goal_message_.pose.orientation.z = q.z();
        goal_message_.pose.orientation.w = q.w();

        goal_point_.x = best_frontier.coordinates.x;
        goal_point_.y = best_frontier.coordinates.y;
        goal_point_.z = best_frontier.coordinates.z;

        goal_available_ = true;
    }

    // =========================================================================
    // Select best frontier — mirrors previous year's select_frontier
    // =========================================================================
    void selectFrontier(const std::vector<Frontier>& points_sorted) {
        if (points_sorted.empty()) {
            RCLCPP_WARN(this->get_logger(),
                "FRONTIER | 0 candidates after filtering (min=%.0fm max=%.0fm). Drone=[%.1f, %.1f, %.1f]",
                min_frontier_distance_, max_frontier_distance_,
                curr_drone_position_.x, curr_drone_position_.y, curr_drone_position_.z);
            return;
        }

        double closest_dist = 1e9, farthest_dist = 0.0;
        Frontier best = points_sorted[0];
        best.score = getScore(best);

        for (const auto& f : points_sorted) {
            double d = euclideanDistance(f.coordinates, curr_drone_position_);
            closest_dist = std::min(closest_dist, d);
            farthest_dist = std::max(farthest_dist, d);
            Frontier candidate = f;
            candidate.score = getScore(candidate);
            if (candidate.score > best.score) {
                best = candidate;
            }
        }

        double best_dist = euclideanDistance(best.coordinates, curr_drone_position_);
        RCLCPP_INFO(this->get_logger(),
            "FRONTIER | %zu candidates [%.0f-%.0fm] | BEST=[%.1f,%.1f,%.1f] dist=%.1fm score=%.1f",
            points_sorted.size(), closest_dist, farthest_dist,
            best.coordinates.x, best.coordinates.y, best.coordinates.z,
            best_dist, best.score);

        if (best.isReachable &&
            euclideanDistance(curr_drone_position_, best.coordinates) < distance_limit_)
        {
            setGoalMessage(best);
        }
    }

    // =========================================================================
    // Publish RViz markers — mirrors previous year's publish_markers
    // =========================================================================
    void publishMarkers(const std::vector<Frontier>& points_sorted) {
        visualization_msgs::msg::MarkerArray markerArray;

        visualization_msgs::msg::Marker clear;
        clear.id     = 0;
        clear.ns     = "frontier";
        clear.action = visualization_msgs::msg::Marker::DELETEALL;
        markerArray.markers.push_back(clear);

        for (int i = 0; i < static_cast<int>(points_sorted.size()); ++i) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "world";
            m.header.stamp    = this->now();
            m.ns              = "frontier";
            m.id              = i + 1;
            m.type            = visualization_msgs::msg::Marker::CUBE;
            m.action          = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = points_sorted[i].coordinates.x;
            m.pose.position.y = points_sorted[i].coordinates.y;
            m.pose.position.z = points_sorted[i].coordinates.z;
            m.pose.orientation.w = 1.0;
            m.scale.x         = octomap_res_;
            m.scale.y         = octomap_res_;
            m.scale.z         = octomap_res_;
            m.color.r         = 1.0f;
            m.color.g         = 0.0f;
            m.color.b         = 0.0f;
            m.color.a         = 1.0f;
            markerArray.markers.push_back(m);
        }

        RCLCPP_INFO(this->get_logger(), "Publishing %zu frontier markers",
            points_sorted.size());
        frontier_pub_->publish(markerArray);
    }

    // =========================================================================
    // Sort / filter frontiers — mirrors previous year's sort_frontiers
    //   - Real neighborcount (cluster density)
    //   - 8-corner occupied-neighbor check (occ_neighbor_threshold)
    //   - Cave entrance filter (entry_tol = 25)
    //   - isReachable always true (same as prev year's working state)
    // =========================================================================
    std::vector<Frontier> sortFrontiers(
        const std::vector<Point3D>& frontiers_clustered,
        octomap::OcTree* octree)
    {
        RCLCPP_INFO(this->get_logger(), "Sorting frontiers (%zu clustered points)",
            frontiers_clustered.size());

        std::vector<Frontier> frontiers_sorted;

        for (const auto& f1 : frontiers_clustered) {
            Frontier frontier;
            frontier.coordinates  = f1;
            frontier.neighborcount = 0;

            // Count cluster density: how many clustered points are within octomap_res
            for (const auto& f2 : frontiers_clustered) {
                if (euclideanDistance(f1, f2) < static_cast<double>(octomap_res_)) {
                    frontier.neighborcount++;
                }
            }

            // Deduplication: skip if already have a frontier within octomap_res
            bool addflag = true;
            for (const auto& f3 : frontiers_sorted) {
                if (euclideanDistance(f3.coordinates, frontier.coordinates) <
                    static_cast<double>(octomap_res_))
                {
                    addflag = false;
                    break;
                }
            }

            // Count occupied corner-neighbors (8 corners, dx≠0 && dy≠0 && dz≠0)
            int nbscore = 0;
            for (int dx = -1; dx <= 1; ++dx) {
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dz = -1; dz <= 1; ++dz) {
                        if (dx != 0 && dy != 0 && dz != 0) {
                            octomap::point3d pt(
                                frontier.coordinates.x + octomap_res_ * dx,
                                frontier.coordinates.y + octomap_res_ * dy,
                                frontier.coordinates.z + octomap_res_ * dz);
                            octomap::OcTreeNode* result = octree->search(pt);
                            if (result != nullptr && octree->isNodeOccupied(result)) {
                                nbscore++;
                            }
                        }
                    }
                }
            }

            // Cave entrance filter: don't return to entrance once inside
            int  entry_tol         = 25;
            bool add_entry_frontier = true;
            if (curr_drone_position_.x < cave_entry_point_.x - entry_tol &&
                frontier.coordinates.x  > cave_entry_point_.x - entry_tol)
            {
                add_entry_frontier = false;
            }

            double dist_to_drone = euclideanDistance(frontier.coordinates, curr_drone_position_);

            if (addflag &&
                frontier.neighborcount > neighborcount_threshold_ &&
                add_entry_frontier &&
                nbscore < occ_neighbor_threshold_ &&
                dist_to_drone >= min_frontier_distance_ &&
                dist_to_drone <= max_frontier_distance_)
            {
                frontier.isReachable = true;
                frontiers_sorted.push_back(frontier);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Frontiers sorted: %zu", frontiers_sorted.size());
        return frontiers_sorted;
    }

    // =========================================================================
    // Main OctoMap callback — mirrors previous year's parseOctomap
    //   - Returns immediately if not in EXPLORE state
    //   - 8-corner neighbor check for frontier detection (dx≠0 && dy≠0 && dz≠0)
    //   - No pre-filters (no altitude, no XY distance, no safety distance)
    //   - Mean shift clustering → sort → select → publish markers
    // =========================================================================
    void parseOctomap(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        // Only process during exploration (same guard as previous year)
        if (statemachine_state_ != "EXPLORE_CAVE" && statemachine_state_ != "EXPLORE") {
            return;
        }

        // Skip stale pre-reset messages: wait until the map is fresh after reset.
        // The octomap_server publishes an empty map right after reset; once we
        // receive it, we clear the flag and resume normal processing.
        if (waiting_for_reset_) {
            octomap::AbstractOcTree* probe = octomap_msgs::msgToMap(*msg);
            octomap::OcTree* probe_tree = dynamic_cast<octomap::OcTree*>(probe);
            size_t leaf_count = probe_tree ? probe_tree->getNumLeafNodes() : 0;
            if (probe_tree) delete probe_tree;

            if (leaf_count > 100) {
                // Still the stale large map — skip it
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "Waiting for post-reset OctoMap (current has %zu nodes, skipping).",
                    leaf_count);
                return;
            }
            // Map is now fresh (empty or very small) — resume
            waiting_for_reset_ = false;
            RCLCPP_INFO(this->get_logger(), "Post-reset OctoMap received. Resuming frontier detection.");
        }

        octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*msg);
        octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);

        if (!octree) {
            RCLCPP_ERROR(this->get_logger(), "Failed to parse octomap message");
            if (tree) delete tree;
            return;
        }

        octomap_res_ = static_cast<float>(octree->getResolution());
        RCLCPP_INFO(this->get_logger(), "Parsing OctoMap. Resolution: %.2f, Nodes: %zu",
            octomap_res_, octree->getNumLeafNodes());

        std::vector<Frontier> frontiers;

        // Frontier detection: free cell with at least one unknown corner-neighbor
        // (identical logic to previous year: dx != 0 && dy != 0 && dz != 0)
        for (auto it = octree->begin_leafs(); it != octree->end_leafs(); ++it) {
            if (!octree->isNodeOccupied(*it)) {
                for (int dx = -1; dx <= 1; ++dx) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dz = -1; dz <= 1; ++dz) {
                            if (dx != 0 && dy != 0 && dz != 0) {
                                octomap::OcTreeKey neighborKey(
                                    it.getKey().k[0] + dx,
                                    it.getKey().k[1] + dy,
                                    it.getKey().k[2] + dz);
                                if (octree->search(neighborKey) == nullptr) {
                                    Frontier fp;
                                    fp.coordinates.x = it.getX();
                                    fp.coordinates.y = it.getY();
                                    fp.coordinates.z = it.getZ();
                                    frontiers.push_back(fp);
                                }
                            }
                        }
                    }
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Found %zu raw frontier points", frontiers.size());

        if (frontiers.empty()) {
            delete octree;
            return;
        }

        // ── Pre-filter: keep only frontiers in [min_frontier_distance_, max_frontier_distance_] ──
        // Reduces points before mean-shift and avoids clustering irrelevant regions.
        {
            std::vector<Frontier> in_range;
            in_range.reserve(frontiers.size());
            for (const auto& f : frontiers) {
                double d = euclideanDistance(f.coordinates, curr_drone_position_);
                if (d >= min_frontier_distance_ && d <= max_frontier_distance_) {
                    in_range.push_back(f);
                }
            }
            RCLCPP_INFO(this->get_logger(),
                "After distance pre-filter [%.0f-%.0fm]: %zu frontier points",
                min_frontier_distance_, max_frontier_distance_, in_range.size());
            frontiers = std::move(in_range);
        }

        if (frontiers.empty()) {
            delete octree;
            return;
        }

        // ── Pre-filter 2: random subsample to max_frontiers_ ──
        // Caps mean-shift O(n²) cost.  With max_frontiers_=3000 and bandwidth=17,
        // clustering converges in a few seconds instead of hanging indefinitely.
        if (static_cast<int>(frontiers.size()) > max_frontiers_) {
            std::mt19937 rng(42);
            std::shuffle(frontiers.begin(), frontiers.end(), rng);
            frontiers.resize(max_frontiers_);
            RCLCPP_INFO(this->get_logger(),
                "Subsampled to %d frontier points for mean-shift.", max_frontiers_);
        }

        std::vector<Point3D> frontierpoints_clustered =
            meanShiftClustering(frontiers, 1.0);

        std::vector<Frontier> frontierpoints_sorted =
            sortFrontiers(frontierpoints_clustered, octree);

        selectFrontier(frontierpoints_sorted);
        publishMarkers(frontierpoints_sorted);

        delete octree;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FrontierDetector>());
    rclcpp::shutdown();
    return 0;
}
