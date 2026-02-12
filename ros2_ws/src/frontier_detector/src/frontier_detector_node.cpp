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
#include <nav_msgs/srv/get_plan.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <vector>
#include <cmath>
#include <memory>
#include <map>

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
        // Parameters
        this->declare_parameter("neighborcount_threshold", 5);
        this->declare_parameter("bandwidth", 1.0);
        this->declare_parameter("k_distance", 1.0);
        this->declare_parameter("k_neighborcount", 1.0);
        this->declare_parameter("k_yaw", 10.0);
        this->declare_parameter("distance_limit", 10.0);
        this->declare_parameter("publish_goal_frequency", 1.0);
        this->declare_parameter("occ_neighbor_threshold", 5);
        this->declare_parameter("altitude_tolerance", 3.0);
        this->declare_parameter("safety_distance", 0.5);
        this->declare_parameter("min_passage_width", 0.5);

        neighborcount_threshold_ = this->get_parameter("neighborcount_threshold").as_int();
        bandwidth_ = this->get_parameter("bandwidth").as_double();
        k_distance_ = this->get_parameter("k_distance").as_double();
        k_neighborcount_ = this->get_parameter("k_neighborcount").as_double();
        k_yaw_ = this->get_parameter("k_yaw").as_double();
        distance_limit_ = this->get_parameter("distance_limit").as_double();
        publish_goal_frequency_ = this->get_parameter("publish_goal_frequency").as_double();
        occ_neighbor_threshold_ = this->get_parameter("occ_neighbor_threshold").as_int();
        altitude_tolerance_ = this->get_parameter("altitude_tolerance").as_double();
        safety_distance_ = this->get_parameter("safety_distance").as_double();
        min_passage_width_ = this->get_parameter("min_passage_width").as_double();

        // Subscribers & Publishers
        // Use transient_local QoS to match the octomap_server publisher
        auto octomap_qos = rclcpp::QoS(1).transient_local().reliable();
        octomap_sub_ = this->create_subscription<octomap_msgs::msg::Octomap>(
            "octomap_binary", octomap_qos, std::bind(&FrontierDetector::octomapCallback, this, std::placeholders::_1));
        
        pose_sub_ = this->create_subscription<geometry_msgs::msg::PoseStamped>(
            "pose_est", 1, std::bind(&FrontierDetector::poseCallback, this, std::placeholders::_1));

        state_sub_ = this->create_subscription<std_msgs::msg::String>(
            "stm_mode", 1, std::bind(&FrontierDetector::stateCallback, this, std::placeholders::_1));

        frontier_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("frontiers", 1);
        goal_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("frontier_goal_pose", 1);
        goal_point_pub_ = this->create_publisher<geometry_msgs::msg::Point>("frontier_goal", 1);

        // Service Server for starting exploration override
        start_exploration_srv_ = this->create_service<std_srvs::srv::Trigger>(
            "start_exploration", std::bind(&FrontierDetector::startExplorationCallback, this, std::placeholders::_1, std::placeholders::_2));

        // Timer
        timer_ = this->create_wall_timer(
            std::chrono::duration<double>(1.0 / publish_goal_frequency_), 
            std::bind(&FrontierDetector::timerCallback, this));

        // TF
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        cave_entry_point_ = {-321.0, 10.0, 15.0};
        RCLCPP_INFO(this->get_logger(), "Frontier Detector Initialized");
    }

private:
    int neighborcount_threshold_;
    double bandwidth_;
    double k_distance_;
    double k_neighborcount_;
    double k_yaw_;
    double distance_limit_;
    double publish_goal_frequency_;
    int occ_neighbor_threshold_;
    double altitude_tolerance_;
    double safety_distance_;
    double min_passage_width_;

    rclcpp::Subscription<octomap_msgs::msg::Octomap>::SharedPtr octomap_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr pose_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr frontier_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr goal_pub_;
    rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr goal_point_pub_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_exploration_srv_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    Point3D curr_drone_position_;
    double drone_yaw_ = 0.0;
    std::string current_state_ = "IDLE";
    bool exploration_active_ = false;
    double octomap_res_ = 0.5;
    octomap_msgs::msg::Octomap::SharedPtr cached_octomap_msg_;
    bool octomap_dirty_ = false; // true when we have a new octomap not yet processed
    int timer_tick_ = 0;  // counts timer ticks for periodic reprocessing

    geometry_msgs::msg::PoseStamped current_goal_pose_;
    geometry_msgs::msg::Point current_goal_point_;
    bool goal_available_ = false;

    Point3D cave_entry_point_;

    void startExplorationCallback(const std::shared_ptr<std_srvs::srv::Trigger::Request> req,
                                  std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        exploration_active_ = true;
        res->success = true;
        res->message = "Exploration triggered in Frontier Detector";
        RCLCPP_INFO(this->get_logger(), "Exploration triggered via service.");
        (void)req;
    }

    void stateCallback(const std_msgs::msg::String::SharedPtr msg) {
        current_state_ = msg->data;
        if (current_state_ == "EXPLORE_CAVE" || current_state_ == "EXPLORE") {
             exploration_active_ = true;
        } else {
             exploration_active_ = false;
        }
    }

    void poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        curr_drone_position_.x = msg->pose.position.x;
        curr_drone_position_.y = msg->pose.position.y;
        curr_drone_position_.z = msg->pose.position.z;

        tf2::Quaternion q(
            msg->pose.orientation.x,
            msg->pose.orientation.y,
            msg->pose.orientation.z,
            msg->pose.orientation.w);
        tf2::Matrix3x3 m(q);
        double roll, pitch;
        m.getRPY(roll, pitch, drone_yaw_);
    }

    void octomapCallback(const octomap_msgs::msg::Octomap::SharedPtr msg) {
        // Always cache the latest octomap
        cached_octomap_msg_ = msg;
        octomap_dirty_ = true;
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
            "Received octomap update (cached). exploration_active=%d", exploration_active_);

        // If exploration is active, process immediately
        if (exploration_active_) {
            processOctomap();
        }
    }

    void processOctomap() {
        if (!cached_octomap_msg_) return;

        octomap::AbstractOcTree* tree = octomap_msgs::msgToMap(*cached_octomap_msg_);
        octomap::OcTree* octree = dynamic_cast<octomap::OcTree*>(tree);

        if (!octree) {
            RCLCPP_ERROR(this->get_logger(), "Failed to cast to OcTree");
            if(tree) delete tree;
            return;
        }

        octomap_res_ = octree->getResolution();
        RCLCPP_INFO(this->get_logger(), 
            "Processing Octomap. Resolution: %.2f, Leafs: %zu", octomap_res_, octree->getNumLeafNodes());

        std::vector<Frontier> frontiers;

        for (auto it = octree->begin_leafs(); it != octree->end_leafs(); ++it) {
            if (!octree->isNodeOccupied(*it)) {
                double fx = it.getX();
                double fy = it.getY();
                double fz = it.getZ();

                // Pre-filter: skip frontiers too far above/below drone altitude
                double dz_abs = std::abs(fz - curr_drone_position_.z);
                if (dz_abs > altitude_tolerance_) continue;

                // Pre-filter: only consider frontiers within distance_limit of drone (XY only)
                double dist_xy = std::sqrt(
                    std::pow(fx - curr_drone_position_.x, 2) +
                    std::pow(fy - curr_drone_position_.y, 2));
                if (dist_xy > distance_limit_) continue;

                bool is_frontier = false;
                for (int dx = -1; dx <= 1 && !is_frontier; ++dx) {
                    for (int dy = -1; dy <= 1 && !is_frontier; ++dy) {
                        for (int dz = -1; dz <= 1 && !is_frontier; ++dz) {
                            if (dx == 0 && dy == 0 && dz == 0) continue;
                            octomap::OcTreeKey neighborKey = it.getKey();
                            neighborKey[0] += dx;
                            neighborKey[1] += dy;
                            neighborKey[2] += dz;
                            if (octree->search(neighborKey) == nullptr) {
                                is_frontier = true;
                            }
                        }
                    }
                }

                if (is_frontier) {
                    Frontier f;
                    f.coordinates.x = fx;
                    f.coordinates.y = fy;
                    f.coordinates.z = fz;
                    frontiers.push_back(f);
                }
            }
        }

        RCLCPP_INFO(this->get_logger(), "Found %zu frontier points (within %.1fm)", frontiers.size(), distance_limit_);

        if (frontiers.empty()) {
            delete octree;
            return;
        }

        // Fast voxel-grid clustering (replaces O(n²) mean shift)
        std::vector<Point3D> clustered_points = voxelGridCluster(frontiers, bandwidth_);
        RCLCPP_INFO(this->get_logger(), "Clustered into %zu points", clustered_points.size());

        std::vector<Frontier> sorted_frontiers = sortFrontiers(clustered_points, octree);
        RCLCPP_INFO(this->get_logger(), "Sorted frontiers size: %zu", sorted_frontiers.size());

        if (!sorted_frontiers.empty()) {
            selectFrontier(sorted_frontiers);
            publishMarkers(sorted_frontiers);
        }

        octomap_dirty_ = false;
        delete octree;
    }

    // Fast O(n) voxel grid clustering - bin points into voxels and return centroids
    std::vector<Point3D> voxelGridCluster(const std::vector<Frontier>& points, double voxel_size) {
        // Use a map keyed by voxel indices to accumulate points
        struct VoxelKey {
            int x, y, z;
            bool operator<(const VoxelKey& o) const {
                if (x != o.x) return x < o.x;
                if (y != o.y) return y < o.y;
                return z < o.z;
            }
        };
        struct VoxelAccum {
            double sum_x = 0, sum_y = 0, sum_z = 0;
            int count = 0;
        };
        
        // Use a larger voxel size for clustering (2m voxels)
        double cluster_voxel = std::max(voxel_size, 2.0);
        std::map<VoxelKey, VoxelAccum> voxels;

        for (const auto& f : points) {
            VoxelKey key;
            key.x = static_cast<int>(std::floor(f.coordinates.x / cluster_voxel));
            key.y = static_cast<int>(std::floor(f.coordinates.y / cluster_voxel));
            key.z = static_cast<int>(std::floor(f.coordinates.z / cluster_voxel));
            
            auto& v = voxels[key];
            v.sum_x += f.coordinates.x;
            v.sum_y += f.coordinates.y;
            v.sum_z += f.coordinates.z;
            v.count++;
        }

        std::vector<Point3D> result;
        for (const auto& [key, v] : voxels) {
            // Only keep clusters with enough points (minimum 3 frontier cells)
            if (v.count >= 3) {
                Point3D p;
                p.x = v.sum_x / v.count;
                p.y = v.sum_y / v.count;
                p.z = v.sum_z / v.count;
                result.push_back(p);
            }
        }
        return result;
    }

    std::vector<Point3D> meanShiftClustering(const std::vector<Frontier>& points, double threshold) {
        std::vector<Point3D> shifted_points;
        for (const auto& p : points) shifted_points.push_back(p.coordinates);

        bool converged = false;
        while (!converged) {
            converged = true;
            for (size_t i = 0; i < shifted_points.size(); ++i) {
                Point3D original = shifted_points[i];
                Point3D shift = {0,0,0};
                double total_weight = 0;

                for (const auto& p : points) {
                    double dist = euclideanDistance(original, p.coordinates);
                    double weight = std::exp(-(dist*dist)/(2*bandwidth_*bandwidth_));
                    shift.x += p.coordinates.x * weight;
                    shift.y += p.coordinates.y * weight;
                    shift.z += p.coordinates.z * weight;
                    total_weight += weight;
                }
                
                if (total_weight > 0) {
                    shift.x /= total_weight;
                    shift.y /= total_weight;
                    shift.z /= total_weight;
                }

                if (euclideanDistance(original, shift) > threshold) {
                    shifted_points[i] = shift;
                    converged = false;
                }
            }
        }
        return shifted_points;
    }

    std::vector<Frontier> sortFrontiers(const std::vector<Point3D>& clustered, octomap::OcTree* octree) {
        std::vector<Frontier> sorted;
        int skipped_outside = 0;
        int skipped_obstacle = 0;
        int skipped_narrow = 0;
        
        for (const auto& pt : clustered) {
            Frontier f;
            f.coordinates = pt;
            f.neighborcount = neighborcount_threshold_ + 1; 

            // Filter frontiers outside the cave when drone is inside
            bool drone_inside_cave = (curr_drone_position_.x < cave_entry_point_.x);
            bool frontier_outside_cave = (f.coordinates.x > cave_entry_point_.x + 2.0);
            
            if (drone_inside_cave && frontier_outside_cave) {
                skipped_outside++;
                continue;
            }
            
            if (octree) {
                double res = octree->getResolution();
                
                // Safety check: reject frontiers too close to obstacles
                if (safety_distance_ > 0.0) {
                    bool too_close = false;
                    int steps = static_cast<int>(std::ceil(safety_distance_ / res));
                    
                    for (int dx = -steps; dx <= steps && !too_close; dx += std::max(1, steps/2)) {
                        for (int dy = -steps; dy <= steps && !too_close; dy += std::max(1, steps/2)) {
                            for (int dz = -steps; dz <= steps && !too_close; dz += std::max(1, steps/2)) {
                                double cx = f.coordinates.x + dx * res;
                                double cy = f.coordinates.y + dy * res;
                                double cz = f.coordinates.z + dz * res;
                                octomap::OcTreeNode* node = octree->search(cx, cy, cz);
                                if (node && octree->isNodeOccupied(node)) {
                                    double dist_to_occ = std::sqrt(
                                        std::pow(f.coordinates.x - cx, 2) +
                                        std::pow(f.coordinates.y - cy, 2) +
                                        std::pow(f.coordinates.z - cz, 2));
                                    if (dist_to_occ < safety_distance_) {
                                        too_close = true;
                                    }
                                }
                            }
                        }
                    }
                    
                    if (too_close) {
                        skipped_obstacle++;
                        continue;
                    }
                }
                
                // Passage width check: reject frontiers in narrow gaps/crevices
                // A narrow gap (like the dark gap between rocks) will have mostly
                // unknown cells around it, while valid passages have known-free cells.
                if (min_passage_width_ > 0.0) {
                    int check_steps = static_cast<int>(std::ceil(min_passage_width_ / res));
                    int free_count = 0;
                    int total_checked = 0;
                    
                    // Sample cells in a sphere around the frontier
                    for (int dx = -check_steps; dx <= check_steps; dx += 2) {
                        for (int dy = -check_steps; dy <= check_steps; dy += 2) {
                            for (int dz = -check_steps; dz <= check_steps; dz += 2) {
                                double cx = f.coordinates.x + dx * res;
                                double cy = f.coordinates.y + dy * res;
                                double cz = f.coordinates.z + dz * res;
                                double dist = std::sqrt(dx*dx + dy*dy + dz*dz) * res;
                                if (dist > min_passage_width_) continue;
                                
                                total_checked++;
                                octomap::OcTreeNode* node = octree->search(cx, cy, cz);
                                if (node && !octree->isNodeOccupied(node)) {
                                    free_count++;
                                }
                                // If node is nullptr (unknown) or occupied, it's not free
                            }
                        }
                    }
                    
                    // If less than 40% of surrounding cells are known-free,
                    // this is likely a narrow gap or a crevice, not a real passage
                    double free_ratio = (total_checked > 0) ? 
                        static_cast<double>(free_count) / total_checked : 0.0;
                    if (free_ratio < 0.4) {
                        skipped_narrow++;
                        continue;
                    }
                }
            }
            
            sorted.push_back(f);
        }
        
        RCLCPP_INFO(this->get_logger(), 
            "After filtering: %zu remain (skipped %d outside, %d close to walls, %d narrow gaps)",
            sorted.size(), skipped_outside, skipped_obstacle, skipped_narrow);
        return sorted;
    }

    void selectFrontier(std::vector<Frontier>& frontiers) {
        if (frontiers.empty()) return;

        double best_score = -1e9;
        Frontier best_f = frontiers[0];

        for (auto& f : frontiers) {
            double dist = euclideanDistance(f.coordinates, curr_drone_position_);
            double yaw_to_f = std::atan2(f.coordinates.y - curr_drone_position_.y, f.coordinates.x - curr_drone_position_.x);
            double yaw_diff = std::abs(yaw_to_f - drone_yaw_);
            if (yaw_diff > M_PI) yaw_diff = 2*M_PI - yaw_diff;

            f.score = -k_distance_ * dist + k_neighborcount_ * f.neighborcount - k_yaw_ * yaw_diff;
            
            // Strong bias toward deeper cave (more negative X = deeper)
            // The cave goes in -X direction; more negative X means deeper in the cave
            double forward_progress = curr_drone_position_.x - f.coordinates.x; // positive when frontier is deeper
            f.score += forward_progress * 2.0;
            
            // Penalize frontiers that go back toward the entrance (positive X direction)
            if (f.coordinates.x > curr_drone_position_.x) {
                f.score -= 10.0; // Strong penalty for going backward
            }

            if (f.score > best_score) {
                best_score = f.score;
                best_f = f;
            }
        }

        RCLCPP_INFO(this->get_logger(), "Selected frontier goal: [%.2f, %.2f, %.2f] score=%.2f",
            best_f.coordinates.x, best_f.coordinates.y, best_f.coordinates.z, best_score);

        current_goal_point_.x = best_f.coordinates.x;
        current_goal_point_.y = best_f.coordinates.y;
        // Force goal Z to drone's current altitude for safe flight
        current_goal_point_.z = curr_drone_position_.z;

        RCLCPP_INFO(this->get_logger(), "Goal Z clamped to drone altitude: %.2f (frontier was %.2f)",
            current_goal_point_.z, best_f.coordinates.z);

        current_goal_pose_.header.frame_id = "world";
        current_goal_pose_.header.stamp = this->now();
        current_goal_pose_.pose.position = current_goal_point_;
        current_goal_pose_.pose.orientation.w = 1.0; 

        goal_available_ = true;
    }

    void publishMarkers(const std::vector<Frontier>& frontiers) {
        visualization_msgs::msg::MarkerArray markers;
        visualization_msgs::msg::Marker clear_marker;
        clear_marker.action = visualization_msgs::msg::Marker::DELETEALL;
        markers.markers.push_back(clear_marker);

        int id = 0;
        for (const auto& f : frontiers) {
            visualization_msgs::msg::Marker m;
            m.header.frame_id = "world";
            m.header.stamp = this->now();
            m.ns = "frontiers";
            m.id = ++id;
            m.type = visualization_msgs::msg::Marker::CUBE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = f.coordinates.x;
            m.pose.position.y = f.coordinates.y;
            m.pose.position.z = f.coordinates.z;
            m.scale.x = octomap_res_;
            m.scale.y = octomap_res_;
            m.scale.z = octomap_res_;
            m.color.r = 1.0;
            m.color.a = 1.0;
            markers.markers.push_back(m);
        }
        frontier_pub_->publish(markers);
    }

    void timerCallback() {
        if (!exploration_active_) return;
        
        timer_tick_++;
        
        // Reprocess cached octomap every 3 ticks (~3 seconds at 1Hz)
        // This ensures frontiers are refreshed as the drone moves,
        // even if no new octomap messages arrive
        bool should_reprocess = (cached_octomap_msg_ != nullptr) && 
            (octomap_dirty_ || (timer_tick_ % 3 == 0));
        
        if (should_reprocess) {
            RCLCPP_INFO(this->get_logger(), "Reprocessing octomap (tick=%d, dirty=%d)", 
                timer_tick_, octomap_dirty_ ? 1 : 0);
            processOctomap();
        }
        
        // Always publish the current goal if we have one
        if (goal_available_) {
            // Update the goal Z to current drone altitude (it changes as drone moves)
            current_goal_point_.z = curr_drone_position_.z;
            current_goal_pose_.pose.position = current_goal_point_;
            current_goal_pose_.header.stamp = this->now();
            
            goal_pub_->publish(current_goal_pose_);
            goal_point_pub_->publish(current_goal_point_);
            RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "Publishing goal: [%.2f, %.2f, %.2f]",
                current_goal_point_.x, current_goal_point_.y, current_goal_point_.z);
        }
    }

    double euclideanDistance(const Point3D& a, const Point3D& b) {
        return std::sqrt(std::pow(a.x-b.x, 2) + std::pow(a.y-b.y, 2) + std::pow(a.z-b.z, 2));
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<FrontierDetector>());
    rclcpp::shutdown();
    return 0;
}
