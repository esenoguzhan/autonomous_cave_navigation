import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseStamped, Twist, Transform
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from nav_msgs.msg import OccupancyGrid
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
import numpy as np
import threading
import time
import math

class CaveExplorer(Node):
    def __init__(self):
        super().__init__('cave_explorer')
        
        # Parameters
        self.declare_parameter('map_topic', 'projected_map')
        self.map_topic = self.get_parameter('map_topic').value
        
        self.declare_parameter('base_frame', 'base_link')
        self.base_frame = self.get_parameter('base_frame').value

        self.declare_parameter('desired_state_topic', 'desired_state')
        self.desired_state_topic = self.get_parameter('desired_state_topic').value
        
        # State
        self.map_data = None
        self.map_info = None
        self.map_frame = 'map'
        self.map_lock = threading.Lock()
        self.exploring = False
        self.exploration_thread = None
        self.current_goal = None
        
        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Callback Group
        self.callback_group = ReentrantCallbackGroup()

        # Subscription
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10,
            callback_group=self.callback_group
        )
        self.get_logger().info(f'Subscribed to {self.map_topic}')

        # Publisher
        self.desired_state_pub = self.create_publisher(
            MultiDOFJointTrajectoryPoint,
            self.desired_state_topic,
            10
        )

        # Service
        self.srv = self.create_service(Trigger, 'start_exploration', self.start_exploration_callback, callback_group=self.callback_group)
        self.get_logger().info('Service start_exploration ready')

    def start_exploration_callback(self, request, response):
        if self.exploring:
            response.success = False
            response.message = "Already exploring"
            return response

        if self.map_data is None:
            response.success = False
            response.message = "Map not yet received"
            return response
        
        self.exploring = True
        self.exploration_thread = threading.Thread(target=self.exploration_loop, daemon=True)
        self.exploration_thread.start()
        
        response.success = True
        response.message = "Exploration started"
        self.get_logger().info("Exploration started via service call")
        return response

    def map_callback(self, msg):
        with self.map_lock:
            self.map_data = np.array(msg.data).reshape((msg.info.height, msg.info.width))
            self.map_info = msg.info
            self.map_frame = msg.header.frame_id

    def get_robot_pose(self):
        try:
            map_frame = self.map_frame
            now = rclpy.time.Time()
            if not self.tf_buffer.can_transform(map_frame, self.base_frame, now):
                if not self.tf_buffer.can_transform(map_frame, self.base_frame, rclpy.time.Time()):
                    return None
                
            trans = self.tf_buffer.lookup_transform(map_frame, self.base_frame, rclpy.time.Time())
            
            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = trans.transform.translation.x
            pose.pose.position.y = trans.transform.translation.y
            pose.pose.position.z = trans.transform.translation.z
            pose.pose.orientation = trans.transform.rotation
            return pose
        except Exception as e:
            self.get_logger().warn(f'TF Error: {e}')
            return None

    def get_frontiers(self):
        with self.map_lock:
            if self.map_data is None:
                return []
            
            grid = self.map_data
            info = self.map_info
            
            free_mask = (grid == 0)
            unknown_mask = (grid == -1)
            
            # Simple edge detection for frontiers
            u_up = np.roll(unknown_mask, -1, axis=0)
            u_down = np.roll(unknown_mask, 1, axis=0)
            u_left = np.roll(unknown_mask, -1, axis=1)
            u_right = np.roll(unknown_mask, 1, axis=1)
            
            u_up[-1, :] = False
            u_down[0, :] = False
            u_left[:, -1] = False
            u_right[:, 0] = False
            
            has_unknown_neighbor = u_up | u_down | u_left | u_right
            frontier_mask = free_mask & has_unknown_neighbor
            
            y_idxs, x_idxs = np.where(frontier_mask)
            
            if len(x_idxs) == 0:
                return []
            
            points = []
            res = info.resolution
            ox = info.origin.position.x
            oy = info.origin.position.y
            
            for y, x in zip(y_idxs, x_idxs):
                wx = (x + 0.5) * res + ox
                wy = (y + 0.5) * res + oy
                points.append([wx, wy])
            
            return points

    def publish_desired_state(self, x, y, z, yaw=0.0):
        msg = MultiDOFJointTrajectoryPoint()
        
        # Position
        transform = Transform()
        transform.translation.x = x
        transform.translation.y = y
        transform.translation.z = z
        
        # Yaw (simple, no rotation)
        # Using identity quaternion for now or simple yaw if needed
        # w = cos(theta/2), z = sin(theta/2)
        transform.rotation.w = math.cos(yaw / 2.0)
        transform.rotation.z = math.sin(yaw / 2.0)
        
        msg.transforms.append(transform)
        
        # Velocities (zero)
        vel = Twist()
        msg.velocities.append(vel)
        
        # Accelerations (zero)
        accel = Twist()
        msg.accelerations.append(accel)
        
        self.desired_state_pub.publish(msg)

    def exploration_loop(self):
        # Wait for map
        while self.map_data is None and self.exploring:
            self.get_logger().info("Waiting for map data...", throttle_duration_sec=2.0)
            time.sleep(1.0)

        self.get_logger().info("Map received. Starting loop.")

        while rclpy.ok() and self.exploring:
            robot_pose = self.get_robot_pose()
            if robot_pose is None:
                self.get_logger().warn("Unknown robot pose")
                time.sleep(1.0)
                continue

            rx = robot_pose.pose.position.x
            ry = robot_pose.pose.position.y
            rz = robot_pose.pose.position.z

            # Check if we reached the goal
            if self.current_goal:
                dist = math.hypot(self.current_goal[0] - rx, self.current_goal[1] - ry)
                if dist < 0.5:
                    self.get_logger().info("Frontier reached.")
                    self.current_goal = None
                else:
                    # Continue moving to goal
                    # Keep valid Z altitude (e.g. current or target)
                    self.publish_desired_state(self.current_goal[0], self.current_goal[1], 15.0)
                    time.sleep(0.1)
                    continue

            # Find new goal
            frontiers = self.get_frontiers()
            if not frontiers:
                self.get_logger().info("No frontiers found. Exploration FINISHED.")
                self.exploring = False
                break

            # Filter frontiers
            f_points = np.array(frontiers)
            
            # Remove points too close to robot (already visited/explored area)
            dists_to_robot = np.linalg.norm(f_points - np.array([rx, ry]), axis=1)
            valid_indices = np.where(dists_to_robot > 1.5)[0]
            
            if len(valid_indices) == 0:
                self.get_logger().info("No valid frontiers (all too close). Finished?")
                # Maybe try closer ones if we are really stuck, but for now stop or retry
                self.exploring = False
                break
                
            f_points = f_points[valid_indices]
            
            # Score remaining points
            # Prefer points that are closer (but > 1.5m away) to minimize travel, 
            # or maybe add heuristic for "direction"
            dists = np.linalg.norm(f_points - np.array([rx, ry]), axis=1)
            sorted_indices = np.argsort(dists)
            
            if len(sorted_indices) > 0:
                target = f_points[sorted_indices[0]]
                self.current_goal = (target[0], target[1])
                self.get_logger().info(f"New frontier: ({target[0]:.2f}, {target[1]:.2f})")
            else:
                time.sleep(1.0)
                
            time.sleep(0.5)

def main(args=None):
    rclpy.init(args=args)
    node = CaveExplorer()
    
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
