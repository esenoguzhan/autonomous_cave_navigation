import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, HistoryPolicy
from geometry_msgs.msg import PoseStamped, Twist, Transform, PoseArray, Pose
from trajectory_msgs.msg import MultiDOFJointTrajectoryPoint
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Empty as EmptyMsg
from std_srvs.srv import Trigger
from trajectory_planner.srv import ExecuteTrajectory
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
        
        self.use_sim_time = self.get_parameter('use_sim_time').value

        self.declare_parameter('desired_state_topic', 'desired_state')
        self.desired_state_topic = self.get_parameter('desired_state_topic').value
        
        # Exploration Parameters
        self.declare_parameter('exploration_altitude', 15.0)
        self.exploration_altitude = self.get_parameter('exploration_altitude').value
        
        self.declare_parameter('safety_distance_m', 0.8)
        self.safety_distance_m = self.get_parameter('safety_distance_m').value
        
        self.declare_parameter('min_frontier_distance_m', 2.0)
        self.min_frontier_distance_m = self.get_parameter('min_frontier_distance_m').value
        
        self.declare_parameter('max_path_timeout_s', 180.0)
        self.max_path_timeout_s = self.get_parameter('max_path_timeout_s').value
        
        self.declare_parameter('path_smoothing_epsilon', 0.1)
        self.path_smoothing_epsilon = self.get_parameter('path_smoothing_epsilon').value
        
        self.declare_parameter('visited_radius', 3.0)
        self.visited_radius = self.get_parameter('visited_radius').value

        self.declare_parameter('exploration_direction_x', -1.0) # -1.0 for negative X, 1.0 for positive X
        self.exploration_direction_x = self.get_parameter('exploration_direction_x').value

        self.declare_parameter('forward_bias', 5.0) # How much to favor moving "forward" vs "closest"
        self.forward_bias = self.get_parameter('forward_bias').value
        
        # State
        self.map_data = None
        self.map_info = None
        self.map_frame = 'map'
        self.map_lock = threading.Lock()
        self.exploring = False
        self.exploration_thread = None
        self.current_goal = None
        self.trajectory_done_event = threading.Event()
        self.visited_goals = []
        
        # TF Buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # Callback Group
        self.callback_group = ReentrantCallbackGroup()

        # QoS Profile for map (needs to be Transient Local for Octomap)
        map_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscription
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            map_qos,
            callback_group=self.callback_group
        )
        self.get_logger().info(f'Subscribed to {self.map_topic}')

        # Trajectory Complete Subscription
        self.create_subscription(
            EmptyMsg,
            'trajectory_complete',
            self.trajectory_complete_callback,
            10,
            callback_group=self.callback_group
        )

        # Service Client for Trajectory Planner
        self.nav_client = self.create_client(ExecuteTrajectory, 'start_navigation', callback_group=self.callback_group)

        # Service to start exploration
        self.srv = self.create_service(Trigger, 'start_exploration', self.start_exploration_callback, callback_group=self.callback_group)
        self.get_logger().info('Service start_exploration ready')

    def trajectory_complete_callback(self, msg):
        self.get_logger().info("Received trajectory complete signal.")
        self.trajectory_done_event.set()

    def start_exploration_callback(self, request, response):
        if self.exploring:
            response.success = False
            response.message = "Already exploring"
            return response

        if self.map_data is None:
            response.success = False
            response.message = "Map not yet received"
            return response
        
        self.visited_goals = [] # Reset visited goals on new start
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
            if not self.tf_buffer.can_transform(map_frame, self.base_frame, rclpy.time.Time()):
                self.get_logger().warn(f"Cannot transform from {map_frame} to {self.base_frame}", throttle_duration_sec=2.0)
                return None
                
            trans = self.tf_buffer.lookup_transform(map_frame, self.base_frame, rclpy.time.Time())
            
            # Debug log to verify frames (throttle to avoid spam)
            # self.get_logger().info(f"Resolved robot pose in {map_frame}", throttle_duration_sec=10.0)
            
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
            occupied_mask = (grid > 50)
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
            
            # --- FILTER FRONTIERS NEAR OBSTACLES (WALLS) ---
            # Dilate occupied mask to find "unsafe" areas
            # Using simple neighborhood check for speed instead of full dilation
            # Check 2-cell radius (approx 0.2m-0.4m depending on resolution)
            
            y_idxs, x_idxs = np.where(frontier_mask)
            if len(x_idxs) == 0:
                return []
                
            points = []
            res = info.resolution
            ox = info.origin.position.x
            oy = info.origin.position.y
            width = info.width
            height = info.height
            
            # Safety radius in cells
            safety_radius_m = self.safety_distance_m
            safety_radius_cells = int(safety_radius_m / res)
            
            for y, x in zip(y_idxs, x_idxs):
                # Check neighbors for collision
                is_safe = True
                for dy in range(-safety_radius_cells, safety_radius_cells + 1):
                    for dx in range(-safety_radius_cells, safety_radius_cells + 1):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if grid[ny, nx] > 50: # If wall nearby
                                is_safe = False
                                break
                    if not is_safe: break
                
                if is_safe:
                    wx = (x + 0.5) * res + ox
                    wy = (y + 0.5) * res + oy
                    points.append([wx, wy])
            
            return points

    def get_occupancy(self, x, y, width, height, data):
        if 0 <= x < width and 0 <= y < height:
            return data[y * width + x]
        return 100 # Treat out of bounds as occupied
        
    def is_cell_safe(self, x, y, width, height, data, safety_margin_cells=3):
        # Check a small radius around the cell for obstacles
        for dy in range(-safety_margin_cells, safety_margin_cells + 1):
            for dx in range(-safety_margin_cells, safety_margin_cells + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    if data[ny * width + nx] > 50:
                        return False
                else:
                    # Treat out-of-bounds as unsafe to be conservative
                    return False
        return True

    def a_star_search(self, start_world, goal_world, grid, info):
        # detailed A* implementation
        res = info.resolution
        origin_x = info.origin.position.x
        origin_y = info.origin.position.y
        width = info.width
        height = info.height
        grid_flat = grid.flatten() # Pre-flatten for speed
        
        # Convert to grid coordinates
        start_grid = (
            int((start_world[0] - origin_x) / res),
            int((start_world[1] - origin_y) / res)
        )
        goal_grid = (
            int((goal_world[0] - origin_x) / res),
            int((goal_world[1] - origin_y) / res)
        )
        
        # Determine safety margin (inflation)
        # e.g., robot radius 0.5m -> ~5 cells at 0.1m res
        safety_cells = 4 
        
        # Validate Start and Goal
        # If start is 'unsafe', we might be stuck. Relax safety at start.
        
        open_set = {start_grid}
        came_from = {}
        
        g_score = {start_grid: 0}
        f_score = {start_grid: np.hypot(start_grid[0]-goal_grid[0], start_grid[1]-goal_grid[1])}
        
        # Directions: 8-connected
        neighbors = [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]
        
        while open_set:
            # Get node with lowest f_score
            current = min(open_set, key=lambda n: f_score.get(n, float('inf')))
            
            if np.hypot(current[0]-goal_grid[0], current[1]-goal_grid[1]) < 3.0: # Arrived close enough (within 3 cells)
                # Reconstruct path
                path = []
                while current in came_from:
                    wx = (current[0] + 0.5) * res + origin_x
                    wy = (current[1] + 0.5) * res + origin_y
                    path.append((wx, wy))
                    current = came_from[current]
                path.reverse()
                return path
                
            open_set.remove(current)
            
            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                nx, ny = neighbor
                
                # Check bounds
                if not (0 <= nx < width and 0 <= ny < height):
                    continue
                
                # Check strict collision at center
                if grid_flat[ny * width + nx] > 50:
                    continue
                    
                # Check safety margin (inflation)
                # We skip this check if we are very close to start (to allow getting out of bad spots)
                dist_from_start = np.hypot(nx - start_grid[0], ny - start_grid[1])
                if dist_from_start > 5.0: # Only enforce full safety after moving a bit
                    if not self.is_cell_safe(nx, ny, width, height, grid_flat, safety_cells):
                        continue # Treat as obstacle
                    
                tentative_g_score = g_score[current] + np.hypot(dx, dy)
                
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + np.hypot(neighbor[0]-goal_grid[0], neighbor[1]-goal_grid[1])
                    if neighbor not in open_set:
                        open_set.add(neighbor)
                        
        return None # No path found

    def simplify_path(self, path):
        epsilon = self.path_smoothing_epsilon
        # Simple line simplification (like Ramer-Douglas-Peucker but simpler)
        if not path: return []
        if len(path) < 3: return path
        
        simplified = [path[0]]
        for i in range(1, len(path)-1):
            # If point is far enough from line segment formed by last added point and next point
            # Here we just subsample for simplicity in trajectory generation
            # We want points roughly every 1-2 meters
            last = simplified[-1]
            curr = path[i]
            dist = np.hypot(curr[0]-last[0], curr[1]-last[1])
            if dist > 2.0: # 2 meter spacing
                simplified.append(curr)
        
        simplified.append(path[-1])
        return simplified

    def exploration_loop(self):
        # Wait for map
        while self.map_data is None and self.exploring:
            self.get_logger().info("Waiting for map data...", throttle_duration_sec=2.0)
            time.sleep(1.0)

        # Wait for service
        while not self.nav_client.wait_for_service(timeout_sec=1.0):
            if not self.exploring: break
            self.get_logger().info("Waiting for trajectory planner service...")

        self.get_logger().info("Map received and Service ready. Starting loop.")

        while rclpy.ok() and self.exploring:
            robot_pose = self.get_robot_pose()
            if robot_pose is None:
                self.get_logger().warn("Unknown robot pose")
                time.sleep(1.0)
                continue

            rx = robot_pose.pose.position.x
            ry = robot_pose.pose.position.y
            
            # Find new goal
            frontiers = self.get_frontiers()
            if not frontiers:
                self.get_logger().info("No frontiers found. Waiting for map update...")
                time.sleep(2.0)
                continue

            # Process frontiers
            f_points = np.array(frontiers)
            self.get_logger().info(f"Found {len(f_points)} candidate frontiers.")
            
            # Remove points too close to robot (already visited/explored area)
            dists_to_robot = np.linalg.norm(f_points - np.array([rx, ry]), axis=1)
            
            # Filter by distance
            # We want something not too close, but also not impossibly far
            valid_indices = np.where(dists_to_robot > self.min_frontier_distance_m)[0]
            
            if len(valid_indices) == 0:
                self.get_logger().warn("No distant frontiers. Trying closer ones...")
                # Try closer ones if no distant ones exist, but still avoid immediate vicinity
                valid_indices = np.where(dists_to_robot > 0.5)[0]
            
            if len(valid_indices) == 0:
                self.get_logger().info("No valid frontiers found (all too close).")
                time.sleep(1.0)
                continue

            f_points_filtered = f_points[valid_indices]
            
            # --- FILTER PREVIOUSLY VISITED GOALS ---
            if self.visited_goals:
                self.get_logger().info(f"Memory: {len(self.visited_goals)} visited locations. Filtering...")
                dist_to_visited = []
                for fp in f_points_filtered:
                    min_d = min([np.linalg.norm(fp - np.array(vg)) for vg in self.visited_goals])
                    dist_to_visited.append(min_d)
                
                fresh_indices = np.where(np.array(dist_to_visited) > self.visited_radius)[0]
                if len(fresh_indices) > 0:
                    f_points_filtered = f_points_filtered[fresh_indices]
                else:
                    self.get_logger().info("All frontiers are near visited locations. Reusing closest ones.")
                    # If all are near visited, we don't filter to avoid getting stuck,
                    # but we keep moving
            
            # --- DEBUG: Log robot State ---
            ox = robot_pose.pose.orientation.x
            oy = robot_pose.pose.orientation.y
            oz = robot_pose.pose.orientation.z
            ow = robot_pose.pose.orientation.w
            
            siny_cosp = 2 * (ow * oz + ox * oy)
            cosy_cosp = 1 - 2 * (oy * oy + oz * oz)
            current_yaw = math.atan2(siny_cosp, cosy_cosp)
            
            self.get_logger().info(f"Robot Pose: x={rx:.2f}, y={ry:.2f}, yaw={current_yaw:.2f} rad")

            # --- FILTER BY DIRECTION (FORWARD PRIORITY) ---
            # Progress into cave (positive is forward)
            # if direction is -1, progress is (rx - frontier_x)
            progress_all = (rx - f_points_filtered[:, 0]) if self.exploration_direction_x < 0 else (f_points_filtered[:, 0] - rx)
            
            # Debug candidates
            if len(progress_all) > 0:
                self.get_logger().info(f"Sample Frontiers (x,y -> progress):")
                for i in range(min(5, len(progress_all))):
                    self.get_logger().info(f"  [{f_points_filtered[i][0]:.2f}, {f_points_filtered[i][1]:.2f}] -> {progress_all[i]:.2f}")
            
            # Identify strictly forward candidates (allow 0.2m slack)
            forward_indices = np.where(progress_all > -0.2)[0]
            
            if len(forward_indices) > 0:
                self.get_logger().info(f"Prioritizing {len(forward_indices)} forward frontiers.")
                f_points_filtered = f_points_filtered[forward_indices]
                progress_filtered = progress_all[forward_indices]
            else:
                self.get_logger().warn("No forward frontiers found!")
                
                # Check orientation
                target_yaw = math.pi if self.exploration_direction_x < 0 else 0.0
                yaw_error = abs(math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw)))
                
                if yaw_error > 0.5:
                    self.get_logger().warn(f"Incorrect orientation (Err={yaw_error:.2f}). Forcing movement into cave...")
                    
                    # Create a blind forward movement to orient and uncover frontiers
                    kick_dist = 2.0
                    dx = kick_dist * self.exploration_direction_x
                    target_pt = [rx + dx, ry]
                    
                    # Construct simple path
                    path = [[rx, ry], target_pt]
                    
                    # Execute this path directly
                    self.execute_path(path)
                    time.sleep(2.0)
                    continue

                self.get_logger().warn("Waiting/scanning...")
                time.sleep(2.0)
                continue

            # --- SORT BY DISTANCE + FORWARD BIAS ---
            # We prioritize frontiers that are deep in the cave
            dists = np.linalg.norm(f_points_filtered - np.array([rx, ry]), axis=1)
            
            # Use the filtered progress for scoring
            scores = dists - (self.forward_bias * progress_filtered)
            
            # Add lateral penalty (optional, but helps keep centered if desired)
            # Penalize deviation from y=0 (simulation center) or current y?
            # User said "instead of left right", implying straight is better.
            # strict "forward" usually means minimizing delta-y.
            # abs_y_offset = np.abs(f_points_filtered[:, 1] - ry) # Penalty for moving sideways from current
            # scores += (1.0 * abs_y_offset) 

            sorted_indices = np.argsort(scores)
            
            selected_path = None
            grid = self.map_data
            info = self.map_info
            
            # Try to find a path to the nearest reachable frontier
            # Check top 5 closest frontiers to save computation
            for i in range(min(5, len(sorted_indices))):
                idx = sorted_indices[i]
                candidate = f_points_filtered[idx]
                
                path = self.a_star_search([rx, ry], candidate, grid, info)
                if path:
                    selected_path = self.simplify_path(path)
                    self.get_logger().info(f"Path found to frontier {candidate} with {len(selected_path)} waypoints. Progress: {progress_filtered[idx]:.2f}")
                    break
            
            if selected_path is None:
                self.get_logger().warn("Could not find path to any top frontiers. Retrying...")
                time.sleep(2.0)
                continue
            
            # Execute the path using helper
            if self.execute_path(selected_path):
                self.visited_goals.append(candidate.tolist())
                # Keep list manageable
                if len(self.visited_goals) > 50:
                    self.visited_goals.pop(0)
            else:
                self.get_logger().warn("Path execution failed.")
                time.sleep(2.0)

            time.sleep(0.5)

    def execute_path(self, path):
        # Send path to C++ Node
        req = ExecuteTrajectory.Request()
        pose_array = PoseArray()
        pose_array.header.frame_id = self.map_frame
        pose_array.header.stamp = self.get_clock().now().to_msg()
        
        path_len = len(path)
        for i in range(path_len):
            pt = path[i]
            pose = Pose()
            pose.position.x = pt[0]
            pose.position.y = pt[1]
            pose.position.z = self.exploration_altitude # Fixed altitude
            
            # Force Yaw to face the exploration direction
            if self.exploration_direction_x < 0:
                yaw = math.pi
            else:
                yaw = 0.0

            # Convert Yaw to Quaternion
            pose.orientation.x = 0.0
            pose.orientation.y = 0.0
            pose.orientation.z = math.sin(yaw / 2.0)
            pose.orientation.w = math.cos(yaw / 2.0)
            
            pose_array.poses.append(pose)
        
        req.waypoints = pose_array
        
        self.get_logger().info("Sending goal path to planner...")
        self.trajectory_done_event.clear()
        
        future = self.nav_client.call_async(req)
        
        # Wait for service response (async future)
        while rclpy.ok() and not future.done():
            if not self.exploring: return False
            time.sleep(0.1)
            
        try:
            response = future.result()
        except Exception as e:
            self.get_logger().error(f"Service call failed with exception: {e}")
            return False
        
        if response is None:
            self.get_logger().warn("Service call returned None. Retrying...")
            return False

        if response.success:
            self.get_logger().info("Planner accepted goal. Moving...")
            
            start_wait = time.time()
            while not self.trajectory_done_event.is_set():
                if not self.exploring:
                    self.get_logger().info("Exploration stopped by user.")
                    return False
                if time.time() - start_wait > self.max_path_timeout_s:
                    self.get_logger().warn("Trajectory timed out!")
                    break
                time.sleep(0.1)
            
            self.get_logger().info("Movement finished.")
            # IMPORTANT: Small delay to allow map to update after moving
            time.sleep(2.0) 
            return True
        else:
            self.get_logger().warn(f"Planner rejected goal: {response.message}")
            return False

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
