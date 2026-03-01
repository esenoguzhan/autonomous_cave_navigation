#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Cave Exploration v2: frontier_detector + sampling_planner_node
    Replaces the original frontier_detector + rrt_planner + trajectory_generator pipeline.
    """
    return LaunchDescription([
        # 3D Frontier Detector
        Node(
            package='frontier_detector',
            executable='frontier_detector_node',
            name='frontier_detector',
            output='screen',
            respawn=True,
            parameters=[{
                'neighborcount_threshold': 50,
                'bandwidth': 17.0,
                'k_distance': -0.5,
                'k_neighborcount': 0.1,
                'k_yaw': 55.0,
                'distance_limit': 350.0,
                'publish_goal_frequency': 2.0,
                'occ_neighbor_threshold': 1,
                'max_frontiers': 500,
                'min_frontier_distance': 10.0,
                'max_frontier_distance': 60.0,
            }]
        ),

        # Sampling-Based Trajectory Planner
        Node(
            package='sampling_based_traj_gen',
            executable='sampling_planner_node',
            name='sampling_planner_node',
            output='screen',
            respawn=True,
            parameters=[{
                'num_samples': 10,
                'min_duration_factor': 1.0,
                'max_duration_factor': 1.8,
                'lateral_spread': 3.0,
                'max_recursion_depth': 4,
                'safety_radius': 2.0,
                'navigate_to_cave_speed': 10.0,
                'cave_exploration_speed': 4.0,
                'collision_check_dt': 0.2,
                'planning_frequency': 2.0,
                'lookahead_distance': 15.0,
                'terminal_speed_fraction': 0.7,
                'goal_debounce_distance': 5.0,
                'min_goal_distance': 5.0,
            }]
        ),
    ])
