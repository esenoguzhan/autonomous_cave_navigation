#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    """
    Cave Exploration v2: frontier_detector + sampling_planner_node
    Replaces the original frontier_detector + rrt_planner + trajectory_generator pipeline.
    """
    return LaunchDescription([
        # 3D Frontier Detector (same params as cave_exploration.launch.py)
        Node(
            package='frontier_detector',
            executable='frontier_detector_node',
            name='frontier_detector',
            output='screen',
            respawn=True,
            parameters=[{
                'neighborcount_threshold': 10,
                'bandwidth': 1.0,
                'k_distance': 1.0,
                'k_neighborcount': 1.0,
                'k_yaw': 1.0,
                'distance_limit': 10.0,
                'publish_goal_frequency': 1.0,
                'occ_neighbor_threshold': 5,
                'altitude_tolerance': 3.0,
                'safety_distance': 1.5,
                'min_passage_width': 3.0
            }]
        ),

        # Sampling-Based Trajectory Planner (replaces rrt_planner + trajectory_generator)
        Node(
            package='sampling_based_traj_gen',
            executable='sampling_planner_node',
            name='sampling_planner_node',
            output='screen',
            respawn=True,
            parameters=[{
                'num_samples': 200,
                'min_duration_factor': 2.0,
                'max_duration_factor': 2.5,
                'lateral_spread': 3.0,
                'max_recursion_depth': 4,
                'safety_radius': 0.5,
                'trajectory_speed': 3.0,
                'collision_check_dt': 0.2,
                'planning_frequency': 1.0
            }]
        ),
    ])
