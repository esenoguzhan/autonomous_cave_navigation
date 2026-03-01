from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='frontier_detector',
            executable='frontier_detector_node',
            name='frontier_detector',
            output='screen',
            respawn=True,
            parameters=[{
                'neighborcount_threshold': 100,
                'bandwidth': 17.0,
                'k_distance': 1.0,
                'k_neighborcount': 0.1,
                'k_yaw': 55.0,
                'distance_limit': 600.0,
                'publish_goal_frequency': 2.0,
                'occ_neighbor_threshold': 1,
                'pre_filter_distance': 50.0,
                'max_frontiers': 3000,
            }]
        ),
        Node(
            package='rrt_planner',
            executable='rrt_planner_node',
            name='rrt_planner',
            output='screen',
            respawn=True,
            parameters=[{
                'step_size_factor': 2.0,
                'bias': 0.15,
                'timeout': 1.0,
                'rrt_frequency': 1.0
            }]
        ),
        Node(
            package='trajectory_generator',
            executable='trajectory_generator_node',
            name='trajectory_generator',
            output='screen',
            respawn=True
        )
    ])

