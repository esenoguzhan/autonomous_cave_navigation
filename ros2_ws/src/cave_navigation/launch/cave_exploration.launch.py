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
                'neighborcount_threshold': 10,
                'bandwidth': 1.0,
                'k_distance': 0.2,
                'k_neighborcount': 1.0,
                'k_yaw': 0.5,
                'distance_limit': 80.0,
                'publish_goal_frequency': 1.0,
                'occ_neighbor_threshold': 5,
                'altitude_tolerance': 10.0,
                'safety_distance': 20.0,
                'min_passage_width': 3.0
            }]
        ),
        Node(
            package='rrt_planner',
            executable='rrt_planner_node',
            name='rrt_planner',
            output='screen',
            respawn=True,
            parameters=[{
                'step_size_factor': 5.0,
                'bias': 0.05,
                'timeout': 1.0,
                'rrt_frequency': 0.1
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

