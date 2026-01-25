#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Load trajectory planner parameters from config file
    trajectory_planner_config = os.path.join(
        get_package_share_directory('trajectory_planner'),
        'config',
        'trajectory_planner_params.yaml'
    )

    trajectory_planner_node = Node(
        package="trajectory_planner",
        executable="trajectory_planner_node",
        name="trajectory_planner_node",
        output="screen",
        parameters=[trajectory_planner_config],
    )

    return LaunchDescription([
        trajectory_planner_node,
    ])
