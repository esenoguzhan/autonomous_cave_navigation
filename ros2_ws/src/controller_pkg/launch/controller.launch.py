#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    """
    Launch file for the geometric controller node.
    Loads all parameters from controller_params.yaml.
    """
    
    # Get path to config file
    pkg_share = get_package_share_directory('controller_pkg')
    default_config = os.path.join(pkg_share, 'config', 'controller_params.yaml')

    # Declare launch argument for config file path
    config_arg = DeclareLaunchArgument(
        'config_file',
        default_value=default_config,
        description='Absolute path to controller configuration YAML file'
    )

    # Controller node
    controller_node = Node(
        package='controller_pkg',
        executable='controller_node',
        name='controller_node',
        output='screen',
        parameters=[LaunchConfiguration('config_file')],
        emulate_tty=True,
        # Ensure node fails if parameters are missing
        arguments=['--ros-args', '--log-level', 'INFO']
    )

    return LaunchDescription([
        config_arg,
        controller_node
    ])
