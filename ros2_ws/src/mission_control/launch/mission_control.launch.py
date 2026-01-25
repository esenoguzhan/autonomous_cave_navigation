#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Include simulation launch file
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("simulation"), "launch", "simulation.launch.py"])
        )
    )
    
    # Load mission control parameters from config file
    mission_control_config = os.path.join(
        get_package_share_directory('mission_control'),
        'config',
        'mission_control_params.yaml'
    )
    
    # Mission control node
    mission_control_node = Node(
        package="mission_control",
        executable="mission_control_node",
        name="mission_control_node",
        output="screen",
        parameters=[mission_control_config],
    )

    return LaunchDescription([
        simulation_launch,
        mission_control_node,
    ])
