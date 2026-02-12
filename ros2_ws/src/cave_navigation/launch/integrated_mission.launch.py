#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution, LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Simulation
    simulation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("simulation"), "launch", "simulation.launch.py"])
        )
    )
    
    # Perception (3D Octomap)
    # Using the one we fixed to be 3D
    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("octomap_server"), "launch", "perception_mapping.launch.py"])
        )
    )

    # Cave Navigation Stack (Frontier + RRT + Trajectory)
    cave_navigation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("cave_navigation"), "launch", "cave_exploration.launch.py"])
        )
    )

    # Mission Control Node
    mission_control_config = os.path.join(
        get_package_share_directory('mission_control'),
        'config',
        'mission_control_params.yaml'
    )
    
    mission_control_node = Node(
        package="mission_control",
        executable="mission_control_node",
        name="mission_control_node",
        output="screen",
        parameters=[
            mission_control_config,
            {'use_sim_time': True}
        ],
    )

    return LaunchDescription([
        simulation_launch,
        perception_launch,
        mission_control_node,
        cave_navigation_launch
    ])
