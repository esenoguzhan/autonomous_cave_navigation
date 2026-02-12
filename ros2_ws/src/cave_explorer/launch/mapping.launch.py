from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = True

    return LaunchDescription([
        Node(
            package='cave_explorer',
            executable='depth_to_pointcloud',
            name='depth_to_pointcloud',
            output='screen',
            parameters=[
                {'depth_topic': '/realsense/depth/image'},
                {'info_topic': '/realsense/depth/camera_info'},
                {'cloud_topic': '/depth_cloud'},
                {'use_sim_time': use_sim_time}
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_camera_alias',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '-1.5708', '--pitch', '0', '--roll', '-1.5708',
                '--frame-id', 'Quadrotor/DepthCamera',
                '--child-frame-id', 'Quadrotor/Sensors/DepthCamera'
            ],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen'
        ),
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                {'resolution': 0.2},
                {'frame_id': 'world'},
                {'base_frame_id': 'true_body'},
                {'sensor_model.max_range': 15.0},
                {'use_sim_time': use_sim_time},
                {'pointcloud_min_z': 12.0},  # Ignore points below flight level
                {'pointcloud_max_z': 18.0},  # Ignore points above flight level
                {'occupancy_min_z': 13.0},   # Filter for 2D map
                {'occupancy_max_z': 17.0},   # Filter for 2D map
                {'filter_speckles': True},
                {'filter_ground': False},
                {'latch': True}
            ],
            remappings=[
                ('cloud_in', '/depth_cloud')
            ]
        )
    ])
