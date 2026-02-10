from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cave_explorer',
            executable='depth_to_pointcloud',
            name='depth_to_pointcloud',
            output='screen',
            parameters=[
                {'depth_topic': '/realsense/depth/image'},
                {'info_topic': '/realsense/depth/camera_info'},
                {'cloud_topic': '/depth_cloud'}
            ]
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='depth_camera_alias',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--yaw', '0', '--pitch', '0', '--roll', '0',
                '--frame-id', 'Quadrotor/DepthCamera',
                '--child-frame-id', 'Quadrotor/Sensors/DepthCamera'
            ],
            output='screen'
        ),
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                {'resolution': 0.1},
                {'frame_id': 'world'},
                {'sensor_model.max_range': 10.0}
            ],
            remappings=[
                ('cloud_in', '/depth_cloud')
            ]
        )
    ])
