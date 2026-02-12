from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    use_sim_time = True 

    static_tf_alias = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='depth_camera_alias',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--yaw', '0', '--pitch', '0', '--roll', '0',
            '--frame-id', 'Quadrotor/DepthCamera',
            '--child-frame-id', 'Quadrotor/Sensors/DepthCamera'
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}]
    )

    pointcloud_node = Node(
        package='cave_explorer',
        executable='depth_to_pointcloud',
        name='depth_to_pointcloud',
        output='screen',
        respawn=True,
        parameters=[{
            'use_sim_time': use_sim_time,
            'depth_topic': '/realsense/depth/image',
            'info_topic': '/realsense/depth/camera_info',
            'cloud_topic': '/realsense/depth/points'
        }]
    )

    octomap_node = Node(
        package='octomap_server',
        executable='octomap_server_node',
        name='octomap_server',
        output='screen',
        respawn=True,
        parameters=[
            {'resolution': 0.2},
            {'frame_id': 'world'},
            {'sensor_model.max_range': 20.0},
            {'publish_free_space': True},
            {'use_sim_time': use_sim_time}
        ],
        remappings=[
            ('cloud_in', '/realsense/depth/points')
        ]
    )

    return LaunchDescription([
        static_tf_alias,
        pointcloud_node,
        octomap_node
    ])