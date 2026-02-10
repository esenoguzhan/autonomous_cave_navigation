from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='cave_explorer',
            executable='cave_explorer',
            name='cave_explorer',
            output='screen',
            parameters=[
                {'map_topic': 'projected_map'},
                {'base_frame': 'true_body'}
            ]
        )
    ])
