from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ldm_unit',
            namespace='ldm_unit',
            executable='ldm_node',
            name='ldm_node'
        ),
        Node(
            package='ldm_unit',
            namespace='ldm_unit',
            executable='audio_decision_node',
            name='audio_decision_node'
        ),
        Node(
            package='ldm_unit',
            namespace='ldm_unit',
            executable='video_decision_node',
            name='video_decision_node'
        ),
        Node(
            package='ldm_unit',
            namespace='ldm_unit',
            executable='camera_percept_node',
            name='camera_percept_node'
        ),
        # Node(
        #     package='ldm_unit',
        #     namespace='ldm_unit',
        #     executable='mic_percept_node',
        #     name='mic_percept_node'
        # ),
        # launch.actions.ExecuteProcess(
        #     cmd=['ros2', 'bag', 'record', '-a', '-o', 'bag'],
        #     output='screen'
        # )

    ])
