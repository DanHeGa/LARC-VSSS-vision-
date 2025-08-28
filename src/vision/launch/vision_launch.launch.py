from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    pkg_name = "vision"
    return LaunchDescription([
        Node(
            package=pkg_name,
            executable='camera_input',
            name='camera_input',
            output='screen',
            parameters=[{'Video_ID' : 0}]
        ),
        Node(
            package=pkg_name,
            executable='image_warp',
            name='image_warp',
            output='screen'
        ),
        Node(
            package=pkg_name,
            executable='model_use',
            name='model_use',
            output='screen'
        ),
        Node(
            package=pkg_name,
            executable='ball_detect',
            name='ball_detect',
            output='screen'
        ),
    ])