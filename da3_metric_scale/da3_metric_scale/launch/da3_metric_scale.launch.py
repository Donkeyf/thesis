import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'da3_metric_scale.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='da3_metric_scale',
            executable='da3_metric_scaler',
            name='da3_metric_scaler',
            output='screen',
            parameters=[LaunchConfiguration('params'),
                        {'use_sim_time': LaunchConfiguration('use_sim_time')}],
        ),
    ])
