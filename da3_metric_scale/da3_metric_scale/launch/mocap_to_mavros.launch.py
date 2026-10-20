import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'mocap_to_mavros.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        Node(
            package='vrpn_mocap',
            executable='client_node',
            name='vrpn_mocap',
            namespace='vrpn_mocap',
            output='screen',
            parameters=[{'server': LaunchConfiguration('motive_ip'), 'port': 3883}],
            condition=IfCondition(LaunchConfiguration('start_vrpn')),
        ),
        Node(
            package='da3_metric_scale',
            executable='mocap_to_mavros',
            name='mocap_to_mavros',
            output='screen',
            parameters=[LaunchConfiguration('params')],
        ),
    ])
