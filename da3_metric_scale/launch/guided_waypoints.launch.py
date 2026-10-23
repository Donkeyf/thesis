"""Waypoint sender for ArduCopter GUIDED via MAVROS (MAVROS must already be running).

  ros2 launch da3_metric_scale guided_waypoints.launch.py
  ros2 launch da3_metric_scale guided_waypoints.launch.py mavros_prefix:=/mavros

Then: arm, switch to GUIDED on the RC, and
  ros2 service call /guided_waypoints/start std_srvs/srv/Trigger
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'guided_waypoints.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('mavros_prefix', default_value='/mavros/mavros'),
        Node(
            package='da3_metric_scale',
            executable='guided_waypoints',
            name='guided_waypoints',
            output='screen',
            parameters=[LaunchConfiguration('params'),
                        {'mavros_prefix': LaunchConfiguration('mavros_prefix')}],
        ),
    ])
