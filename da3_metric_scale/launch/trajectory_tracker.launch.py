"""Spline/trajectory tracker -> ArduCopter GUIDED via MAVROS setpoint_raw/local.

  ros2 launch da3_metric_scale trajectory_tracker.launch.py [mavros_prefix:=/mavros]

Planner publishes to /trajectory_tracker/trajectory (MultiDOFJointTrajectory) or
/trajectory_tracker/path (nav_msgs/Path). Take off, switch to GUIDED, then:
  ros2 service call /trajectory_tracker/enable std_srvs/srv/Trigger
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'trajectory_tracker.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('mavros_prefix', default_value='/mavros/mavros'),
        DeclareLaunchArgument('trajectory_topic', default_value='/trajectory_tracker/trajectory'),
        DeclareLaunchArgument('path_topic', default_value='/trajectory_tracker/path'),
        Node(
            package='da3_metric_scale',
            executable='trajectory_tracker',
            name='trajectory_tracker',
            output='screen',
            parameters=[LaunchConfiguration('params'),
                        {'mavros_prefix': LaunchConfiguration('mavros_prefix')}],
            remappings=[('~/trajectory', LaunchConfiguration('trajectory_topic')),
                        ('~/path', LaunchConfiguration('path_topic'))],
        ),
    ])
