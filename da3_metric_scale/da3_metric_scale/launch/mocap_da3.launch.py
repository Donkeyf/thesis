"""Starts vrpn_mocap, mocap_to_mavros (OptiTrack -> ArduPilot) and da3_metric_scaler.

ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<ip>
Toggle parts with start_vrpn / start_mavros_bridge / start_scaler:=false.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('da3_metric_scale')
    bridge_params = os.path.join(share, 'config', 'mocap_to_mavros.yaml')
    scaler_params = os.path.join(share, 'config', 'da3_metric_scale.yaml')

    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}

    return LaunchDescription([
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        DeclareLaunchArgument('start_mavros_bridge', default_value='true'),
        DeclareLaunchArgument('start_scaler', default_value='true'),
        DeclareLaunchArgument('bridge_params', default_value=bridge_params),
        DeclareLaunchArgument('scaler_params', default_value=scaler_params),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        Node(
            package='vrpn_mocap',
            executable='client_node',
            name='vrpn_mocap',
            namespace='vrpn_mocap',
            output='screen',
            parameters=[{'server': LaunchConfiguration('motive_ip'), 'port': 3883},
                        use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_vrpn')),
        ),
        Node(
            package='da3_metric_scale',
            executable='mocap_to_mavros',
            name='mocap_to_mavros',
            output='screen',
            parameters=[LaunchConfiguration('bridge_params'), use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_mavros_bridge')),
        ),
        Node(
            package='da3_metric_scale',
            executable='da3_metric_scaler',
            name='da3_metric_scaler',
            output='screen',
            parameters=[LaunchConfiguration('scaler_params'), use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_scaler')),
        ),
    ])
