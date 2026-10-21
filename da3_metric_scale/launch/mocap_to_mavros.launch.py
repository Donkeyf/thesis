"""vrpn_mocap -> topic_tools throttle (30 Hz) -> mocap_to_mavros (ArduPilot only)."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'mocap_to_mavros.yaml')

    pose_topic = LaunchConfiguration('mocap_pose_topic')
    throttled_topic = LaunchConfiguration('mocap_pose_throttled_topic')
    pose_input = PythonExpression([
        "'", throttled_topic, "' if '", LaunchConfiguration('start_throttle'),
        "'.lower() == 'true' else '", pose_topic, "'"])

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        DeclareLaunchArgument('start_throttle', default_value='true'),
        DeclareLaunchArgument('mocap_pose_topic', default_value='/vrpn_mocap/dilon/pose'),
        DeclareLaunchArgument('mocap_pose_throttled_topic',
                              default_value='/vrpn_mocap/dilon/pose_throttled'),
        DeclareLaunchArgument('throttle_rate', default_value='30.0'),
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
            package='topic_tools',
            executable='throttle',
            name='mocap_pose_throttle',
            output='screen',
            arguments=['messages', pose_topic, LaunchConfiguration('throttle_rate'),
                       throttled_topic],
            condition=IfCondition(LaunchConfiguration('start_throttle')),
        ),
        Node(
            package='da3_metric_scale',
            executable='mocap_to_mavros',
            name='mocap_to_mavros',
            output='screen',
            parameters=[LaunchConfiguration('params'), {'mocap_pose_topic': pose_input}],
        ),
    ])
