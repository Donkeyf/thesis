"""Starts vrpn_mocap, a mocap pose throttle, mocap_to_mavros (OptiTrack -> ArduPilot)
and da3_metric_scaler.

ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<ip>
Toggle parts with start_vrpn / start_throttle / start_mavros_bridge / start_scaler:=false.

Both the ArduPilot bridge and the scaler read the throttled pose (raw pose if
start_throttle:=false).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('da3_metric_scale')
    bridge_params = os.path.join(share, 'config', 'mocap_to_mavros.yaml')
    scaler_params = os.path.join(share, 'config', 'da3_metric_scale.yaml')

    use_sim_time = {'use_sim_time': LaunchConfiguration('use_sim_time')}
    pose_topic = LaunchConfiguration('mocap_pose_topic')
    throttled_topic = LaunchConfiguration('mocap_pose_throttled_topic')

    # Pose input for both nodes: throttled topic when the throttle runs, raw otherwise
    pose_input = PythonExpression([
        "'", throttled_topic, "' if '", LaunchConfiguration('start_throttle'),
        "'.lower() == 'true' else '", pose_topic, "'"])

    return LaunchDescription([
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        DeclareLaunchArgument('start_throttle', default_value='true'),
        DeclareLaunchArgument('start_mavros_bridge', default_value='true'),
        DeclareLaunchArgument('start_scaler', default_value='true'),
        DeclareLaunchArgument('mocap_pose_topic', default_value='/vrpn_mocap/dilon/pose'),
        DeclareLaunchArgument('mocap_pose_throttled_topic',
                              default_value='/vrpn_mocap/dilon/pose_throttled'),
        DeclareLaunchArgument('throttle_rate', default_value='30.0'),
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
        # ros2 run topic_tools throttle messages <in> <rate> <out>
        Node(
            package='topic_tools',
            executable='throttle',
            name='mocap_pose_throttle',
            output='screen',
            arguments=['messages', pose_topic, LaunchConfiguration('throttle_rate'),
                       throttled_topic],
            parameters=[use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_throttle')),
        ),
        Node(
            package='da3_metric_scale',
            executable='mocap_to_mavros',
            name='mocap_to_mavros',
            output='screen',
            parameters=[LaunchConfiguration('bridge_params'),
                        {'mocap_pose_topic': pose_input},
                        use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_mavros_bridge')),
        ),
        Node(
            package='da3_metric_scale',
            executable='da3_metric_scaler',
            name='da3_metric_scaler',
            output='screen',
            parameters=[LaunchConfiguration('scaler_params'),
                        {'mocap_pose_topic': pose_input},
                        use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_scaler')),
        ),
    ])
