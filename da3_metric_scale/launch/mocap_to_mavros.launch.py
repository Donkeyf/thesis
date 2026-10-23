"""OptiTrack -> ArduPilot: MAVROS + vrpn_mocap -> topic_tools throttle (30 Hz) -> mocap_to_mavros.

  ros2 launch da3_metric_scale mocap_to_mavros.launch.py motive_ip:=<ip>

  start_mavros:=false     if MAVROS already runs elsewhere (e.g. thesis.launch.py) -
                          two MAVROS instances on one serial port will fight.
  fcu_url:=/dev/ttyACM0:921600   FC connection (Pixracer USB)
  gcs_url:=udp-b://@14550        optional: forward MAVLink to Mission Planner / QGC
  mavros_prefix:=/mavros         if your MAVROS topics are /mavros/... not /mavros/mavros/...
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _mavros_param_files():
    """apm_pluginlists.yaml / apm_config.yaml from the mavros package, as apm.launch uses."""
    try:
        share = get_package_share_directory('mavros')
    except Exception:
        return []
    files = [os.path.join(share, 'launch', f)
             for f in ('apm_pluginlists.yaml', 'apm_config.yaml')]
    return [f for f in files if os.path.isfile(f)]


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('da3_metric_scale'), 'config', 'mocap_to_mavros.yaml')

    pose_topic = LaunchConfiguration('mocap_pose_topic')
    throttled_topic = LaunchConfiguration('mocap_pose_throttled_topic')
    pose_input = PythonExpression([
        "'", throttled_topic, "' if '", LaunchConfiguration('start_throttle'),
        "'.lower() == 'true' else '", pose_topic, "'"])
    prefix = LaunchConfiguration('mavros_prefix')

    return LaunchDescription([
        DeclareLaunchArgument('params', default_value=default_params),
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        DeclareLaunchArgument('start_throttle', default_value='true'),
        DeclareLaunchArgument('start_mavros', default_value='true'),
        DeclareLaunchArgument('mocap_pose_topic', default_value='/vrpn_mocap/dilon/pose'),
        DeclareLaunchArgument('mocap_pose_throttled_topic',
                              default_value='/vrpn_mocap/dilon/pose_throttled'),
        DeclareLaunchArgument('throttle_rate', default_value='30.0'),

        # ---- MAVROS ----
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:921600'),
        DeclareLaunchArgument('gcs_url', default_value=''),
        DeclareLaunchArgument('tgt_system', default_value='1'),
        DeclareLaunchArgument('tgt_component', default_value='1'),
        DeclareLaunchArgument('mavros_namespace', default_value='mavros'),
        DeclareLaunchArgument('mavros_prefix', default_value='/mavros/mavros',
                              description='Prefix of MAVROS topics as seen in ros2 topic list'),

        Node(
            package='mavros',
            executable='mavros_node',
            namespace=LaunchConfiguration('mavros_namespace'),
            output='screen',
            parameters=_mavros_param_files() + [{
                'fcu_url': LaunchConfiguration('fcu_url'),
                'gcs_url': LaunchConfiguration('gcs_url'),
                'tgt_system': ParameterValue(LaunchConfiguration('tgt_system'), value_type=int),
                'tgt_component': ParameterValue(LaunchConfiguration('tgt_component'),
                                                value_type=int),
                'fcu_protocol': 'v2.0',
            }],
            condition=IfCondition(LaunchConfiguration('start_mavros')),
        ),
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
            parameters=[LaunchConfiguration('params'), {
                'mocap_pose_topic': pose_input,
                'vision_pose_topic': [prefix, '/vision_pose/pose'],
                'mavros_state_topic': [prefix, '/state'],
                'origin_topic': [prefix, '/global_position/set_gp_origin'],
            }],
        ),
    ])
