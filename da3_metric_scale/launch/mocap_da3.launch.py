"""Starts vrpn_mocap, a mocap pose throttle, mocap_to_mavros (OptiTrack -> ArduPilot),
Depth Anything 3 (GerdsenAI wrapper) and da3_metric_scaler.

Live:
  ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<ip>

Rosbag (DA3 + scaler only; start this BEFORE `ros2 bag play <bag> --clock`):
  ros2 launch da3_metric_scale mocap_da3.launch.py start_vrpn:=false start_throttle:=false \
      start_mavros_bridge:=false use_sim_time:=true mocap_pose_topic:=/mavros/dilon/pose_throttled

Toggle parts with start_vrpn / start_throttle / start_mavros_bridge / start_da3 / start_scaler.
The bridge and scaler read the throttled pose (raw pose if start_throttle:=false).
DA3 defaults to the host TensorRT service (da3_use_shared_memory:=true): start that service
first, the same way as when running the DA3 wrapper on its own.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('da3_metric_scale')
    bridge_params = os.path.join(share, 'config', 'mocap_to_mavros.yaml')
    scaler_params = os.path.join(share, 'config', 'da3_metric_scale.yaml')

    use_sim_time = {'use_sim_time': ParameterValue(
        LaunchConfiguration('use_sim_time'), value_type=bool)}
    pose_topic = LaunchConfiguration('mocap_pose_topic')
    throttled_topic = LaunchConfiguration('mocap_pose_throttled_topic')
    image_topic = LaunchConfiguration('image_topic')
    da3_name = LaunchConfiguration('da3_node_name')

    # Pose input for bridge + scaler: throttled topic when the throttle runs, raw otherwise
    pose_input = PythonExpression([
        "'", throttled_topic, "' if '", LaunchConfiguration('start_throttle'),
        "'.lower() == 'true' else '", pose_topic, "'"])
    # DA3 publishes on ~/depth -> /<node name>/depth
    da3_depth_topic = PythonExpression(["'/' + '", da3_name, "' + '/depth'"])

    return LaunchDescription([
        # ---- what to start ----
        DeclareLaunchArgument('start_vrpn', default_value='true'),
        DeclareLaunchArgument('start_throttle', default_value='true'),
        DeclareLaunchArgument('start_mavros_bridge', default_value='true'),
        DeclareLaunchArgument('start_da3', default_value='true'),
        DeclareLaunchArgument('start_scaler', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),

        # ---- topics ----
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('mocap_pose_topic', default_value='/vrpn_mocap/dilon/pose'),
        DeclareLaunchArgument('mocap_pose_throttled_topic',
                              default_value='/vrpn_mocap/dilon/pose_throttled'),
        DeclareLaunchArgument('throttle_rate', default_value='30.0'),
        DeclareLaunchArgument('image_topic', default_value='/image_throttled',
                              description='Image fed to DA3 and paired by the scaler'),

        # ---- DA3 ----
        DeclareLaunchArgument('da3_node_name', default_value='depth_anything_3'),
        DeclareLaunchArgument('da3_model', default_value='depth-anything/DA3-BASE',
                              description='DA3-SMALL / DA3-BASE / DA3-LARGE / ...'),
        DeclareLaunchArgument('da3_device', default_value='cuda'),
        DeclareLaunchArgument('da3_use_shared_memory', default_value='true',
                              description='true = host TensorRT service (start it first); '
                                          'false = in-container PyTorch DA3 package'),
        DeclareLaunchArgument('da3_publish_colored', default_value='true'),
        DeclareLaunchArgument('da3_log_inference_time', default_value='true'),

        # ---- params files ----
        DeclareLaunchArgument('bridge_params', default_value=bridge_params),
        DeclareLaunchArgument('scaler_params', default_value=scaler_params),

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
        # Depth Anything 3 (GerdsenAI wrapper). Copies the input image header onto
        # the depth, which is what the scaler pairs on.
        Node(
            package='depth_anything_3_ros2',
            executable='depth_anything_3_node',
            name=da3_name,
            output='screen',
            remappings=[
                ('~/image_raw', image_topic),
                ('~/camera_info', '/camera_info'),  # optional; not needed by the scaler
            ],
            parameters=[{
                'model_name': LaunchConfiguration('da3_model'),
                'device': LaunchConfiguration('da3_device'),
                # MUST stay false: per-frame min-max normalisation destroys the scale
                'normalize_depth': False,
                # Output at camera resolution (640x480) instead of 518x518
                'keep_image_size': True,
                'publish_colored': ParameterValue(
                    LaunchConfiguration('da3_publish_colored'), value_type=bool),
                'publish_confidence': False,
                'queue_size': 1,
                'log_inference_time': ParameterValue(
                    LaunchConfiguration('da3_log_inference_time'), value_type=bool),
                'use_shared_memory': ParameterValue(
                    LaunchConfiguration('da3_use_shared_memory'), value_type=bool),
            }, use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_da3')),
        ),
        Node(
            package='da3_metric_scale',
            executable='da3_metric_scaler',
            name='da3_metric_scaler',
            output='screen',
            parameters=[LaunchConfiguration('scaler_params'),
                        {'mocap_pose_topic': pose_input,
                         'image_topic': image_topic,
                         'depth_topic': da3_depth_topic},
                        use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_scaler')),
        ),
    ])
