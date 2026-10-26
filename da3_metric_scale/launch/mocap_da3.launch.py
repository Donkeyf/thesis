"""Starts vrpn_mocap, a mocap pose throttle, mocap_to_mavros (OptiTrack -> ArduPilot),
Depth Anything 3 (GerdsenAI wrapper) and da3_metric_scaler.

Live (also starts the USB camera: usb_cam 640x480 @ 100 fps on /image_raw, throttled to
image_rate = 10 Hz on /image_da3 for DA3 and everything downstream):
  ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<ip>
  camera args: start_camera (true), video_device (/dev/video0), camera_width (640),
               camera_height (480), camera_fps (100.0), pixel_format (mjpeg2rgb)

Rosbag (add start_camera:=false; start this BEFORE `ros2 bag play <bag> --clock`):
  (bag recorded with /image_throttled: add raw_image_topic:=/image_throttled)
  ros2 launch da3_metric_scale mocap_da3.launch.py start_vrpn:=false start_throttle:=false \
      start_mavros_bridge:=false use_sim_time:=true mocap_pose_topic:=/mavros/dilon/pose_throttled

Toggle parts with start_vrpn / start_throttle / start_mavros_bridge / start_da3 / start_scaler /
start_nvblox / start_esdf_publisher.

Rates: image_throttle (image_rate, 10 Hz) -> DA3 -> scaler -> nvblox -> esdf_publisher, all 10 Hz.

Mapping: da3_metric_scaler -> /da3_metric/depth (+ TF mocap->camera_optical) -> nvblox_node
(3D ESDF in global_frame, default 'map' = EKF local frame, linked to 'mocap' by a static TF)
-> esdf_publisher -> /esdf_publisher/esdf_grid, /esdf_publisher/esdf_cloud.
The bridge and scaler read the throttled pose (raw pose if start_throttle:=false).
DA3 runs on TensorRT: this launch also starts scripts/trt_inference_service_shm.py from the
wrapper repo (found automatically, or set da3_service_script:=<path>). start_da3_service:=false
if you already run the service yourself.
"""
import glob
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _find_trt_service_script():
    # GerdsenAI wrapper source: <ws>/src/**/scripts/trt_inference_service_shm.py
    for pattern in ('/workspaces/isaac_ros-dev/src/*/scripts/trt_inference_service_shm.py',
                    '/workspaces/isaac_ros-dev/src/*/*/scripts/trt_inference_service_shm.py'):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return ''


def generate_launch_description():
    share = get_package_share_directory('da3_metric_scale')
    bridge_params = os.path.join(share, 'config', 'mocap_to_mavros.yaml')
    scaler_params = os.path.join(share, 'config', 'da3_metric_scale.yaml')
    nvblox_params = os.path.join(share, 'config', 'nvblox_da3.yaml')
    esdf_params = os.path.join(share, 'config', 'esdf_publisher.yaml')
    global_frame = LaunchConfiguration('global_frame')

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

        # ---- USB camera (usb_cam) ----
        DeclareLaunchArgument('start_camera', default_value='true',
                              description='start usb_cam (false for rosbags)'),
        DeclareLaunchArgument('video_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_width', default_value='640',
                              description='must match the Kalibr calibration resolution'),
        DeclareLaunchArgument('camera_height', default_value='480'),
        DeclareLaunchArgument('camera_fps', default_value='100.0'),
        DeclareLaunchArgument('pixel_format', default_value='mjpeg2rgb',
                              description='mjpeg2rgb | yuyv2rgb | ...'),

        # ---- topics ----
        DeclareLaunchArgument('motive_ip', default_value='192.168.1.100'),
        DeclareLaunchArgument('mocap_pose_topic', default_value='/vrpn_mocap/dilon/pose'),
        DeclareLaunchArgument('mocap_pose_throttled_topic',
                              default_value='/vrpn_mocap/dilon/pose_throttled'),
        DeclareLaunchArgument('throttle_rate', default_value='30.0'),
        DeclareLaunchArgument('start_image_throttle', default_value='true',
                              description='throttle raw_image_topic -> image_topic at image_rate'),
        DeclareLaunchArgument('raw_image_topic', default_value='/image_raw',
                              description='camera stream (100 fps)'),
        DeclareLaunchArgument('image_rate', default_value='10.0',
                              description='Hz into DA3; sets the rate of the whole chain'),
        DeclareLaunchArgument('image_topic', default_value='/image_da3',
                              description='Image fed to DA3 and paired by the scaler'),

        # ---- DA3 ----
        DeclareLaunchArgument('da3_node_name', default_value='depth_anything_3'),
        DeclareLaunchArgument('da3_model', default_value='depth-anything/DA3-BASE',
                              description='DA3-SMALL / DA3-BASE / DA3-LARGE / ...'),
        DeclareLaunchArgument('da3_device', default_value='cuda'),
        DeclareLaunchArgument('da3_use_shared_memory', default_value='true',
                              description='true = host TensorRT service (start it first); '
                                          'false = in-container PyTorch DA3 package'),
        DeclareLaunchArgument('start_da3_service', default_value='true',
                              description='Start the TensorRT inference service '
                                          '(scripts/trt_inference_service_shm.py)'),
        DeclareLaunchArgument('da3_service_script', default_value=_find_trt_service_script(),
                              description='Path to trt_inference_service_shm.py'),
        DeclareLaunchArgument('da3_engine', default_value='models/tensorrt/da3-small-fp16.engine',
                              description='TRT engine; relative paths are from the wrapper repo'),
        DeclareLaunchArgument('da3_service_python', default_value='python3'),
        DeclareLaunchArgument('da3_service_startup_delay', default_value='3.0',
                              description='s to wait after starting the service before DA3 node'),
        DeclareLaunchArgument('da3_publish_colored', default_value='true'),
        DeclareLaunchArgument('da3_log_inference_time', default_value='true'),

        # ---- params files ----
        DeclareLaunchArgument('bridge_params', default_value=bridge_params),
        DeclareLaunchArgument('scaler_params', default_value=scaler_params),

        # ---- nvblox / ESDF ----
        DeclareLaunchArgument('start_nvblox', default_value='true'),
        DeclareLaunchArgument('start_esdf_publisher', default_value='true'),
        DeclareLaunchArgument('nvblox_params', default_value=nvblox_params),
        DeclareLaunchArgument('esdf_params', default_value=esdf_params),
        DeclareLaunchArgument('global_frame', default_value='map',
                              description='nvblox/ESDF frame; map = EKF/MAVROS local frame'),
        DeclareLaunchArgument('mocap_frame', default_value='mocap',
                              description='scaler world_frame (mocap world)'),
        DeclareLaunchArgument('map_mocap_z_offset', default_value='0.0',
                              description='EKF z - mocap z (m), for the map->mocap static TF'),
        DeclareLaunchArgument('metric_depth_topic', default_value='/da3_metric/depth'),
        DeclareLaunchArgument('metric_info_topic', default_value='/da3_metric/camera_info'),
        DeclareLaunchArgument('nvblox_log_level', default_value='warn',
                              description='nvblox logs every ESDF service call at info'),

        # ros2 run usb_cam usb_cam_node_exe -> /image_raw (+ /camera_info) at camera_fps
        Node(
            package='usb_cam',
            executable='usb_cam_node_exe',
            name='usb_cam',
            output='screen',
            parameters=[{
                'video_device': LaunchConfiguration('video_device'),
                'image_width': ParameterValue(LaunchConfiguration('camera_width'), value_type=int),
                'image_height': ParameterValue(LaunchConfiguration('camera_height'), value_type=int),
                'framerate': ParameterValue(LaunchConfiguration('camera_fps'), value_type=float),
                'pixel_format': LaunchConfiguration('pixel_format'),
            }, use_sim_time],
            remappings=[('image_raw', LaunchConfiguration('raw_image_topic'))],
            condition=IfCondition(LaunchConfiguration('start_camera')),
        ),
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
        # Camera 100 fps -> image_rate (10 Hz) for DA3; everything downstream follows
        Node(
            package='topic_tools',
            executable='throttle',
            name='image_throttle',
            output='screen',
            arguments=['messages', LaunchConfiguration('raw_image_topic'),
                       LaunchConfiguration('image_rate'), image_topic],
            parameters=[use_sim_time],
            condition=IfCondition(LaunchConfiguration('start_image_throttle')),
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
        # TensorRT inference service (same as the wrapper's run.sh step 5).
        # Clears stale status/IPC files so the node can't see an old "ready".
        ExecuteProcess(
            cmd=['bash', '-c', [
                'mkdir -p /dev/shm/da3 /tmp/da3_shared && '
                'rm -f /dev/shm/da3/* /tmp/da3_shared/* ; '
                'cd "$(dirname "', LaunchConfiguration('da3_service_script'), '")/.." && '
                'exec ', LaunchConfiguration('da3_service_python'), ' "',
                LaunchConfiguration('da3_service_script'), '" --engine "',
                LaunchConfiguration('da3_engine'), '"']],
            name='da3_trt_service',
            output='screen',
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_da3_service'), "'.lower() == 'true' and '",
                LaunchConfiguration('start_da3'), "'.lower() == 'true' and '",
                LaunchConfiguration('da3_use_shared_memory'), "'.lower() == 'true'"])),
        ),
        # Depth Anything 3 (GerdsenAI wrapper). Copies the input image header onto
        # the depth, which is what the scaler pairs on. Delayed so the TRT service
        # has written its "ready" status first.
        TimerAction(period=LaunchConfiguration('da3_service_startup_delay'), actions=[Node(
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
        )]),
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
        # map (EKF local ENU) -> mocap world. x/y/yaw are identical (R_enu_world = I);
        # only the EKF height offset differs.
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='map_to_mocap_tf',
            arguments=['--x', '0', '--y', '0', '--z', LaunchConfiguration('map_mocap_z_offset'),
                       '--frame-id', global_frame,
                       '--child-frame-id', LaunchConfiguration('mocap_frame')],
            parameters=[use_sim_time],
            condition=IfCondition(PythonExpression([
                "'", global_frame, "' != '", LaunchConfiguration('mocap_frame'), "' and '",
                LaunchConfiguration('start_nvblox'), "'.lower() == 'true'"])),
        ),
        # nvblox: metric DA3 depth + TF -> TSDF -> 3D ESDF
        Node(
            package='nvblox_ros',
            executable='nvblox_node',
            name='nvblox_node',
            output='screen',
            parameters=[LaunchConfiguration('nvblox_params'),
                        {'global_frame': global_frame},
                        use_sim_time],
            remappings=[('camera_0/depth/image', LaunchConfiguration('metric_depth_topic')),
                        ('camera_0/depth/camera_info', LaunchConfiguration('metric_info_topic'))],
            ros_arguments=['--log-level', LaunchConfiguration('nvblox_log_level')],
            condition=IfCondition(LaunchConfiguration('start_nvblox')),
        ),
        # 3D ESDF as topics: /esdf_publisher/esdf_grid (EsdfGrid), /esdf_publisher/esdf_cloud
        Node(
            package='da3_metric_scale',
            executable='esdf_publisher',
            name='esdf_publisher',
            output='screen',
            parameters=[LaunchConfiguration('esdf_params'),
                        {'global_frame': global_frame},
                        use_sim_time],
            condition=IfCondition(PythonExpression([
                "'", LaunchConfiguration('start_nvblox'), "'.lower() == 'true' and '",
                LaunchConfiguration('start_esdf_publisher'), "'.lower() == 'true'"])),
        ),
    ])
