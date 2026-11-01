"""RAPTOR planner (Fast-Planner + Sec. VI risk-aware refinement + Sec. VII perception-aware yaw)
+ feature map (good features for the yaw planner) + trajectory tracker -> GUIDED.

Needs the mapping pipeline running (mocap_da3.launch.py: DA3 -> scaler -> nvblox ->
esdf_publisher) and MAVROS (mocap_to_mavros.launch.py). Then:

  ros2 launch da3_metric_scale raptor_planner.launch.py
  # take off, switch to GUIDED, then
  ros2 service call /trajectory_tracker/enable std_srvs/srv/Trigger
  # goal: RViz "2D Goal Pose" (Fixed Frame map, z = default_goal_z) or
  ros2 topic pub --once /raptor_planner/goal geometry_msgs/msg/PoseStamped \\
    "{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5, z: 1.0}}}"

Args: start_tracker (true), start_feature_map (true), mavros_prefix (/mavros/mavros),
      planner_params, tracker_params, feature_params, use_sim_time (false)
RViz: /raptor_planner/yaw_plan (Path, Pose Style: Arrows) = planned yaw waypoints,
      /feature_map/landmarks (PointCloud2, colour by 'score') = landmarks kept in view.
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
    prefix = LaunchConfiguration('mavros_prefix')
    sim = LaunchConfiguration('use_sim_time')
    return LaunchDescription([
        DeclareLaunchArgument('planner_params',
                              default_value=os.path.join(share, 'config', 'raptor_planner.yaml')),
        DeclareLaunchArgument('tracker_params',
                              default_value=os.path.join(share, 'config', 'trajectory_tracker.yaml')),
        DeclareLaunchArgument('feature_params',
                              default_value=os.path.join(share, 'config', 'feature_map.yaml')),
        DeclareLaunchArgument('start_feature_map', default_value='true'),
        DeclareLaunchArgument('mavros_prefix', default_value='/mavros/mavros'),
        DeclareLaunchArgument('start_tracker', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='da3_metric_scale',
            executable='raptor_planner',
            name='raptor_planner',
            output='screen',
            parameters=[LaunchConfiguration('planner_params'),
                        {'mavros_prefix': prefix, 'use_sim_time': sim,
                         'trajectory_topic': '/trajectory_tracker/trajectory'}],
        ),
        Node(
            package='da3_metric_scale',
            executable='feature_map',
            name='feature_map',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_feature_map')),
            parameters=[LaunchConfiguration('feature_params'), {'use_sim_time': sim}],
        ),
        Node(
            package='da3_metric_scale',
            executable='trajectory_tracker',
            name='trajectory_tracker',
            output='screen',
            condition=IfCondition(LaunchConfiguration('start_tracker')),
            parameters=[LaunchConfiguration('tracker_params'),
                        {'mavros_prefix': prefix, 'use_sim_time': sim,
                         'yaw_mode': 'trajectory'}],
        ),
    ])
