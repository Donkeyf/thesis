"""Fast-Planner on the nvblox ESDF + trajectory tracker -> ArduCopter GUIDED.

Needs the mapping pipeline running (mocap_da3.launch.py: DA3 -> scaler -> nvblox ->
esdf_publisher) and MAVROS (mocap_to_mavros.launch.py). Then:

  ros2 launch da3_metric_scale fast_planner.launch.py
  # take off, switch to GUIDED, then
  ros2 service call /trajectory_tracker/enable std_srvs/srv/Trigger
  # goal: RViz "2D Goal Pose" (Fixed Frame map, z = default_goal_z) or
  ros2 topic pub --once /fast_planner/goal geometry_msgs/msg/PoseStamped \\
    "{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5, z: 1.0}}}"

Args: start_tracker (true), mavros_prefix (/mavros/mavros), planner_params, tracker_params,
      use_sim_time (false)
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
                              default_value=os.path.join(share, 'config', 'fast_planner.yaml')),
        DeclareLaunchArgument('tracker_params',
                              default_value=os.path.join(share, 'config', 'trajectory_tracker.yaml')),
        DeclareLaunchArgument('mavros_prefix', default_value='/mavros/mavros'),
        DeclareLaunchArgument('start_tracker', default_value='true'),
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        Node(
            package='da3_metric_scale',
            executable='fast_planner',
            name='fast_planner',
            output='screen',
            parameters=[LaunchConfiguration('planner_params'),
                        {'mavros_prefix': prefix, 'use_sim_time': sim,
                         'trajectory_topic': '/trajectory_tracker/trajectory'}],
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
