# da3_metric_scale

ROS 2 (Jazzy, Kilted, Lyrical) package with three nodes:

| Node | Does |
|---|---|
| `mocap_to_mavros` | OptiTrack pose → ENU/FLU → `/mavros/mavros/vision_pose/pose` (ArduPilot ExternalNav), sets EKF origin |
| `da3_metric_scaler` | Metric scale for DA3 depth from mocap camera poses (floor plane + triangulation), publishes metric depth + camera TF for nvblox |
| `guided_waypoints` | Sends waypoints to ArduCopter via MAVROS setpoints, only while connected + armed + GUIDED; pauses when you leave GUIDED |

## Build

```bash
cd /workspaces/isaac_ros-dev
cp -r da3_metric_scale src/
colcon build --packages-select da3_metric_scale
source install/setup.bash
```

Needs `vrpn_mocap`, `mavros_msgs`, `geographic_msgs`, OpenCV, Eigen.

## Run

DA3 and its TensorRT service (`scripts/trt_inference_service_shm.py` from the wrapper repo) are
launched from here too. Use `start_da3_service:=false` if you start the service yourself, and
`da3_service_script:=<path>` if it isn't found automatically.

```bash
# everything, live
ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<motive_ip>

# rosbag: DA3 + scaler only. Start this first, then:  ros2 bag play <bag> --clock --rate 0.25
ros2 launch da3_metric_scale mocap_da3.launch.py start_vrpn:=false start_throttle:=false \
  start_mavros_bridge:=false use_sim_time:=true mocap_pose_topic:=/mavros/dilon/pose_throttled

# only the ArduPilot bridge (also starts MAVROS; start_mavros:=false if it already runs)
ros2 launch da3_metric_scale mocap_to_mavros.launch.py motive_ip:=<ip> fcu_url:=/dev/ttyACM0:921600
```

### Waypoints (GUIDED)

```bash
ros2 launch da3_metric_scale guided_waypoints.launch.py          # MAVROS + bridge already running
# arm, switch to GUIDED on the RC (take off first unless takeoff_alt > 0), then:
ros2 service call /guided_waypoints/start std_srvs/srv/Trigger
ros2 service call /guided_waypoints/stop  std_srvs/srv/Trigger   # stop sending, FC holds
ros2 topic echo /guided_waypoints/status
# single point / new mission on the fly (EKF local ENU frame, frame_id map):
ros2 topic pub --once /guided_waypoints/goto geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 0.5, y: 0.0, z: 1.0}, orientation: {w: 1.0}}}"
```
Waypoints and fence are in `config/guided_waypoints.yaml`.

DA3 is started with `normalize_depth:=false` (required, per-frame normalisation breaks the
scale) and `keep_image_size:=true`. Change model with `da3_model:=depth-anything/DA3-SMALL`.

## Configure

- `config/mocap_to_mavros.yaml` – topics, rate, `R_enu_world`, `R_body_flu`, `p_body_fc`, EKF origin
- `config/da3_metric_scale.yaml` – paste Kalibr `cam0/T_cam_imu`, `p_body_imu`, DA3 topics, method

Both configs assume the mocap world is **Z-up** and rigid body "dilon" is published with
x = left, −y = nose forward, z = up. Offsets (`p_body_imu`, `p_body_fc`) are in those body axes.

## ArduPilot

`VISO_TYPE=1`, `EK3_SRC1_POSXY=6`, `EK3_SRC1_POSZ=6`, `EK3_SRC1_YAW=6`, `EK3_SRC1_VELXY=0`,
`COMPASS_USE=0`, `GPS1_TYPE=0`, `VISO_POS_*=0` if `p_body_fc` is set.
