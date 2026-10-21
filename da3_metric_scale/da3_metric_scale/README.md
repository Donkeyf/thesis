# da3_metric_scale

ROS 2 (Jazzy) package with two nodes:

| Node | Does |
|---|---|
| `mocap_to_mavros` | OptiTrack pose → ENU/FLU → `/mavros/mavros/vision_pose/pose` (ArduPilot ExternalNav), sets EKF origin |
| `da3_metric_scaler` | Metric scale for DA3 depth from mocap camera poses (floor plane + triangulation), publishes metric depth + camera TF for nvblox |

## Build

```bash
cd /workspaces/isaac_ros-dev
cp -r da3_metric_scale src/
colcon build --packages-select da3_metric_scale
source install/setup.bash
```

Needs `vrpn_mocap`, `mavros_msgs`, `geographic_msgs`, OpenCV, Eigen.

## Run

```bash
# everything
ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<motive_ip>

# only the ArduPilot bridge
ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=<ip> start_scaler:=false

# only the scaler (e.g. on a bag that already has the mocap topic)
ros2 launch da3_metric_scale mocap_da3.launch.py start_vrpn:=false start_mavros_bridge:=false use_sim_time:=true
```

## Configure

- `config/mocap_to_mavros.yaml` – topics, rate, `R_enu_world`, `R_body_flu`, `p_body_fc`, EKF origin
- `config/da3_metric_scale.yaml` – paste Kalibr `cam0/T_cam_imu`, `p_body_imu`, DA3 topics, method

Both configs assume the mocap world is **Z-up** and rigid body "dilon" is published with
x = left, −y = nose forward, z = up. Offsets (`p_body_imu`, `p_body_fc`) are in those body axes.

## ArduPilot

`VISO_TYPE=1`, `EK3_SRC1_POSXY=6`, `EK3_SRC1_POSZ=6`, `EK3_SRC1_YAW=6`, `EK3_SRC1_VELXY=0`,
`COMPASS_USE=0`, `GPS1_TYPE=0`, `VISO_POS_*=0` if `p_body_fc` is set.
