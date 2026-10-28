# da3_metric_scale

ROS 2 (Jazzy, Kilted, Lyrical) package with seven nodes (+ nvblox launch):

| Node | Does |
|---|---|
| `mocap_to_mavros` | OptiTrack pose → ENU/FLU → `/mavros/mavros/vision_pose/pose` (ArduPilot ExternalNav), sets EKF origin |
| `da3_metric_scaler` | Metric scale for DA3 depth from mocap camera poses (floor plane + triangulation), publishes metric depth + camera TF for nvblox |
| `guided_waypoints` | Sends waypoints to ArduCopter via MAVROS setpoints, only while connected + armed + GUIDED; pauses when you leave GUIDED |
| `esdf_publisher` | Pulls nvblox's 3D ESDF (`get_esdf_and_gradient`) around the drone and publishes `~/esdf_grid` (`da3_metric_scale/msg/EsdfGrid`) and `~/esdf_cloud` (PointCloud2 x,y,z,distance) |
| `fast_planner` | Fast-Planner (Zhou et al., RA-L 2019) on the ESDF grid: kinodynamic A* → B-spline optimisation (NLopt) → time adjustment → replanning; streams the trajectory to `trajectory_tracker`, only while armed + GUIDED |
| `raptor_planner` | `fast_planner` + RAPTOR (Zhou et al., T-RO 2021) Sec. VI risk-aware trajectory refinement: makes the point where the trajectory enters unknown space visible early enough to stop |
| `trajectory_tracker` | Samples a re-planned spline (MultiDOFJointTrajectory or Path, 5-10 Hz) at 50 Hz and streams position + velocity + acceleration + yaw + yaw-rate to GUIDED via `setpoint_raw/local` |

## Build

```bash
cd /workspaces/isaac_ros-dev
cp -r da3_metric_scale src/
colcon build --packages-select da3_metric_scale
source install/setup.bash
```

Needs `vrpn_mocap`, `mavros_msgs`, `geographic_msgs`, OpenCV, Eigen, Isaac ROS nvblox
(`nvblox_ros`, `nvblox_msgs`; without `nvblox_msgs` the `esdf_publisher` is skipped) and NLopt
for the planner (`sudo apt install libnlopt-cxx-dev`; without it `fast_planner` is skipped).

## Run

DA3 and its TensorRT service (`scripts/trt_inference_service_shm.py` from the wrapper repo) are
launched from here too. Use `start_da3_service:=false` if you start the service yourself, and
`da3_service_script:=<path>` if it isn't found automatically.

```bash
# everything, live, one launch: MAVROS + OptiTrack -> ArduPilot + usb_cam (100 fps -> 10 Hz)
# + DA3 + scaler + nvblox + ESDF
ros2 launch da3_metric_scale mocap_da3.launch.py motive_ip:=169.254.179.249 \
  fcu_url:=/dev/ttyACM0:921600 map_mocap_z_offset:=0.12
#   start_mavros:=false if MAVROS already runs; gcs_url:=udp://@<laptop-ip>:14550 for Mission Planner
#   camera args: video_device:=/dev/video0 camera_width:=640 camera_height:=480
#                camera_fps:=100.0 pixel_format:=mjpeg2rgb   (start_camera:=false for bags)

# rosbag: DA3 + scaler only. Start this first, then:  ros2 bag play <bag> --clock --rate 0.25
ros2 launch da3_metric_scale mocap_da3.launch.py start_camera:=false start_vrpn:=false start_throttle:=false \
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

### Spline / trajectory tracking (GUIDED)

```bash
ros2 launch da3_metric_scale trajectory_tracker.launch.py         # MAVROS + bridge already running
# planner -> /trajectory_tracker/trajectory (trajectory_msgs/MultiDOFJointTrajectory, preferred)
#         or /trajectory_tracker/path      (nav_msgs/Path)
# take off, switch to GUIDED, then:
ros2 service call /trajectory_tracker/enable  std_srvs/srv/Trigger
ros2 service call /trajectory_tracker/disable std_srvs/srv/Trigger
ros2 topic echo /trajectory_tracker/status
# test without a planner: 0.5 m circle at 1 m, 12 s per lap
python3 scripts/test_circle_trajectory.py --center 0 0 1.0 --radius 0.5 --period 12
```
Limits, feedforward switches and fence are in `config/trajectory_tracker.yaml`.

### Mapping / ESDF (nvblox)

The chain runs at **10 Hz**: `image_throttle` (`raw_image_topic` /image_raw -> `image_topic` /image_da3 at
`image_rate:=10.0`) -> DA3 -> scaler -> nvblox (ESDF 10 Hz) -> `esdf_publisher` (10 Hz).
For a bag that only has /image_throttled pass `raw_image_topic:=/image_throttled`.

`mocap_da3.launch.py` also starts **nvblox** (`nvblox_ros/nvblox_node`, 3D ESDF, 5 cm voxels) on the
scaler's rectified metric depth (`/da3_metric/depth`, TF `mocap -> camera_optical`) and the
`esdf_publisher`. The map is in `global_frame` (default `map`, the EKF/MAVROS local frame) with a
static TF `map -> mocap` (`map_mocap_z_offset:=0.12` if EKF z reads above mocap z), so the ESDF,
`trajectory_tracker` and MAVROS all share one frame.

```bash
ros2 topic hz /esdf_publisher/esdf_grid           # 3D ESDF for the planner (10 Hz)
# RViz (fixed frame map): /esdf_publisher/esdf_cloud (colour by 'distance'),
#   /nvblox_node/mesh, /nvblox_node/static_esdf_pointcloud (slice at 1 m)
start_nvblox:=false / start_esdf_publisher:=false to turn them off
```
EsdfGrid indexing: distance of voxel (ix,iy,iz) = `distances[(ix*ny + iy)*nz + iz]`, centre =
`origin + (i + 0.5) * voxel_size_m`; `unobserved_value` (-1000) = never seen.
Configs: `config/nvblox_da3.yaml`, `config/esdf_publisher.yaml`.

DA3 is started with `normalize_depth:=false` (required, per-frame normalisation breaks the
scale) and `keep_image_size:=true`. Change model with `da3_model:=depth-anything/DA3-SMALL`.

### Planning (Fast-Planner + tracker)

```bash
ros2 launch da3_metric_scale mocap_to_mavros.launch.py      # MAVROS + mocap bridge
ros2 launch da3_metric_scale mocap_da3.launch.py start_vrpn:=false start_throttle:=false \
  start_mavros_bridge:=false mocap_pose_topic:=/vrpn_mocap/dilon/pose_throttled   # mapping -> ESDF
ros2 launch da3_metric_scale fast_planner.launch.py          # planner + trajectory_tracker
# take off (Loiter/AltHold), switch to GUIDED, then
ros2 service call /trajectory_tracker/enable std_srvs/srv/Trigger
# goal: RViz "2D Goal Pose" (fixed frame map; z = default_goal_z 1.0 m) or
ros2 topic pub --once /fast_planner/goal geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: map}, pose: {position: {x: 1.0, y: 0.5, z: 1.0}}}"
ros2 topic echo /fast_planner/status                         # FSM state, last plan
ros2 service call /fast_planner/cancel std_srvs/srv/Trigger  # stop and hold
```

**Mission / GUIDED automation:** put goals in `mission_goals` (groups of x, y, z, yaw_deg in
`fast_planner.yaml` / `raptor_planner.yaml`). When the FC enters GUIDED (armed) the planner enables
the tracker itself (retrying until the tracker accepts, i.e. airborne) and flies the goals in order,
holding `mission_hold_time` at each. Leaving GUIDED pauses it; returning resumes at the same goal.
```bash
ros2 service call /fast_planner/start_mission std_srvs/srv/Trigger   # restart from goal 1
ros2 service call /fast_planner/cancel std_srvs/srv/Trigger          # stop mission, hold
ros2 topic echo /fast_planner/status                                 # ... | mission 2/3 | ...
```
Test: `g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_mission_sim.cpp -lnlopt -o msim && ./msim`

Per plan (`include/da3_metric_scale/`, ROS-free and unit-tested):
1. **Kinodynamic A*** (`kino_astar.hpp`): double integrator, (2r+1)³ primitives, J = ∫|u|² + ρT,
   closed-form heuristic, analytic expansion. A node is free if ESDF > `clearance` and inside the fence.
2. **B-spline fit** (`non_uniform_bspline.hpp`): uniform cubic, knot span ts = `ctrl_pt_dist / max_vel`;
   the first/last 3 control points are solved exactly from the start/end pos, vel, acc.
3. **Optimisation** (`bspline_optimizer.hpp`, NLopt `LD_LBFGS`, analytic gradients):
   λ1·f_s (elastic band) + λ2·f_c (ESDF, trilinear, d_thr) + λ3·(f_v + f_a). The paper's weights
   10 / 0.8 / 0.01 are the defaults. The first and last 3 control points stay fixed.
4. **Time adjustment** (Algorithm 2): knot spans with |V_i| > vmax or |A_i| > amax are stretched
   by min(α, ratio), α = 1.1, iterated. Knots that shape the start state are left alone, so
   replans stay continuous in position, velocity and acceleration.
5. **Check**: dense ESDF check (> `check_clearance`). If the optimised curve fails, the plain
   A* fit is tried, otherwise the plan is rejected.

Replanning: every `replan_period` (3 s) from the active trajectory at now + lookahead (receding
horizon). The start state accounts for the real vehicle (`replan_start: auto`): while the drone
tracks the reference within `replan_vel_mismatch` / `replan_pos_mismatch` the new plan continues
the reference (smooth), otherwise it starts from the measured pose + velocity predicted over the
lookahead (`velocity_local`, or differentiated pose if that stream is missing). Plans stay
`fence_margin` (0.1 m) inside the fence, which is shared with the tracker. Replans happen also immediately when a new ESDF shows the remaining trajectory in collision, and an
**emergency hold** when that collision is less than `emergency_time` away, or when the ESDF stops
arriving. Leaving GUIDED / disarming pauses the planner (goal kept, replans from odom on return).
The trajectory is republished at 10 Hz (pos, vel, acc, yaw, yaw rate every 50 ms), so the
tracker's 0.5 s staleness hold only triggers if the planner dies. Yaw follows the velocity
(`yaw_mode: velocity`) so the camera sees where it flies.

RViz: `/fast_planner/bspline_path`, `/fast_planner/astar_path`, `/fast_planner/control_points`
(Path), `/fast_planner/current_goal`, `/trajectory_tracker/reference`, `/esdf_publisher/esdf_cloud`.

Offline tests (no ROS needed):
```bash
g++ -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 test/test_fast_planner.cpp -lnlopt -o tfp && ./tfp
g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_planner_sim.cpp -lnlopt -o sim && ./sim
```
`test_planner_sim` runs the real node against a mocked ROS API in closed loop (room with
obstacles, a wall and pop-up obstacles appearing mid-flight, leaving GUIDED, losing the ESDF).
Config: `config/fast_planner.yaml`.

### RAPTOR planner (risk-aware refinement, Sec. VI)

Same pipeline, inputs, outputs, services and tracker as `fast_planner`, plus one stage between
the NLopt optimisation and the time adjustment (`include/da3_metric_scale/risk_aware_refiner.hpp`):

1. **Frontier intersection**: the first point p_f where the trajectory enters unobserved space
   (unobserved ESDF voxels, or outside the ESDF box).
2. **Critical view**: visibility ψ(p, p_f) = smallest ESDF distance on the line of sight (Eq. 5).
   t_c is the first time from which ψ ≥ `psi_min` holds until p_f, giving p_c and the
   critical view direction v_c.
3. **Worst-case check** (Eq. 6): if v(t_c)²/2a_max ≤ |p_c − p_f| − R_q, the drone can stop when it
   first sees p_f, and nothing is changed.
4. **Iterative refinement** (Algorithm 3): pick t_s, then re-optimise with
   f_r = f_p2 + λ_r (f_dv + w_r f_dsf) (Eqs. 10–14, analytic gradients, NLopt). This puts the drone
   on the critical view ray at t_s (within δ_v) and at least d_s = v²/2a_max + R_q from p_f. The
   speed estimate grows by α until the criterion holds. If the refined curve fails the collision
   check, the unrefined one is used.

```bash
ros2 launch da3_metric_scale raptor_planner.launch.py        # instead of fast_planner.launch.py
ros2 topic echo /raptor_planner/status                       # "... | risk: refined, safe after 2 it ..."
```
RViz (PoseStamped): `/raptor_planner/frontier_point` (p_f), `/raptor_planner/critical_view_point`
(p_c, arrow along v_c), `/raptor_planner/view_point` (p_r(t_s)), plus the same paths as fast_planner.
Config: `config/raptor_planner.yaml` (`risk.*`). Keep `unknown_is_occupied: false`, otherwise
trajectories never enter unknown space and nothing is refined.

Offline test (paper Fig. 10/11 scene: wall corner with unobserved space behind it):
```bash
g++ -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 test/test_risk_refine.cpp -lnlopt -o trr && ./trr
g++ -O2 -std=c++17 -DRAPTOR_NODE -DSIM_UNKNOWN -Itest/mock -Iinclude -I/usr/include/eigen3 \
    test/test_planner_sim.cpp -lnlopt -o simr && ./simr
```

## Configure

- `config/mocap_to_mavros.yaml` – topics, rate, `R_enu_world`, `R_body_flu`, `p_body_fc`, EKF origin
- `config/da3_metric_scale.yaml` – paste Kalibr `cam0/T_cam_imu`, `p_body_imu`, DA3 topics, method

Both configs assume the mocap world is **Z-up** and rigid body "dilon" is published with
x = left, −y = nose forward, z = up. Offsets (`p_body_imu`, `p_body_fc`) are in those body axes.

## ArduPilot

`VISO_TYPE=1`, `EK3_SRC1_POSXY=6`, `EK3_SRC1_POSZ=6`, `EK3_SRC1_YAW=6`, `EK3_SRC1_VELXY=0`,
`COMPASS_USE=0`, `GPS1_TYPE=0`, `VISO_POS_*=0` if `p_body_fc` is set.
