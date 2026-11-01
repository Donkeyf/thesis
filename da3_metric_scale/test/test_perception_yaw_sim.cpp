// Closed-loop test of raptor_planner with yaw_mode perception (mocked ROS):
//   a) unknown space beside the path  -> the published yaw turns towards it
//   b) map known, landmarks on one side -> the published yaw keeps them in view
//   c) yaw rate of the published trajectory stays within max_yaw_rate (+ tolerance)
//   g++ -O2 -std=c++17 -DRAPTOR_NODE -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_perception_yaw_sim.cpp -lnlopt -o pyaw && ./pyaw
#define main raptor_main
#include "../src/fast_planner_node.cpp"
#undef main

using rclcpp::Publisher;
using rclcpp::Subscription;

template<class T> std::shared_ptr<T> get(rclcpp::Node & n, const std::string & k, bool sub)
{
  auto & m = sub ? n.subs : n.pubs;
  auto it = m.find(k);
  if (it == m.end()) {std::printf("missing %s\n", k.c_str()); std::exit(2);}
  return std::static_pointer_cast<T>(it->second);
}

static constexpr double DEG = 180.0 / M_PI;

// box around the vehicle (like esdf_publisher): 6 x 6 x 2 m, 5 cm; floor at z = 0
da3_metric_scale::msg::EsdfGrid grid(const V3 & c, bool unknown_left)
{
  da3_metric_scale::msg::EsdfGrid g;
  g.header.frame_id = "map";
  g.voxel_size_m = 0.05;
  g.origin.x = std::round((c.x() - 3) / 0.05) * 0.05;
  g.origin.y = std::round((c.y() - 3) / 0.05) * 0.05;
  g.origin.z = -0.2;
  g.nx = 120; g.ny = 120; g.nz = 44;
  g.distances.resize(g.nx * g.ny * g.nz);
  for (uint32_t i = 0; i < g.nx; ++i) for (uint32_t j = 0; j < g.ny; ++j) for (uint32_t k = 0; k < g.nz; ++k) {
    const V3 p(g.origin.x + (i + 0.5) * 0.05, g.origin.y + (j + 0.5) * 0.05, g.origin.z + (k + 0.5) * 0.05);
    float d = std::min(2.0, p.z());
    if (unknown_left && p.y() > 1.0 && p.x() > 0.5 && p.x() < 6.0) {d = -1000.f;}
    g.distances[(i * g.ny + j) * g.nz + k] = d;
  }
  return g;
}

struct Run {double mean_off{0}, max_rate{0}; size_t yaw_msgs{0}; V3 end;};

Run fly(bool unknown_left, bool landmarks_right, double T)
{
  rclcpp::g_now = 1000.0;
  rclcpp::g_params.clear();
  rclcpp::g_params["mission_goals"] = std::vector<double>{6.0, 0.0, 1.0, 0.0};
  rclcpp::g_params["fence_min"] = std::vector<double>{-10.0, -10.0, 0.3};
  rclcpp::g_params["fence_max"] = std::vector<double>{10.0, 10.0, 1.6};
  rclcpp::g_params["max_vel"] = 0.6;
  rclcpp::g_params["risk.enable"] = false;           // isolate the yaw planner
  rclcpp::g_params["replan_period"] = 1.0;
  rclcpp::g_service_handlers["/trajectory_tracker/enable"] = [](std::shared_ptr<void>) {
      auto r = std::make_shared<std_srvs::srv::Trigger::Response>();
      r->success = true; return std::static_pointer_cast<void>(r);};

  FastPlannerNode node;
  auto esdf = get<Subscription<da3_metric_scale::msg::EsdfGrid>>(node, "/esdf_publisher/esdf_grid", true);
  auto pose = get<Subscription<PoseStamped>>(node, "/mavros/mavros/local_position/pose", true);
  auto state = get<Subscription<mavros_msgs::msg::State>>(node, "/mavros/mavros/state", true);
  auto lms = get<Subscription<sensor_msgs::msg::PointCloud2>>(node, "/feature_map/landmarks", true);
  auto traj = get<Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>>(node, "/trajectory_tracker/trajectory", false);
  auto yawp = get<Publisher<nav_msgs::msg::Path>>(node, "~/yaw_plan", false);

  sensor_msgs::msg::PointCloud2 pc;
  pc.header.frame_id = "map";
  const char * names[4] = {"x", "y", "z", "score"};
  for (uint32_t i = 0; i < 4; ++i) {
    sensor_msgs::msg::PointField f; f.name = names[i]; f.offset = 4 * i;
    f.datatype = sensor_msgs::msg::PointField::FLOAT32; f.count = 1; pc.fields.push_back(f);
  }
  pc.point_step = 16;
  if (landmarks_right) {
    for (double x = 0; x <= 7.0; x += 0.2) {
      for (double z = 0.6; z <= 1.4; z += 0.2) {
        const float v[4] = {static_cast<float>(x), -2.0f, static_cast<float>(z), 0.8f};
        pc.data.insert(pc.data.end(), reinterpret_cast<const uint8_t *>(v),
          reinterpret_cast<const uint8_t *>(v) + 16);
        ++pc.width;
      }
    }
  }
  pc.height = 1;
  pc.row_step = pc.point_step * pc.width;

  mavros_msgs::msg::State st;
  st.connected = true; st.armed = true; st.mode = "GUIDED";
  V3 x(0, 0, 1.0);
  double yaw = 0, last_g = 0, last_l = 0;
  Run run;
  double off_sum = 0; int off_n = 0;
  const double t_end = rclcpp::g_now + T;
  while (rclcpp::g_now < t_end) {
    PoseStamped ps; ps.header.frame_id = "map";
    ps.pose.position.x = x.x(); ps.pose.position.y = x.y(); ps.pose.position.z = x.z();
    ps.pose.orientation.z = std::sin(0.5 * yaw); ps.pose.orientation.w = std::cos(0.5 * yaw);
    pose->inject(ps);
    state->inject(st);
    if (rclcpp::g_now - last_g >= 0.1) {esdf->inject(grid(x, unknown_left)); last_g = rclcpp::g_now;}
    if (rclcpp::g_now - last_l >= 0.5) {pc.header.stamp = node.now(); lms->inject(pc); last_l = rclcpp::g_now;}
    node.spinUntil(rclcpp::g_now + 0.02);
    if (!traj->sent.empty()) {
      const auto & m = traj->sent.back();
      const auto & p0 = m.points.front();
      x = V3(p0.transforms[0].translation.x, p0.transforms[0].translation.y, p0.transforms[0].translation.z);
      yaw = 2 * std::atan2(p0.transforms[0].rotation.z, p0.transforms[0].rotation.w);
      run.max_rate = std::max(run.max_rate, std::fabs(p0.velocities[0].angular.z));
      const double vx = p0.velocities[0].linear.x, vy = p0.velocities[0].linear.y;
      if (std::hypot(vx, vy) > 0.3) {   // offset of the camera from the flight direction
        off_sum += std::atan2(std::sin(yaw - std::atan2(vy, vx)), std::cos(yaw - std::atan2(vy, vx)));
        ++off_n;
      }
    }
  }
  run.mean_off = off_n ? off_sum / off_n : 0.0;
  run.yaw_msgs = yawp->sent.size();
  run.end = x;
  return run;
}

int main()
{
  int fails = 0;
  auto check = [&](bool ok, const char * w) {if (!ok) {std::printf("FAIL: %s\n", w); ++fails;}};
  rclcpp::g_quiet = true;

  const Run a = fly(false, false, 14.0);
  std::printf("known map, no landmarks : mean camera offset from velocity %+5.1f deg, max yaw rate %.0f deg/s, end %.2f %.2f\n",
    a.mean_off * DEG, a.max_rate * DEG, a.end.x(), a.end.y());
  const Run b = fly(true, false, 14.0);
  std::printf("unknown on the left     : mean camera offset from velocity %+5.1f deg, max yaw rate %.0f deg/s, end %.2f %.2f\n",
    b.mean_off * DEG, b.max_rate * DEG, b.end.x(), b.end.y());
  const Run c = fly(false, true, 14.0);
  std::printf("landmarks on the right  : mean camera offset from velocity %+5.1f deg, max yaw rate %.0f deg/s, end %.2f %.2f\n",
    c.mean_off * DEG, c.max_rate * DEG, c.end.x(), c.end.y());

  check(std::fabs(a.mean_off) < 8.0 / DEG, "known map: camera should look along the velocity");
  check(b.mean_off > 15.0 / DEG, "unknown left: camera should turn left");
  check(c.mean_off < -15.0 / DEG, "landmarks right: camera should turn right");
  check(a.yaw_msgs > 0 && b.yaw_msgs > 0, "no ~/yaw_plan published");
  for (const Run * r : {&a, &b, &c}) {
    check(r->max_rate < 50.0 / DEG, "yaw rate above the limit");
    check((r->end - V3(6, 0, 1)).norm() < 0.3, "did not reach the goal");
  }
  std::printf("%s (%d failures)\n", fails ? "PERCEPTION YAW SIM FAILED" : "PERCEPTION YAW SIM OK", fails);
  return fails;
}
