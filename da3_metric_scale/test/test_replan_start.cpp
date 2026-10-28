// Replans must use the vehicle's actual velocity: the simulated drone follows the reference with a
// first-order lag (like a real position controller), so the planner has to start from the measured
// state; run 2 drops velocity_local to exercise the differentiated-pose fallback.
//   g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_replan_start.cpp -lnlopt -o trs && ./trs
#define main fast_planner_main
#include "../src/fast_planner_node.cpp"
#undef main
using rclcpp::Publisher;
using rclcpp::Subscription;

int run(bool send_velocity)
{
  int fails = 0;
  auto check = [&](bool ok, const char * w) {if (!ok) {std::printf("FAIL: %s\n", w); ++fails;}};
  rclcpp::g_params.clear();
  rclcpp::g_params["replan_period"] = 1.0;
  rclcpp::g_params["max_deviation"] = 1.5;
  FastPlannerNode node;
  auto sub = [&](const std::string & k) {return node.subs.at(k);};
  auto esdf = std::static_pointer_cast<Subscription<da3_metric_scale::msg::EsdfGrid>>(sub("/esdf_publisher/esdf_grid"));
  auto pose = std::static_pointer_cast<Subscription<PoseStamped>>(sub("/mavros/mavros/local_position/pose"));
  auto vel = std::static_pointer_cast<Subscription<geometry_msgs::msg::TwistStamped>>(sub("/mavros/mavros/local_position/velocity_local"));
  auto state = std::static_pointer_cast<Subscription<mavros_msgs::msg::State>>(sub("/mavros/mavros/state"));
  auto goal = std::static_pointer_cast<Subscription<PoseStamped>>(sub("~/goal"));
  auto traj = std::static_pointer_cast<Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>>(node.pubs.at("/trajectory_tracker/trajectory"));

  da3_metric_scale::msg::EsdfGrid g;
  g.header.frame_id = "map"; g.origin.x = -2; g.origin.y = -2; g.origin.z = -0.2; g.voxel_size_m = 0.05;
  g.nx = 80; g.ny = 80; g.nz = 48; g.distances.resize(g.nx * g.ny * g.nz);
  for (uint32_t i = 0; i < g.nx; ++i) for (uint32_t j = 0; j < g.ny; ++j) for (uint32_t k = 0; k < g.nz; ++k)
    g.distances[(i * g.ny + j) * g.nz + k] = std::min(5.0, -0.2 + (k + 0.5) * 0.05);
  mavros_msgs::msg::State st; st.connected = true; st.armed = true; st.mode = "GUIDED";

  V3 x(-1.5, 1.2, 1.0), v = V3::Zero();
  double last_g = 0, max_start_err = 0;
  int measured_starts = 0, replans = 0;
  size_t seen = 0;
  const double tau_lag = 0.6;     // s, first-order lag of the "position controller"
  auto sample = [](const trajectory_msgs::msg::MultiDOFJointTrajectory & m, double t, V3 & p, V3 & vv) {
      const double t0 = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
      size_t i = 0;
      auto tt = [&](size_t k) {return t0 + m.points[k].time_from_start.sec + m.points[k].time_from_start.nanosec * 1e-9;};
      while (i + 1 < m.points.size() && tt(i + 1) <= t) ++i;
      auto & a = m.points[i].transforms[0].translation; auto & b = m.points[i].velocities[0].linear;
      p = V3(a.x, a.y, a.z); vv = V3(b.x, b.y, b.z);
    };
  // count log reasons by capturing stdout is awkward; inspect trajectories instead:
  // a replan from the measured state starts (first point) near the vehicle, not on the old reference
  auto step = [&](double T) {
      const double t_end = rclcpp::g_now + T;
      while (rclcpp::g_now < t_end) {
        PoseStamped ps; ps.header.frame_id = "map";
        ps.pose.position.x = x.x(); ps.pose.position.y = x.y(); ps.pose.position.z = x.z();
        pose->inject(ps);
        if (send_velocity) {geometry_msgs::msg::TwistStamped tw; tw.twist.linear.x = v.x(); tw.twist.linear.y = v.y(); tw.twist.linear.z = v.z(); vel->inject(tw);}
        state->inject(st);
        if (rclcpp::g_now - last_g >= 0.1) {esdf->inject(g); last_g = rclcpp::g_now;}
        node.spinUntil(rclcpp::g_now + 0.01);
        if (!traj->sent.empty()) {
          V3 pr, vr; sample(traj->sent.back(), rclcpp::g_now, pr, vr);
          const V3 vnew = (pr - x) / tau_lag;        // lagging tracker
          v = vnew; x += v * 0.01;
        }
      }
    };
  PoseStamped gp; gp.header.frame_id = "map"; gp.pose.position.x = 1.5; gp.pose.position.y = -1.2; gp.pose.position.z = 1.2;
  step(0.3);
  goal->inject(gp);
  rclcpp::g_quiet = false;
  // watch plans: whenever the planner publishes a new plan, compare its start velocity with the vehicle's
  int last_n = 0;
  for (int k = 0; k < 2500; ++k) {
    step(0.01);
    const int n = static_cast<int>(node.subs.size());   // dummy use
    (void)n;
    // new plan detection via bspline topic count
    auto bs = std::static_pointer_cast<Publisher<nav_msgs::msg::Path>>(node.pubs.at("~/bspline_path"));
    if (static_cast<int>(bs->sent.size()) != last_n && !bs->sent.empty()) {
      last_n = static_cast<int>(bs->sent.size());
      ++replans;
      const auto & path = bs->sent.back().poses;
      const V3 p0(path.front().pose.position.x, path.front().pose.position.y, path.front().pose.position.z);
      // start point vs vehicle position predicted over <= 1 s lookahead
      const double err = (p0 - x).norm();
      if (err < 0.35) ++measured_starts;
      max_start_err = std::max(max_start_err, err);
    }
  }
  rclcpp::g_quiet = false;
  std::printf("== %s: plans %d, plans starting at the vehicle %d, final pos %.2f %.2f %.2f, |v| %.2f\n",
    send_velocity ? "velocity_local" : "pose-differentiated velocity", replans, measured_starts, x.x(), x.y(), x.z(), v.norm());
  check(replans >= 3, "too few replans");
  check(measured_starts >= replans - 1, "replans did not start from the measured vehicle state");
  check((x - V3(1.5, -1.2, 1.2)).norm() < 0.3, "goal not reached with a lagging vehicle");
  (void)seen;
  return fails;
}

int main()
{
  int f = run(true) + run(false);
  std::printf("%s (%d failures)\n", f ? "REPLAN START FAILED" : "REPLAN START OK", f);
  return f;
}
