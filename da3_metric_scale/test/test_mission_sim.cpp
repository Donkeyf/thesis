// Closed-loop test of the planner's GUIDED automation (mocked ROS): mission auto-start on GUIDED,
// tracker auto-enable (refused while on the ground, retried), pause / resume when leaving GUIDED.
//   g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_mission_sim.cpp -lnlopt -o msim && ./msim
#define main fast_planner_main
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

int main()
{
  int fails = 0;
  auto check = [&](bool ok, const char * w) {if (!ok) {std::printf("FAIL: %s\n", w); ++fails;}};
  rclcpp::g_params["mission_goals"] = std::vector<double>{
    1.0, 1.0, 1.0, 0.0,
    1.0, -1.0, 1.2, 90.0,
    -1.0, -1.0, 1.0, 0.0};
  rclcpp::g_params["mission_hold_time"] = 1.0;

  // fake trajectory_tracker: enable succeeds only when airborne; leaving GUIDED disables it
  bool airborne = false, tracker_enabled = false;
  mavros_msgs::msg::State st;
  st.connected = true; st.armed = true; st.mode = "LOITER";
  int enable_calls = 0;
  rclcpp::g_service_handlers["/trajectory_tracker/enable"] = [&](std::shared_ptr<void>) {
      ++enable_calls;
      auto r = std::make_shared<std_srvs::srv::Trigger::Response>();
      if (st.mode != "GUIDED") {r->success = false; r->message = "not GUIDED";}
      else if (!airborne) {r->success = false; r->message = "on the ground - take off first";}
      else {tracker_enabled = true; r->success = true; r->message = "enabled";}
      return std::static_pointer_cast<void>(r);
    };

  FastPlannerNode node;
  auto esdf = get<Subscription<da3_metric_scale::msg::EsdfGrid>>(node, "/esdf_publisher/esdf_grid", true);
  auto pose = get<Subscription<PoseStamped>>(node, "/mavros/mavros/local_position/pose", true);
  auto state = get<Subscription<mavros_msgs::msg::State>>(node, "/mavros/mavros/state", true);
  auto tstat = get<Subscription<std_msgs::msg::String>>(node, "/trajectory_tracker/status", true);
  auto traj = get<Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>>(node, "/trajectory_tracker/trajectory", false);
  auto status = get<Publisher<std_msgs::msg::String>>(node, "~/status", false);

  // empty room with a floor
  da3_metric_scale::msg::EsdfGrid g;
  g.header.frame_id = "map";
  g.origin.x = -2; g.origin.y = -2; g.origin.z = -0.2; g.voxel_size_m = 0.05;
  g.nx = 80; g.ny = 80; g.nz = 48;
  g.distances.resize(g.nx * g.ny * g.nz);
  for (uint32_t i = 0; i < g.nx; ++i) for (uint32_t j = 0; j < g.ny; ++j) for (uint32_t k = 0; k < g.nz; ++k)
    g.distances[(i * g.ny + j) * g.nz + k] = std::min(5.0, -0.2 + (k + 0.5) * 0.05);

  V3 x(-1.5, 1.2, 1.0);
  double last_g = 0, last_s = 0;
  auto run = [&](double T) {
    const double t_end = rclcpp::g_now + T;
    while (rclcpp::g_now < t_end) {
      PoseStamped ps; ps.header.frame_id = "map";
      ps.pose.position.x = x.x(); ps.pose.position.y = x.y(); ps.pose.position.z = x.z();
      pose->inject(ps);
      state->inject(st);
      if (st.mode != "GUIDED") tracker_enabled = false;
      if (rclcpp::g_now - last_g >= 0.1) {esdf->inject(g); last_g = rclcpp::g_now;}
      if (rclcpp::g_now - last_s >= 1.0) {
        std_msgs::msg::String m; m.data = tracker_enabled ? "TRACKING | ..." : "DISABLED | ...";
        tstat->inject(m); last_s = rclcpp::g_now;
      }
      node.spinUntil(rclcpp::g_now + 0.01);
      // vehicle follows the newest trajectory only while the tracker is enabled
      if (tracker_enabled && !traj->sent.empty()) {
        const auto & m = traj->sent.back();
        const double t0 = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
        size_t i = 0;
        auto tt = [&](size_t k) {return t0 + m.points[k].time_from_start.sec + m.points[k].time_from_start.nanosec * 1e-9;};
        while (i + 1 < m.points.size() && tt(i + 1) <= rclcpp::g_now) ++i;
        auto & tr = m.points[i].transforms[0].translation;
        x = V3(tr.x, tr.y, tr.z);
      }
    }
  };
  auto stat = [&]() {return status->sent.empty() ? std::string("-") : status->sent.back().data;};

  run(2.0);                                         // LOITER: nothing should happen
  check(stat().find("mission") == std::string::npos, "mission started outside GUIDED");

  st.mode = "GUIDED";                               // GUIDED while still on the ground
  run(3.0);
  std::printf("== GUIDED on ground: enable calls %d, tracker %d | %s\n", enable_calls, tracker_enabled, stat().c_str());
  check(!tracker_enabled && enable_calls >= 2, "enable should be refused and retried while on the ground");
  check(stat().find("mission 1/3") != std::string::npos, "mission did not auto-start on GUIDED");

  airborne = true;                                  // take-off happened -> next retry succeeds
  run(9.0);
  std::printf("== airborne: enable calls %d, tracker %d, pos %.2f %.2f %.2f | %s\n", enable_calls, tracker_enabled,
    x.x(), x.y(), x.z(), stat().c_str());
  check(tracker_enabled, "tracker not enabled after take-off");

  run(4.0);                                         // progress into goal 2
  std::string before = stat();
  st.mode = "LOITER";                               // pilot takes over mid-mission
  run(3.0);
  const V3 x_loiter = x;
  run(2.0);
  std::printf("== LOITER mid-mission: %s | moved %.3f m while in LOITER\n", stat().c_str(), (x - x_loiter).norm());
  check((x - x_loiter).norm() < 1e-9, "vehicle moved while tracker disabled");
  check(stat().find("WAIT_READY") != std::string::npos, "planner not paused in LOITER");

  const int calls_before = enable_calls;
  st.mode = "GUIDED";                               // back to GUIDED -> re-enable + resume
  run(40.0);
  std::printf("== resumed: re-enable calls %d, pos %.2f %.2f %.2f | %s\n", enable_calls - calls_before,
    x.x(), x.y(), x.z(), stat().c_str());
  check(enable_calls > calls_before, "tracker not re-enabled after returning to GUIDED");
  check(stat().find("mission done") != std::string::npos, "mission did not complete");
  check((x - V3(-1.0, -1.0, 1.0)).norm() < 0.2, "did not end at the last mission goal");
  (void)before;

  // manual goal overrides; ~/start_mission restarts
  auto srv = std::static_pointer_cast<rclcpp::Service<std_srvs::srv::Trigger>>(node.srvs["~/start_mission"]);
  auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
  auto res = std::make_shared<std_srvs::srv::Trigger::Response>();
  srv->cb(req, res);
  run(1.0);
  std::printf("== ~/start_mission: %d '%s' | %s\n", res->success, res->message.c_str(), stat().c_str());
  check(res->success && stat().find("mission 1/3") != std::string::npos, "start_mission failed");

  std::printf("%s (%d failures)\n", fails ? "MISSION SIM FAILED" : "MISSION SIM OK", fails);
  return fails;
}
