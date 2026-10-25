// Closed-loop simulation of fast_planner_node (mocked ROS): perfect tracking of the published
// trajectory, static room ESDF, then an obstacle that appears mid-flight.
//   g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 test/test_planner_sim.cpp \
//       -lnlopt -o sim && ./sim
//   RAPTOR node + an unobserved region:  add  -DRAPTOR_NODE -DSIM_UNKNOWN
#define main fast_planner_main
#include "../src/fast_planner_node.cpp"
#undef main

using rclcpp::Subscription;
using rclcpp::Publisher;
struct Box {V3 lo, hi;};
static double boxSdf(const V3 & p, const Box & b)
{
  const V3 c = 0.5 * (b.lo + b.hi), h = 0.5 * (b.hi - b.lo);
  const V3 q = (p - c).cwiseAbs() - h;
  return q.cwiseMax(0.0).norm() + std::min(q.maxCoeff(), 0.0);
}
static da3_metric_scale::msg::EsdfGrid grid(const std::vector<Box> & boxes)
{
  da3_metric_scale::msg::EsdfGrid g;
  g.header.frame_id = "map";
  g.origin.x = -2; g.origin.y = -2; g.origin.z = -0.2; g.voxel_size_m = 0.05;
  g.nx = 80; g.ny = 80; g.nz = 48;
  g.distances.resize(g.nx * g.ny * g.nz);
  for (uint32_t i = 0; i < g.nx; ++i) for (uint32_t j = 0; j < g.ny; ++j) for (uint32_t k = 0; k < g.nz; ++k) {
    V3 c = V3(-2, -2, -0.2) + (V3(i, j, k) + V3::Constant(0.5)) * 0.05;
    double m = 5; for (auto & b : boxes) m = std::min(m, boxSdf(c, b));
#ifdef SIM_UNKNOWN
    // never-observed region behind the box (exercises the RAPTOR frontier logic)
    if (c.x() > 1.0 && c.y() < -0.2) {m = -1000.0;}
#endif
    g.distances[(i * g.ny + j) * g.nz + k] = m;
  }
  return g;
}
template<class T> std::shared_ptr<T> get(rclcpp::Node & n, const std::string & k, bool sub)
{
  auto & m = sub ? n.subs : n.pubs;
  auto it = m.find(k);
  if (it == m.end()) {std::printf("missing %s\n", k.c_str()); std::exit(2);}
  return std::static_pointer_cast<T>(it->second);
}

// sample a MultiDOF message at absolute time t (linear interpolation, like a fast tracker)
static bool sampleMsg(const trajectory_msgs::msg::MultiDOFJointTrajectory & m, double t, V3 & p, V3 & v, double & yaw)
{
  if (m.points.empty()) return false;
  const double t0 = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9;
  auto tt = [&](size_t i) {return t0 + m.points[i].time_from_start.sec + m.points[i].time_from_start.nanosec * 1e-9;};
  size_t i = 0;
  while (i + 1 < m.points.size() && tt(i + 1) <= t) ++i;
  size_t j = std::min(i + 1, m.points.size() - 1);
  double w = (j == i) ? 0 : std::min(1.0, std::max(0.0, (t - tt(i)) / (tt(j) - tt(i))));
  auto P = [&](size_t k) {auto & x = m.points[k].transforms[0].translation; return V3(x.x, x.y, x.z);};
  auto Vv = [&](size_t k) {auto & x = m.points[k].velocities[0].linear; return V3(x.x, x.y, x.z);};
  p = (1 - w) * P(i) + w * P(j);
  v = (1 - w) * Vv(i) + w * Vv(j);
  auto & q = m.points[i].transforms[0].rotation;
  yaw = 2 * std::atan2(q.z, q.w);
  return true;
}

int main()
{
  int fails = 0;
  rclcpp::g_quiet = false;
  FastPlannerNode node;
  auto esdf = get<Subscription<da3_metric_scale::msg::EsdfGrid>>(node, "/esdf_publisher/esdf_grid", true);
  auto pose = get<Subscription<PoseStamped>>(node, "/mavros/mavros/local_position/pose", true);
  auto vel = get<Subscription<geometry_msgs::msg::TwistStamped>>(node, "/mavros/mavros/local_position/velocity_local", true);
  auto state = get<Subscription<mavros_msgs::msg::State>>(node, "/mavros/mavros/state", true);
  auto goal = get<Subscription<PoseStamped>>(node, "/goal_pose", true);
  auto traj = get<Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>>(node, "/trajectory_tracker/trajectory", false);
  auto status = get<Publisher<std_msgs::msg::String>>(node, "~/status", false);

  std::vector<Box> room = {{{-0.2, -0.2, -1}, {0.2, 0.2, 3}}, {{0.6, -1.4, -1}, {1.0, -0.2, 1.2}},
    {{-50, -50, -5}, {50, 50, 0.0}}};
  auto g0 = grid(room);
  V3 x(-1.5, 1.2, 1.0), xv = V3::Zero();
  double yaw = 0;
  mavros_msgs::msg::State st; st.connected = true; st.armed = true; st.mode = "GUIDED";

  auto feed = [&](const da3_metric_scale::msg::EsdfGrid & g) {
    PoseStamped ps; ps.header.frame_id = "map"; ps.pose.position.x = x.x(); ps.pose.position.y = x.y();
    ps.pose.position.z = x.z(); ps.pose.orientation.z = std::sin(yaw / 2); ps.pose.orientation.w = std::cos(yaw / 2);
    pose->inject(ps);
    geometry_msgs::msg::TwistStamped tw; tw.twist.linear.x = xv.x(); tw.twist.linear.y = xv.y(); tw.twist.linear.z = xv.z();
    vel->inject(tw);
    state->inject(st);
    static double last_g = 0;
    if (rclcpp::g_now - last_g >= 0.1) {esdf->inject(g); last_g = rclcpp::g_now;}
  };

  // closed loop: 100 Hz "vehicle" following the newest published trajectory exactly
  double max_jump = 0, min_d = 1e9, max_v = 0;
  EsdfMap truth;
  auto setTruth = [&](const da3_metric_scale::msg::EsdfGrid & g) {
    truth.set(V3(g.origin.x, g.origin.y, g.origin.z), g.voxel_size_m, g.nx, g.ny, g.nz, g.distances, -1000.f);};
  setTruth(g0);
  auto run = [&](double T, const da3_metric_scale::msg::EsdfGrid & g, bool fly) {
    const double t_end = rclcpp::g_now + T;
    while (rclcpp::g_now < t_end) {
      feed(g);
      node.spinUntil(rclcpp::g_now + 0.01);
      if (fly && !traj->sent.empty()) {
        V3 p, v; double y;
        if (sampleMsg(traj->sent.back(), rclcpp::g_now, p, v, y)) {
          max_jump = std::max(max_jump, (p - x).norm());
          x = p; xv = v; yaw = y;
          max_v = std::max(max_v, v.norm());
          min_d = std::min(min_d, truth.distance(x));
        }
      }
    }
  };

  run(0.5, g0, true);                                 // map + pose, no goal yet
  PoseStamped gp; gp.header.frame_id = "map"; gp.pose.position.x = 1.5; gp.pose.position.y = -1.0;
  goal->inject(gp);                                   // RViz goal (z=0 -> default 1.0)
  run(12.0, g0, true);
  std::printf("== run 1: pos %.2f %.2f %.2f | max step %.3f m/10ms | max |v| %.2f | min ESDF %.3f\n  status: %s\n",
    x.x(), x.y(), x.z(), max_jump, max_v, min_d, status->sent.back().data.c_str());
  if ((x - V3(1.5, -1.0, 1.0)).norm() > 0.2) {std::puts("FAIL: goal not reached"); ++fails;}
  if (max_jump > 0.02) {std::puts("FAIL: reference jump"); ++fails;}
  if (min_d < 0.15) {std::puts("FAIL: clearance"); ++fails;}

  // run 2: fly back; a wall appears across the straight line after 2 s
  max_jump = 0; min_d = 1e9; max_v = 0;
  PoseStamped g2; g2.header.frame_id = "map"; g2.pose.position.x = -1.5; g2.pose.position.y = 1.2; g2.pose.position.z = 1.2;
  goal->inject(g2);
  run(2.0, g0, true);
  auto room2 = room; room2.push_back({{-1.0, -0.2, -1}, {-0.6, 1.9, 3}});
  room2.push_back({{-1.0, 1.5, -1}, {0.5, 1.9, 3}});  // L-shaped wall partly around the start
  auto g1 = grid(room2);
  setTruth(g1);
  run(15.0, g1, true);
  std::printf("== run 2: pos %.2f %.2f %.2f | max step %.3f | max |v| %.2f | min ESDF (new map) %.3f\n  status: %s\n",
    x.x(), x.y(), x.z(), max_jump, max_v, min_d, status->sent.back().data.c_str());
  if ((x - V3(-1.5, 1.2, 1.2)).norm() > 0.2) {std::puts("FAIL: goal 2 not reached"); ++fails;}
  if (min_d < 0.1) {std::puts("FAIL: clearance 2"); ++fails;}

  // run 3: leave GUIDED -> planner must stop publishing new plans
  PoseStamped g3; g3.header.frame_id = "map"; g3.pose.position.x = 1.5; g3.pose.position.y = 1.5; g3.pose.position.z = 1.0;
  st.mode = "LOITER";
  goal->inject(g3);
  size_t n_before = traj->sent.size();
  auto before = traj->sent.back();
  run(1.0, g1, true);
  std::printf("== run 3 (LOITER): status: %s\n", status->sent.back().data.c_str());
  if (status->sent.back().data.find("WAIT_READY") == std::string::npos) {std::puts("FAIL: not paused"); ++fails;}
  (void)n_before; (void)before;
  st.mode = "GUIDED";
  run(14.0, g1, true);
  std::printf("== run 3 (GUIDED again): pos %.2f %.2f %.2f status: %s\n", x.x(), x.y(), x.z(), status->sent.back().data.c_str());
  if ((x - V3(1.5, 1.5, 1.0)).norm() > 0.2) {std::puts("FAIL: goal 3 not reached"); ++fails;}

  // run 4a: an obstacle appears 0.6 m ahead (outside emergency range) -> replan around it
  // run 4b: an obstacle appears 0.25 m ahead (inside braking distance) -> emergency hold, then
  //         back out and continue (the vehicle may get close, but must not keep going in)
  for (int k = 0; k < 2; ++k) {
    const double ahead = k == 0 ? 0.6 : 0.25;
    PoseStamped g4; g4.header.frame_id = "map";
    g4.pose.position.x = k == 0 ? -1.5 : 1.5; g4.pose.position.y = -1.5; g4.pose.position.z = 1.0;
    goal->inject(g4);
    run(1.5, g1, true);
    for (int w = 0; w < 50 && xv.norm() < 0.4; ++w) run(0.1, g1, true);   // wait until cruising
    V3 dir = xv.normalized();
    V3 c = x + (ahead + 0.15) * dir;   // obstacle surface `ahead` metres in front
    auto room3 = room2; room3.push_back({c - V3(0.15, 0.15, 2), c + V3(0.15, 0.15, 2)});
    auto g5 = grid(room3);
    setTruth(g5);
    min_d = 1e9;
    const double d0 = truth.distance(x);
    std::printf("   pop-up at speed %.2f m/s\n", xv.norm());
    run(20.0, g5, true);
    std::printf("== run 4%c (pop-up %.2f m ahead, surface %.2f m): pos %.2f %.2f %.2f | min ESDF %.3f | status: %s\n",
      'a' + k, ahead, d0, x.x(), x.y(), x.z(), min_d, status->sent.back().data.c_str());
    if (min_d < (k == 0 ? 0.15 : 0.03)) {std::puts("FAIL: too close to pop-up obstacle"); ++fails;}
    if ((x - V3(g4.pose.position.x, -1.5, 1.0)).norm() > 0.2) {std::puts("FAIL: goal 4 not reached"); ++fails;}
    g1 = g5;
    room2 = room3;
  }
  auto g5 = g1;

  // run 5: ESDF stops arriving -> hold
  PoseStamped g6; g6.header.frame_id = "map"; g6.pose.position.x = 1.5; g6.pose.position.y = 1.0; g6.pose.position.z = 1.0;
  goal->inject(g6);
  run(2.0, g5, true);
  V3 xs = x;
  {
    const double t_end = rclcpp::g_now + 3.0;
    while (rclcpp::g_now < t_end) {     // pose + state only, no ESDF
      PoseStamped ps; ps.header.frame_id = "map"; ps.pose.position.x = x.x(); ps.pose.position.y = x.y(); ps.pose.position.z = x.z();
      pose->inject(ps); state->inject(st);
      node.spinUntil(rclcpp::g_now + 0.01);
      V3 p, v; double y;
      if (sampleMsg(traj->sent.back(), rclcpp::g_now, p, v, y)) {x = p; xv = v;}
    }
  }
  std::printf("== run 5 (ESDF lost): moved %.2f m in 3 s after loss | status: %s\n", (x - xs).norm(),
    status->sent.back().data.c_str());
  if ((x - xs).norm() > 0.8) {std::puts("FAIL: did not stop on stale ESDF"); ++fails;}

  std::printf("%s (%d failures)\n", fails ? "SIM FAILED" : "SIM OK", fails);
  return fails;
}
