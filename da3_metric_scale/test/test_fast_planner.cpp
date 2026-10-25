// Offline test of the Fast-Planner core on synthetic ESDFs (no ROS).
//   g++ -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 test/test_fast_planner.cpp -lnlopt_cxx -o /tmp/tfp
#include <cstdio>
#include <iostream>
#include <vector>

#include "da3_metric_scale/fast_planner_core.hpp"

using namespace da3_metric_scale;
using V3 = Eigen::Vector3d;

struct Box {V3 lo, hi;};
static double boxSdf(const V3 & p, const Box & b)
{
  const V3 c = 0.5 * (b.lo + b.hi), h = 0.5 * (b.hi - b.lo);
  const V3 q = (p - c).cwiseAbs() - h;
  return q.cwiseMax(0.0).norm() + std::min(q.maxCoeff(), 0.0);
}

static EsdfMap makeMap(const V3 & lo, const V3 & hi, double vs, const std::vector<Box> & boxes)
{
  const int nx = std::ceil((hi.x() - lo.x()) / vs), ny = std::ceil((hi.y() - lo.y()) / vs),
    nz = std::ceil((hi.z() - lo.z()) / vs);
  std::vector<float> d(static_cast<size_t>(nx) * ny * nz);
  for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
    const V3 c = lo + (V3(i, j, k) + V3::Constant(0.5)) * vs;
    double m = 5.0;
    for (const auto & b : boxes) m = std::min(m, boxSdf(c, b));
    d[(static_cast<size_t>(i) * ny + j) * nz + k] = static_cast<float>(m);
  }
  EsdfMap map;
  map.unknown_distance = 1.0;
  map.set(lo, vs, nx, ny, nz, d, -1000.f);
  return map;
}

static int report(const char * name, const FastPlannerCore & core, const EsdfMap & map,
  const PlanResult & r, const V3 & goal)
{
  std::printf("[%s] ok=%d status=%d '%s'\n  A* %.1f ms (%d nodes, %zu samples, T=%.2f) | opt %.1f ms "
    "(%d evals, cost %.3f -> %.3f, s=%.3f c=%.4f f=%.4f) | time adj %.1f ms\n",
    name, r.ok, (int)r.status, r.message.c_str(), r.t_astar_ms, r.expanded, r.astar.size(),
    r.astar.empty() ? 0 : r.astar.back().t, r.t_opt_ms, r.opt.evaluations, r.opt.cost_before,
    r.opt.cost_after, r.opt.f_smooth, r.opt.f_coll, r.opt.f_feas, r.t_time_ms);
  if (!r.ok) return 1;
  std::printf("  duration %.2f s (A* %.2f), max|v| %.2f max|a| %.2f, min ESDF %.3f, end err %.3f, "
    "ctrl pts %ld, fallback %d\n", r.duration, r.astar.back().t, r.max_vel, r.max_acc, r.min_dist,
    (r.pos.at(r.duration) - goal).norm(), (long)r.pos.ctrl().rows(), r.used_fallback);
  double md_astar = 1e9;
  for (auto & s : r.astar) md_astar = std::min(md_astar, map.distance(s.pos));
  std::printf("  min ESDF along A*: %.3f\n", md_astar);
  return 0;
}

int main()
{
  int fails = 0;
  // ---- 1) the user's two-wall scene (test_kino.cpp), 3 m/s, 15 m
  {
    std::vector<Box> b = {
      {{4.9, -50, -1}, {5.3, 1.0, 4}}, {{4.9, 2.0, -1}, {5.3, 50, 4}},
      {{9.9, -50, -1}, {10.3, -2.0, 4}}, {{9.9, -1.0, -1}, {10.3, 50, 4}},
      {{-50, -50, -5}, {50, 50, 0.0}}, {{-50, -50, 3.0}, {50, 50, 8}}};
    EsdfMap map = makeMap({-1, -5, -0.2}, {16, 5, 3.2}, 0.1, b);
    PlannerParams P;
    P.astar.max_vel = 3.0; P.astar.max_acc = 2.0; P.astar.resolution = 0.2; P.astar.horizon = 50;
    P.opt.max_vel = 3.0; P.opt.max_acc = 2.0;
    P.clearance = 0.2; P.check_clearance = 0.1; P.ctrl_pt_dist = 0.5; P.opt.dist_threshold = 0.4;
    P.opt.max_time = 0.1;
    FastPlannerCore core(P);
    V3 goal(15, 0, 1);
    auto r = core.plan(map, {0, 0, 1}, V3::Zero(), V3::Zero(), goal);
    fails += report("two walls", core, map, r, goal);
  }
  // ---- 2) room scale (4x4 m flight volume, pillar + box), 0.8 m/s, ESDF 5 cm like nvblox
  {
    std::vector<Box> b = {
      {{-0.2, -0.2, -1}, {0.2, 0.2, 3}},           // pillar in the middle
      {{0.6, -1.4, -1}, {1.0, -0.2, 1.2}},         // box
      {{-50, -50, -5}, {50, 50, 0.0}}};            // floor
    EsdfMap map = makeMap({-2, -2, -0.2}, {2, 2, 2.2}, 0.05, b);
    PlannerParams P;
    P.astar.max_vel = 0.8; P.astar.max_acc = 1.0; P.astar.resolution = 0.1; P.astar.tau = 0.4;
    P.astar.rho = 0.3; P.astar.horizon = 5.0; P.astar.goal_tol = 0.15; P.astar.lambda_heu = 2.0;
    P.opt.max_vel = 0.8; P.opt.max_acc = 1.0; P.opt.dist_threshold = 0.4; P.opt.max_time = 0.03;
    P.clearance = 0.3; P.check_clearance = 0.15; P.ctrl_pt_dist = 0.3;
    P.use_fence = true; P.fence_min = {-1.8, -1.8, 0.4}; P.fence_max = {1.8, 1.8, 1.6};
    FastPlannerCore core(P);
    V3 goal(1.5, -1.0, 1.0);
    auto r = core.plan(map, {-1.5, 1.2, 1.0}, V3::Zero(), V3::Zero(), goal);
    fails += report("room", core, map, r, goal);
    // replan from the middle of the trajectory (non-zero v, a), back to the start
    if (r.ok) {
      const double t = 0.5 * r.duration;
      V3 p = r.pos.at(t), v = r.vel.at(t), a = r.acc.at(t);
      V3 g2(-1.5, -1.5, 1.2);
      auto r2 = core.plan(map, p, v, a, g2);
      fails += report("room replan", core, map, r2, g2);
      if (r2.ok) {
        std::printf("  start continuity: dp %.2e dv %.2e da %.2e\n", (r2.pos.at(0) - p).norm(),
          (r2.vel.at(0) - v).norm(), (r2.acc.at(0) - a).norm());
        if ((r2.pos.at(0) - p).norm() > 1e-6 || (r2.vel.at(0) - v).norm() > 1e-6 ||
          (r2.acc.at(0) - a).norm() > 1e-6) {std::puts("  FAIL continuity"); ++fails;}
      }
    }
    // optimisation off for comparison
    P.optimize = false;
    FastPlannerCore core2(P);
    auto r3 = core2.plan(map, {-1.5, 1.2, 1.0}, V3::Zero(), V3::Zero(), goal);
    fails += report("room no-opt", core2, map, r3, goal);
    // goal blocked by the pillar region -> should still produce something or fail cleanly
    auto r4 = core.plan(map, {-1.5, 1.2, 1.0}, V3::Zero(), V3::Zero(), V3(0, 0, 1.0));
    std::printf("[goal in obstacle] ok=%d '%s'\n", r4.ok, r4.message.c_str());
  }
  // ---- 3) gradient check of the trilinear ESDF
  {
    std::vector<Box> b = {{{-0.3, -0.3, -0.3}, {0.3, 0.3, 0.3}}};
    EsdfMap map = makeMap({-1, -1, -1}, {1, 1, 1}, 0.05, b);
    V3 x(0.43, -0.21, 0.12), g;
    map.distance(x, &g);
    V3 gn;
    for (int k = 0; k < 3; ++k) {
      V3 e = V3::Zero(); e(k) = 1e-5;
      gn(k) = (map.distance(x + e) - map.distance(x - e)) / 2e-5;
    }
    std::printf("[esdf grad] analytic %s numeric %s\n", 
      (std::to_string(g.x()) + "," + std::to_string(g.y()) + "," + std::to_string(g.z())).c_str(),
      (std::to_string(gn.x()) + "," + std::to_string(gn.y()) + "," + std::to_string(gn.z())).c_str());
    if ((g - gn).norm() > 1e-3) {std::puts("  FAIL grad"); ++fails;}
  }
  std::printf("%s (%d failures)\n", fails ? "FAILED" : "ALL OK", fails);
  return fails;
}
