// Offline test of the RAPTOR Sec. VI risk-aware refinement (no ROS).
//   g++ -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 test/test_risk_refine.cpp -lnlopt -o trr && ./trr
// Scene (paper Fig. 10): a wall on the right of the flight path ends at a corner; the goal is
// around the corner in space that has never been observed.
#include <cstdio>
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

static int fails = 0;
static void check(bool ok, const char * what) {if (!ok) {std::printf("  FAIL: %s\n", what); ++fails;}}

// first time t from which psi(p(t), p_f) >= psi_min holds until t_f
static double firstReliableView(const RiskAwareRefiner & rf, const EsdfMap & map,
  const NonUniformBspline & s, const V3 & pf, double tf, double psi_min)
{
  double last_bad = -1;
  for (double t = 0; t < tf; t += 0.02) if (rf.psi(map, s.at(t), pf) < psi_min) last_bad = t;
  return last_bad < 0 ? 0 : last_bad + 0.02;
}

// earliest t from which p_f stays viewable (psi >= psi_min - delta_v, the paper's Eq. 7 tolerance)
// until t_f; returns distance to p_f there and the speed there
static void viewReport(const RiskAwareRefiner & rf, const EsdfMap & map, const PlanResult & r,
  const RiskParams & rp, double amax, const char * name, double & margin_out)
{
  double tf = -1;
  for (double t = 0; t <= r.duration; t += 0.02) if (map.isUnknown(r.pos.at(t))) {tf = t; break;}
  margin_out = 1e9;
  if (tf < 0) {std::printf("  %-12s never enters unknown space\n", name); return;}
  const V3 pf = r.pos.at(tf);
  const double tv = firstReliableView(rf, map, r.pos, pf, tf, rp.psi_min - rp.delta_v);
  const double dv = (r.pos.at(tv) - pf).norm(), sp = r.vel.at(tv).norm();
  const double need = sp * sp / (2 * amax) + rp.R_q;
  margin_out = dv - need;
  std::printf("  %-12s T %.2f s | p_f first viewable %.2f m before it at %.2f m/s -> stopping needs %.2f m: %s"
    " | min ESDF %.2f\n", name, r.duration, dv, sp, need, dv >= need ? "OK" : "TOO LATE", r.min_dist);
}

struct Scene {EsdfMap map; Box hidden;};
// wall block x < wall_x_end, y < wall_y (the path runs along its top face), corner at
// (wall_x_end, wall_y); everything behind the corner (x > end, y < wall_y) is unobserved
static Scene makeScene(V3 lo, V3 hi, double vs, double wall_x_end, double wall_y, Box hidden)
{
  const int nx = std::lround((hi.x() - lo.x()) / vs), ny = std::lround((hi.y() - lo.y()) / vs),
    nz = std::lround((hi.z() - lo.z()) / vs);
  Box wall{{lo.x() - 1, lo.y() - 1, -1}, {wall_x_end, wall_y, 4}};
  Box floor{{-50, -50, -5}, {50, 50, 0.0}};
  std::vector<float> d(static_cast<size_t>(nx) * ny * nz);
  for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
    const V3 c = lo + (V3(i, j, k) + V3::Constant(0.5)) * vs;
    const bool unseen = c.x() > wall_x_end && c.y() < wall_y;
    const double m = std::min(boxSdf(c, wall), boxSdf(c, floor));   // hidden obstacle not in the map
    d[(static_cast<size_t>(i) * ny + j) * nz + k] = unseen ? -1000.f : static_cast<float>(m);
  }
  Scene sc;
  sc.map.unknown_distance = 1.0;
  sc.map.set(lo, vs, nx, ny, nz, d, -1000.f);
  sc.hidden = hidden;
  return sc;
}

int main()
{
  struct Case {const char * name; double vmax, amax, v0; V3 start, goal; bool room; bool expect_refine;};
  const Case cases[] = {
    {"corridor 2 m/s (paper Fig. 10/11)", 2.0, 2.0, 1.5, {-5, 0.1, 1}, {4.5, -3, 1}, false, true},
    {"corridor 2.5 m/s", 2.5, 2.0, 2.0, {-5, 0.1, 1}, {4.5, -3, 1}, false, false},
    {"room 0.8 m/s", 0.8, 1.0, 0.4, {-1.6, 0.1, 1}, {1.4, -1.4, 1}, true, false},
  };
  for (const auto & C : cases) {
    Scene sc = C.room ?
      makeScene({-2, -2, -0.2}, {2, 2, 2.2}, 0.05, 0.3, -0.25, {{0.9, -2.5, -1}, {1.1, -1.9, 3}}) :
      makeScene({-6, -4, -0.2}, {6, 4, 2.8}, 0.1, 2.0, -0.3, {{3.0, -2.0, -1}, {3.6, -0.8, 4}});
    PlannerParams P;
    P.astar.max_vel = C.vmax; P.astar.max_acc = C.amax; P.astar.lambda_heu = 5;
    if (C.room) {
      P.astar.tau = 0.6; P.astar.rho = 0.3; P.astar.resolution = 0.1; P.astar.horizon = 5; P.astar.goal_tol = 0.15;
      P.ctrl_pt_dist = 0.3; P.use_fence = true; P.fence_min = {-1.9, -1.9, 0.5}; P.fence_max = {1.9, 1.9, 1.6};
    } else {
      P.astar.tau = 0.5; P.astar.rho = 2.0; P.astar.resolution = 0.2; P.astar.horizon = 20; P.astar.goal_tol = 0.2;
      P.ctrl_pt_dist = 0.5;
    }
    P.opt.max_vel = C.vmax; P.opt.max_acc = C.amax;
    P.clearance = 0.3; P.check_clearance = 0.15;
    const V3 v0(C.v0, 0, 0);
    FastPlannerCore base(P);
    PlanResult r0 = base.plan(sc.map, C.start, v0, V3::Zero(), C.goal);
    P.risk.enable = true;
    FastPlannerCore raptor(P);
    PlanResult r1 = raptor.plan(sc.map, C.start, v0, V3::Zero(), C.goal);

    std::printf("\n== %s\n", C.name);
    check(r0.ok && r1.ok, "plans failed");
    if (!r0.ok || !r1.ok) {std::printf("  %s | %s\n", r0.message.c_str(), r1.message.c_str()); continue;}
    const auto & V = r1.refine.vis;
    std::printf("  visibility: frontier %d visible %d  t_f %.2f p_f (%.2f %.2f)  t_c %.2f p_c (%.2f %.2f)\n",
      V.has_frontier, V.visible, V.t_f, V.p_f.x(), V.p_f.y(), V.t_c, V.p_c.x(), V.p_c.y());
    std::printf("  refinement: %s (%.1f ms, used %d)\n", r1.refine.message.c_str(), r1.t_refine_ms, r1.used_refined);
    RiskAwareRefiner rf(P.risk, P.opt);
    double m0, m1;
    viewReport(rf, sc.map, r0, P.risk, C.amax, "Fast-Planner", m0);
    viewReport(rf, sc.map, r1, P.risk, C.amax, "RAPTOR", m1);
    // RAPTOR's output must meet the stopping criterion (paper metric) and never be worse
    check(m1 >= -0.05, "RAPTOR trajectory violates the stopping criterion");
    check(m1 >= m0 - 0.05, "refined trajectory sees the frontier later than the original");
    if (C.expect_refine) {check(r1.used_refined, "expected a refinement");}
    check(r1.min_dist > P.check_clearance, "refined trajectory clearance");
  }

  // gradient check of the view / safety terms (finite differences through the optimiser cost)
  {
    Eigen::MatrixXd Q(10, 3);
    for (int i = 0; i < 10; ++i) Q.row(i) = V3(0.3 * i, 0.1 * std::sin(i), 1.0).transpose();
    OptParams op; op.lambda_smooth = 0; op.lambda_collision = 0; op.lambda_feasibility = 0;
    ViewConstraint vc; vc.enabled = true; vc.t_s = 0.75; vc.p_f = V3(2.0, -0.8, 1.2);
    vc.v_c = V3(-1, 1, 0).normalized(); vc.delta_v = 0.05; vc.d_s = 3.0; vc.lambda_r = 1; vc.w_r = 0.7;
    // evaluate cost via a 0-iteration optimisation is awkward; replicate the model directly
    auto f = [&](const Eigen::MatrixXd & q) {
      int j; double w[4]; uniformCubicBasis(vc.t_s, 0.25, q.rows(), j, w);
      V3 p = V3::Zero(); for (int m = 0; m < 4; ++m) p += w[m] * q.row(j + m).transpose();
      V3 r = p - vc.p_f; Eigen::Matrix3d P = Eigen::Matrix3d::Identity() - vc.v_c * vc.v_c.transpose();
      double ndv = (P * r).norm(), fv = ndv > vc.delta_v ? std::pow(ndv - vc.delta_v, 2) : 0;
      double nr = r.norm(), fs = nr < vc.d_s ? std::pow(nr - vc.d_s, 2) : 0;
      return fv + vc.w_r * fs;
    };
    // numeric optimum direction sanity: optimiser must reduce the view cost
    Eigen::MatrixXd Q2 = Q;
    op.max_iterations = 100; op.max_time = 0.1;
    BsplineOptimizer opt(op, [](const V3 &, V3 & g) {g.setZero(); return 10.0;});
    opt.setViewConstraint(vc);
    auto res = opt.optimize(Q2, 0.25);
    std::printf("\n[view-term descent] f %.4f -> %.4f (optimiser %.4f -> %.4f)\n", f(Q), f(Q2), res.cost_before, res.cost_after);
    check(std::fabs(res.cost_before - f(Q)) < 1e-9, "cost model mismatch");
    check(f(Q2) < 0.5 * f(Q), "view terms not reduced");
  }
  std::printf("\n%s (%d failures)\n", fails ? "FAILED" : "ALL OK", fails);
  return fails;
}
