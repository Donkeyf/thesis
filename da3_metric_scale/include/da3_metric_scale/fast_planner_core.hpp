// fast_planner_core.hpp - ROS-free Fast-Planner pipeline (Zhou et al., RA-L 2019):
//   1) kinodynamic hybrid A* front end (kino_astar.hpp) on the ESDF   (Sec. III)
//   2) uniform cubic B-spline fit + NLopt gradient optimisation         (Sec. IV)
//   3) non-uniform B-spline iterative time adjustment                   (Sec. V)
//   4) dense collision / feasibility check of the final curve
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

#include "da3_metric_scale/bspline_optimizer.hpp"
#include "da3_metric_scale/esdf_map.hpp"
#include "da3_metric_scale/kino_astar.hpp"
#include "da3_metric_scale/non_uniform_bspline.hpp"

namespace da3_metric_scale
{

struct PlannerParams
{
  kino::Params astar;
  OptParams opt;
  double clearance = 0.3;        // A* node/primitive is free if ESDF > clearance [m]
  double min_clearance = 0.1;    // never relax the A* clearance below this (start near obstacle)
  double check_clearance = 0.15; // final spline must keep ESDF > this [m]
  double check_dt = 0.02;        // final spline sampling [s]
  double ctrl_pt_dist = 0.3;     // control point spacing ~ ctrl_pt_dist / max_vel seconds
  double time_alpha = 1.1;       // alpha in Algorithm 2 (max stretch per iteration)
  int max_time_iterations = 30;
  double feasibility_tolerance = 0.05;  // accept sampled |v|,|a| up to (1+tol) * limit
  bool optimize = true;          // false = A* + fit only (debugging)
  bool use_fence = false;
  Eigen::Vector3d fence_min{-1e9, -1e9, -1e9};
  Eigen::Vector3d fence_max{1e9, 1e9, 1e9};
};

struct PlanResult
{
  bool ok = false;
  std::string message;
  kino::Status status = kino::Status::NO_PATH;
  NonUniformBspline pos, vel, acc;
  std::vector<kino::State> astar;       // front-end path
  Eigen::MatrixXd ctrl_initial;          // fitted control points before optimisation
  OptResult opt;
  double t_astar_ms = 0, t_opt_ms = 0, t_time_ms = 0;
  int expanded = 0;
  double duration = 0, max_vel = 0, max_acc = 0, min_dist = 0;
  bool used_fallback = false;            // optimiser output collided -> fitted A* path used
  bool at_goal = false;                  // start already within goal tolerance
};

class FastPlannerCore
{
public:
  explicit FastPlannerCore(const PlannerParams & p) : P_(p) {}
  PlannerParams & params() {return P_;}

  bool insideFence(const Eigen::Vector3d & x, double tol = 0.0) const
  {
    if (!P_.use_fence) {return true;}
    return (x.array() >= P_.fence_min.array() - tol).all() &&
           (x.array() <= P_.fence_max.array() + tol).all();
  }

  PlanResult plan(
    const EsdfMap & map, const Eigen::Vector3d & p0, const Eigen::Vector3d & v0,
    const Eigen::Vector3d & a0, const Eigen::Vector3d & goal,
    const Eigen::Vector3d & goal_vel = Eigen::Vector3d::Zero()) const
  {
    using clk = std::chrono::steady_clock;
    if (P_.use_fence && !insideFence(p0)) {
      // e.g. just after take-off below fence_min.z: grow the fence to contain the start
      PlannerParams q = P_;
      q.fence_min = q.fence_min.cwiseMin(p0 - V3c(0.1));
      q.fence_max = q.fence_max.cwiseMax(p0 + V3c(0.1));
      return FastPlannerCore(q).plan(map, p0, v0, a0, goal, goal_vel);
    }
    PlanResult R;
    if ((p0 - goal).norm() < P_.astar.goal_tol && v0.norm() < 0.1) {
      R.at_goal = true;
      R.message = "already at goal";
      return R;
    }

    // ---------------------------------------------------------------- 1) kinodynamic A*
    // clearance: relaxed near obstacles (the optimised curve may pass at check_clearance, so a
    // replan can start closer than `clearance`), and retried once if A* is stuck at the start
    const double d_start = map.distance(p0);
    double clr = std::min(P_.clearance, std::max(P_.min_clearance, 0.8 * d_start));
    kino::Params ap = P_.astar;
    // v0 may slightly exceed the limit after a replan; A* must still be able to start
    ap.max_vel = std::max(ap.max_vel, v0.cwiseAbs().maxCoeff() + 1e-3);
    kino::Result ar;
    auto t0 = clk::now();
    // escape zone: if the start is closer than min_clearance (e.g. after an emergency stop),
    // points near the start only need 90 % of the start clearance so A* can back out
    const double escape_r = 2.0 * P_.clearance;
    const double escape_clr = 0.9 * std::max(0.0, d_start);
    for (int attempt = 0; attempt < 2; ++attempt) {
      auto is_free = [&](const Eigen::Vector3d & x) {
          if (!insideFence(x)) {return false;}
          const double d = map.distance(x);
          if (d > clr) {return true;}
          return d_start < clr && (x - p0).norm() < escape_r && d > escape_clr;
        };
      kino::KinodynamicAStar astar(ap, is_free);
      ar = astar.search(p0, v0, goal, goal_vel);
      R.expanded += ar.expanded;
      const double relaxed = std::max(P_.min_clearance, P_.check_clearance);
      if (ar.status != kino::Status::NO_PATH || ar.expanded > 50 || clr <= relaxed + 1e-6) {break;}
      clr = relaxed;
    }
    R.t_astar_ms = ms(t0);
    R.status = ar.status;
    R.astar = ar.traj;
    if (ar.status == kino::Status::NO_PATH || ar.traj.size() < 2) {
      R.message = "kinodynamic A*: no path (" + std::to_string(R.expanded) + " nodes, clearance " +
        std::to_string(clr).substr(0, 4) + " m)";
      return R;
    }

    // ---------------------------------------------------------------- 2) B-spline fit
    const double T = ar.traj.back().t;
    double ts = P_.ctrl_pt_dist / std::max(0.1, P_.opt.max_vel);
    int K = std::max(4, static_cast<int>(std::ceil(T / ts)));
    ts = T / K;
    std::vector<Eigen::Vector3d> samples(K + 1);
    for (int k = 0; k <= K; ++k) {samples[k] = sampleAstar(ar.traj, k * ts);}
    samples.front() = p0;
    const Eigen::Vector3d v1 = ar.status == kino::Status::REACH_GOAL ? goal_vel : ar.traj.back().vel;
    Eigen::MatrixXd ctrl;
    if (!fitUniformCubic(samples, ts, v0, a0, v1, Eigen::Vector3d::Zero(), ctrl)) {
      R.message = "B-spline fit failed";
      return R;
    }
    R.ctrl_initial = ctrl;

    // ---------------------------------------------------------------- 3) NLopt optimisation
    t0 = clk::now();
    Eigen::MatrixXd ctrl_opt = ctrl;
    if (P_.optimize) {
      OptParams op = P_.opt;
      op.use_fence = P_.use_fence;
      op.fence_min = P_.fence_min;
      op.fence_max = P_.fence_max;
      BsplineOptimizer opt(op, [&map](const Eigen::Vector3d & x, Eigen::Vector3d & g) {
          return map.distance(x, &g);
        });
      R.opt = opt.optimize(ctrl_opt, ts);
    }
    R.t_opt_ms = ms(t0);

    // ---------------------------------------------------------------- 4) time adjustment + checks
    t0 = clk::now();
    NonUniformBspline best;
    std::string why;
    if (finalize(map, ctrl_opt, ts, best, R, why)) {
      R.pos = best;
    } else if (P_.optimize && finalize(map, ctrl, ts, best, R, why)) {
      R.pos = best;                // optimised curve unsafe -> plain fit of the A* path
      R.used_fallback = true;
    } else {
      R.t_time_ms = ms(t0);
      R.message = "final trajectory unsafe: " + why;
      return R;
    }
    R.t_time_ms = ms(t0);
    R.vel = R.pos.derivative();
    R.acc = R.vel.derivative();
    R.duration = R.pos.duration();
    R.ok = true;
    R.message = std::string(R.status == kino::Status::REACH_GOAL ? "to goal" : "to horizon") +
      (R.used_fallback ? " (optimiser output rejected, A* fit used)" : "");
    return R;
  }

  // dense check of a spline against the map; returns first unsafe time (or -1)
  double firstCollision(
    const EsdfMap & map, const NonUniformBspline & pos, double t_from, double t_to,
    double clearance, double * min_d = nullptr, bool check_fence = true) const
  {
    double md = 1e9;
    for (double t = t_from; t <= t_to + 1e-9; t += P_.check_dt) {
      const Eigen::Vector3d x = pos.at(t);
      const double d = map.distance(x);
      md = std::min(md, d);
      if (d <= clearance || (check_fence && !insideFence(x, 0.05))) {
        if (min_d) {*min_d = md;}
        return t;
      }
    }
    if (min_d) {*min_d = md;}
    return -1.0;
  }

private:
  static Eigen::Vector3d V3c(double v) {return Eigen::Vector3d::Constant(v);}

  static double ms(std::chrono::steady_clock::time_point t0)
  {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }

  // A* samples hold the constant input applied from each sample on -> exact propagation
  static Eigen::Vector3d sampleAstar(const std::vector<kino::State> & tr, double t)
  {
    if (t <= tr.front().t) {return tr.front().pos;}
    if (t >= tr.back().t) {return tr.back().pos;}
    auto it = std::upper_bound(tr.begin(), tr.end(), t,
        [](double tv, const kino::State & s) {return tv < s.t;});
    const kino::State & a = *(it - 1);
    const kino::State & b = *it;
    const double dt = t - a.t;
    const Eigen::Vector3d p = a.pos + a.vel * dt + 0.5 * a.acc * dt * dt;
    // the analytic tail is sampled more coarsely than its jerk; blend toward the next sample
    const double h = b.t - a.t;
    const Eigen::Vector3d pe = a.pos + a.vel * h + 0.5 * a.acc * h * h;
    return p + (b.pos - pe) * (dt / h);
  }

  bool finalize(
    const EsdfMap & map, const Eigen::MatrixXd & ctrl, double ts, NonUniformBspline & out,
    PlanResult & R, std::string & why) const
  {
    NonUniformBspline s(ctrl, 3, ts);
    const double vmax = P_.opt.max_vel, amax = P_.opt.max_acc;
    const int first_free_knot = 6;   // knots u_0..u_5 shape the start state of a cubic
    for (int it = 0; it < P_.max_time_iterations; ++it) {
      if (s.adjustTime(vmax, amax, P_.time_alpha, first_free_knot)) {break;}
    }
    // control points are conservative; verify on the curve, then stretch uniformly if needed
    double vm = 0, am = 0;
    maxDerivatives(s, vm, am);
    const double tol = 1.0 + P_.feasibility_tolerance;
    if (vm > vmax * tol || am > amax * tol) {
      const double ratio = std::max({1.0, vm / vmax, std::sqrt(am / amax)}) + 1e-3;
      s.lengthen(ratio, first_free_knot);
      maxDerivatives(s, vm, am);
    }
    double md = 0;
    const double tc = firstCollision(map, s, 0.0, s.duration(), P_.check_clearance, &md);
    R.max_vel = vm;
    R.max_acc = am;
    R.min_dist = md;
    if (tc >= 0) {
      char buf[160];
      std::snprintf(buf, sizeof(buf), "collision/fence at t=%.2f s (ESDF %.2f m <= %.2f)", tc,
        map.distance(s.at(tc)), P_.check_clearance);
      why = buf;
      return false;
    }
    out = s;
    return true;
  }

  void maxDerivatives(const NonUniformBspline & s, double & vm, double & am) const
  {
    const NonUniformBspline v = s.derivative(), a = v.derivative();
    vm = am = 0;
    // skip the first 0.3 s: that piece is inherited from the previous trajectory / vehicle state
    for (double t = 0.3; t <= s.duration(); t += P_.check_dt) {
      vm = std::max(vm, v.at(t).cwiseAbs().maxCoeff());
      am = std::max(am, a.at(t).cwiseAbs().maxCoeff());
    }
  }

  PlannerParams P_;
};

}  // namespace da3_metric_scale
