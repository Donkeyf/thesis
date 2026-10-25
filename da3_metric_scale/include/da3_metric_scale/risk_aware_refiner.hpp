// risk_aware_refiner.hpp - RAPTOR Sec. VI, risk-aware trajectory refinement (Zhou et al., T-RO 2021).
//
// Input: an optimised uniform cubic B-spline p_i(t) (Fast-Planner output before time adjustment).
// Algorithm 3:
//   1  p_f, t_f  <- FrontierIntersection(p_i)     first point of p_i(t) in unknown space
//   2  visible, v_c, p_c, t_c <- CriticalView     psi(p_i(t), p_f) >= psi_min for all t >= t_c   (Eq. 5)
//   3  if visible: return p_i
//   5  v_c = |p_i'(t_c)|, d_cf = |p_c - p_f|
//   6  if v_c^2 / 2a_max <= d_cf - R_q: return p_i                                              (Eq. 6)
//   8  v_s^ <- average speed of p_i on [t_0, t_f]
//   9  repeat
//  10    d_s^ = v_s^2 / 2a_max + R_q
//  11    t_s = argmin_t f_dv + w_r f_dsf   (on p_i)
//  12    p_r <- RefineTrajectory(p_i, p_f, v_c, t_s, d_s^)   minimise f_r (Eq. 14) with NLopt
//  13    v_s = |p_r'(t_s)|, d_sf = |p_r(t_s) - p_f|
//  14    v_s^ <- alpha v_s^
//  15  until v_s^2 / 2a_max <= d_sf - R_q
// Visibility psi(p, p_f) = min ESDF distance along the segment p -> p_f (Eq. 5); the ESDF only
// knows observed obstacles, exactly as in the paper.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "da3_metric_scale/bspline_optimizer.hpp"
#include "da3_metric_scale/esdf_map.hpp"
#include "da3_metric_scale/non_uniform_bspline.hpp"

namespace da3_metric_scale
{

struct RiskParams
{
  bool enable = false;
  double psi_min = 0.25;        // m, required visibility level of p_f (Eq. 5)
  double delta_v = 0.05;        // m, view constraint tolerance (Eq. 7)
  double R_q = 0.3;             // m, quadrotor size + disturbance margin (Eq. 6)
  double lambda_r = 10.0;       // weight of the refinement terms (Eq. 14)
  double w_r = 1.0;             // relative weight of f_dsf (Eq. 14)
  double alpha = 1.1;           // speed-estimate growth factor (Alg. 3 line 14)
  int max_iterations = 5;       // cap on the repeat loop
  double sample_dt = 0.02;      // s, sampling of p_i(t)
  double min_frontier_time = 0.3;  // s, frontier closer than this in time: nothing to refine
};

struct VisibilityStatus
{
  bool has_frontier = false;    // trajectory enters unknown space
  bool visible = false;         // psi >= psi_min from every point before t_f
  double t_f = 0, t_c = 0;
  Eigen::Vector3d p_f{Eigen::Vector3d::Zero()}, p_c{Eigen::Vector3d::Zero()};
  Eigen::Vector3d v_c{Eigen::Vector3d::Zero()};   // critical view direction (unit)
};

struct RefineResult
{
  bool refined = false;         // ctrl was modified
  bool safe = true;             // worst-case criterion (Eq. 6 / line 15) satisfied
  std::string message = "off";
  VisibilityStatus vis;
  double t_s = 0, d_s = 0, v_s = 0, d_sf = 0;
  Eigen::Vector3d p_s{Eigen::Vector3d::Zero()};   // p_r(t_s)
  int iterations = 0;
  OptResult opt;
};

class RiskAwareRefiner
{
public:
  RiskAwareRefiner(const RiskParams & rp, const OptParams & op)
  : R_(rp), O_(op) {}

  // ---------------------------------------------------------------- Sec. VI-A
  VisibilityStatus visibility(const EsdfMap & map, const Eigen::MatrixXd & ctrl, double ts) const
  {
    VisibilityStatus v;
    const NonUniformBspline s(ctrl, 3, ts);
    const double T = s.duration();
    // 1) FrontierIntersection
    double t = 0;
    for (; t <= T + 1e-9; t += R_.sample_dt) {
      if (map.isUnknown(s.at(t))) {break;}
    }
    if (t > T + 1e-9) {return v;}               // stays in known space
    v.has_frontier = true;
    v.t_f = t;
    v.p_f = s.at(t);
    // 2/3) CriticalView: last time before t_f with psi < psi_min
    double last_bad = -1;
    for (double tk = 0; tk < v.t_f - 1e-9; tk += R_.sample_dt) {
      if (psi(map, s.at(tk), v.p_f) < R_.psi_min) {last_bad = tk;}
    }
    if (last_bad < 0) {
      v.visible = true;
      return v;
    }
    v.t_c = std::min(v.t_f, last_bad + R_.sample_dt);
    v.p_c = s.at(v.t_c);
    const Eigen::Vector3d d = v.p_c - v.p_f;
    v.v_c = d.norm() > 1e-6 ? Eigen::Vector3d(d / d.norm()) : Eigen::Vector3d::Zero();
    return v;
  }

  // visibility level psi(p, p_f): smallest ESDF distance on the segment (Eq. 5)
  double psi(const EsdfMap & map, const Eigen::Vector3d & p, const Eigen::Vector3d & pf) const
  {
    const double L = (pf - p).norm();
    const double step = std::max(0.25 * map.voxelSize(), 0.01);
    const int n = std::max(1, static_cast<int>(std::ceil(L / step)));
    double m = 1e9;
    for (int i = 0; i <= n; ++i) {m = std::min(m, map.distance(p + (pf - p) * (double(i) / n)));}
    return m;
  }

  // ---------------------------------------------------------------- Sec. VI-B, Algorithm 3
  RefineResult refine(const EsdfMap & map, Eigen::MatrixXd & ctrl, double ts) const
  {
    RefineResult res;
    if (!R_.enable) {return res;}
    res.vis = visibility(map, ctrl, ts);
    const VisibilityStatus & V = res.vis;
    if (!V.has_frontier) {res.message = "no frontier on trajectory"; return res;}
    if (V.t_f < R_.min_frontier_time) {res.message = "frontier too close in time"; return res;}
    if (V.visible) {res.message = "frontier visible"; return res;}
    if (V.v_c.squaredNorm() < 0.5) {
      res.message = "frontier point itself poorly visible (near obstacle), not refined";
      return res;
    }

    const NonUniformBspline pi(ctrl, 3, ts);
    const NonUniformBspline vi = pi.derivative();
    const double amax = O_.max_acc;
    // lines 5-7: worst-case safety criterion at the critical view point (Eq. 6)
    const double vc = vi.at(V.t_c).norm();
    const double dcf = (V.p_c - V.p_f).norm();
    if (vc * vc / (2 * amax) <= dcf - R_.R_q) {
      res.message = "safe at critical view";
      return res;
    }
    res.safe = false;

    // line 8: average speed on [t_0, t_f]
    double len = 0;
    for (double t = R_.sample_dt; t <= V.t_f + 1e-9; t += R_.sample_dt) {
      len += (pi.at(t) - pi.at(t - R_.sample_dt)).norm();
    }
    double v_hat = len / std::max(V.t_f, 1e-3);

    Eigen::MatrixXd best = ctrl;
    double best_margin = vc * vc / (2 * amax) - (dcf - R_.R_q);   // > 0 means violated
    for (int it = 0; it < R_.max_iterations; ++it) {
      res.iterations = it + 1;
      const double ds = v_hat * v_hat / (2 * amax) + R_.R_q;           // line 10
      const double t_s = selectTs(pi, V, ds, ts);                       // line 11
      Eigen::MatrixXd q = ctrl;                                         // line 12 (refine p_i)
      ViewConstraint vcst;
      vcst.enabled = true;
      vcst.t_s = t_s;
      vcst.p_f = V.p_f;
      vcst.v_c = V.v_c;
      vcst.delta_v = R_.delta_v;
      vcst.d_s = ds;
      vcst.lambda_r = R_.lambda_r;
      vcst.w_r = R_.w_r;
      BsplineOptimizer opt(O_, [&map](const Eigen::Vector3d & x, Eigen::Vector3d & g) {
          return map.distance(x, &g);
        });
      opt.setViewConstraint(vcst);
      res.opt = opt.optimize(q, ts);
      const NonUniformBspline pr(q, 3, ts);
      const double vs = pr.derivative().at(t_s).norm();                 // line 13
      const double dsf = (pr.at(t_s) - V.p_f).norm();
      const double margin = vs * vs / (2 * amax) - (dsf - R_.R_q);
      res.t_s = t_s;
      res.d_s = ds;
      res.v_s = vs;
      res.d_sf = dsf;
      res.p_s = pr.at(t_s);
      if (margin < best_margin) {
        best_margin = margin;
        best = q;
      }
      if (margin <= 0) {                                                // line 15
        res.safe = true;
        break;
      }
      v_hat *= R_.alpha;                                                // line 14
    }
    // "refined" only if a control point actually moved (NLopt round-off is not a refinement)
    res.refined = (best - ctrl).rowwise().norm().maxCoeff() > 1e-3;
    if (!res.refined) {best = ctrl;}
    ctrl = best;
    char buf[200];
    std::snprintf(buf, sizeof(buf), "%s%s after %d it: t_f %.2f t_c %.2f t_s %.2f, v_s %.2f m/s, "
      "d_sf %.2f m (need %.2f)", res.refined ? "refined" : "already on the view ray",
      res.safe ? ", safe" : ", criterion NOT met",
      res.iterations, V.t_f, V.t_c, res.t_s, res.v_s, res.d_sf,
      res.v_s * res.v_s / (2 * amax) + R_.R_q);
    res.message = buf;
    return res;
  }

private:
  // line 11: t_s = argmin_t f_dv + w_r f_dsf on the initial trajectory, t in [ts, t_c]
  // (the first knot span is mostly fixed by the start state). Earliest minimiser on ties.
  double selectTs(
    const NonUniformBspline & pi, const VisibilityStatus & V, double ds, double ts) const
  {
    const Eigen::Matrix3d Pperp = Eigen::Matrix3d::Identity() - V.v_c * V.v_c.transpose();
    double best_t = std::max(ts, std::min(V.t_c, ts)), best_f = 1e18;
    for (double t = ts; t <= V.t_c + 1e-9; t += R_.sample_dt) {
      const Eigen::Vector3d r = pi.at(t) - V.p_f;
      const double ndv = (Pperp * r).norm();
      const double fdv = ndv > R_.delta_v ? (ndv - R_.delta_v) * (ndv - R_.delta_v) : 0.0;
      const double nr = r.norm();
      const double fds = nr < ds ? (nr - ds) * (nr - ds) : 0.0;
      const double f = fdv + R_.w_r * fds;
      if (f < best_f - 1e-12) {best_f = f; best_t = t;}
    }
    return best_t;
  }

  RiskParams R_;
  OptParams O_;
};

}  // namespace da3_metric_scale
