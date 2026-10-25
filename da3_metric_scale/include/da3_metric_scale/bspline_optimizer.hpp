// bspline_optimizer.hpp - gradient-based B-spline optimisation (Zhou et al. 2019, Sec. IV), NLopt.
//
//   f_total = lambda_s * f_s + lambda_c * f_c + lambda_f * (f_v + f_a)          (Eq. 9)
//   f_s  elastic band, sum ||(Q_{i+1} - Q_i) - (Q_i - Q_{i-1})||^2                (Eq. 10)
//   f_c  sum F_c(d(Q_i)),  F_c = (d - d_thr)^2 if d <= d_thr else 0              (Eq. 11, 12)
//   f_v  sum over axes F_v(V_i),  V_i = (Q_{i+1} - Q_i) / dt                       (Eq. 13, 15)
//   f_a  sum over axes F_a(A_i),  A_i = (V_{i+1} - V_i) / dt                       (Eq. 14)
//        F(x) = (x^2 - x_max^2)^2 if x^2 > x_max^2 else 0
// Decision variables: Q_{p_b} .. Q_{N-p_b} (the first and last p_b = 3 control points are
// fixed, they encode the boundary states). d(.) is the trilinear ESDF; an optional box
// "fence" adds the same penalty for leaving the flight volume. All gradients are analytic,
// solved with NLopt (default LD_LBFGS).
#pragma once

#include <Eigen/Dense>
#include <nlopt.hpp>

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace da3_metric_scale
{

struct OptParams
{
  double lambda_smooth = 10.0;     // lambda_1
  double lambda_collision = 0.8;   // lambda_2
  double lambda_feasibility = 0.01;// lambda_3
  double dist_threshold = 0.4;     // d_thr [m]
  double max_vel = 1.0;            // per axis [m/s]
  double max_acc = 1.0;            // per axis [m/s^2]
  int max_iterations = 200;
  double max_time = 0.03;          // s
  std::string algorithm = "LD_LBFGS";  // or LD_MMA, LD_TNEWTON_PRECOND_RESTART, ...
  bool use_fence = false;
  Eigen::Vector3d fence_min{-1e9, -1e9, -1e9};
  Eigen::Vector3d fence_max{1e9, 1e9, 1e9};
};

// RAPTOR (Zhou et al., T-RO 2021) Sec. VI-B2 view + safety constraints at time t_s, Eqs. 7-14:
//   d_v  = (p(t_s) - p_f) - ((p(t_s) - p_f) . v_c) v_c                   (8)
//   f_dv  = F(delta_v, ||d_v||)            -> (||d_v|| - delta_v)^2 if ||d_v|| > delta_v   (10)
//   f_dsf = F(||p(t_s) - p_f||, d_s)       -> (||p - p_f|| - d_s)^2 if ||p - p_f|| < d_s   (11)
//   f_r   = f_p2 + lambda_r (f_dv + w_r f_dsf)                                             (14)
// p(t_s) = sum_k N_k(t_s) q_k on the uniform cubic, so dp/dq_k = N_k(t_s) I (Eqs. 12, 13).
struct ViewConstraint
{
  bool enabled = false;
  double t_s = 0;                  // curve time [s] (uniform knot span dt)
  Eigen::Vector3d p_f{Eigen::Vector3d::Zero()};   // frontier intersecting point
  Eigen::Vector3d v_c{Eigen::Vector3d::UnitX()};  // critical view direction (unit, p_c - p_f)
  double delta_v = 0.1;            // [m]
  double d_s = 0.5;                // safe reaction distance [m]
  double lambda_r = 10.0;
  double w_r = 1.0;
};

// uniform cubic B-spline basis at curve time t: segment j and weights of Q_j..Q_{j+3}
inline void uniformCubicBasis(double t, double dt, int n_ctrl, int & j, double w[4])
{
  const int nseg = n_ctrl - 3;
  double s = t / dt;
  j = static_cast<int>(std::floor(s));
  j = std::max(0, std::min(nseg - 1, j));
  const double u = std::min(1.0, std::max(0.0, s - j));
  const double u2 = u * u, u3 = u2 * u;
  w[0] = (1 - 3 * u + 3 * u2 - u3) / 6.0;
  w[1] = (4 - 6 * u2 + 3 * u3) / 6.0;
  w[2] = (1 + 3 * u + 3 * u2 - 3 * u3) / 6.0;
  w[3] = u3 / 6.0;
}

struct OptResult
{
  bool ok = false;
  double cost_before = 0, cost_after = 0;
  double f_smooth = 0, f_coll = 0, f_feas = 0, f_view = 0, f_safe = 0;
  int evaluations = 0;
  std::string message;
};

class BsplineOptimizer
{
public:
  // distance callback: returns d and fills gradient
  using DistFn = std::function<double(const Eigen::Vector3d &, Eigen::Vector3d &)>;

  BsplineOptimizer(const OptParams & p, DistFn dist)
  : P_(p), dist_(std::move(dist)) {}

  void setViewConstraint(const ViewConstraint & vc) {vc_ = vc;}

  // ctrl: (N+1)x3 uniform cubic control points, dt: knot span. Optimised in place.
  OptResult optimize(Eigen::MatrixXd & ctrl, double dt)
  {
    OptResult r;
    Q_ = ctrl;
    dt_ = dt;
    const int n = static_cast<int>(ctrl.rows());
    first_ = pb_;
    last_ = n - 1 - pb_;          // inclusive
    if (last_ < first_) {r.message = "too few control points"; return r;}
    const int dim = 3 * (last_ - first_ + 1);

    std::vector<double> x(dim);
    for (int i = first_; i <= last_; ++i) {
      for (int k = 0; k < 3; ++k) {x[3 * (i - first_) + k] = ctrl(i, k);}
    }
    std::vector<double> g(dim);
    r.cost_before = cost(x.data(), g.data(), nullptr);

    best_cost_ = std::numeric_limits<double>::infinity();
    best_x_ = x;
    evals_ = 0;
    nlopt::algorithm alg = nlopt::LD_LBFGS;
    if (P_.algorithm == "LD_MMA") {alg = nlopt::LD_MMA;}
    else if (P_.algorithm == "LD_TNEWTON_PRECOND_RESTART") {alg = nlopt::LD_TNEWTON_PRECOND_RESTART;}
    else if (P_.algorithm == "LD_VAR2") {alg = nlopt::LD_VAR2;}
    else if (P_.algorithm == "LD_SLSQP") {alg = nlopt::LD_SLSQP;}
    try {
      nlopt::opt opt(alg, dim);
      opt.set_min_objective(&BsplineOptimizer::nloptCost, this);
      opt.set_maxeval(P_.max_iterations);
      opt.set_maxtime(P_.max_time);
      opt.set_xtol_rel(1e-5);
      opt.set_ftol_rel(1e-6);
      double fmin = 0;
      const nlopt::result res = opt.optimize(x, fmin);
      r.message = "nlopt result " + std::to_string(static_cast<int>(res));
    } catch (const std::exception & e) {
      // roundoff_limited / forced stop still leave a usable best point
      r.message = std::string("nlopt: ") + e.what();
    }
    x = best_x_;
    double parts[5];
    r.cost_after = cost(x.data(), g.data(), parts);
    r.f_smooth = parts[0];
    r.f_coll = parts[1];
    r.f_feas = parts[2];
    r.f_view = parts[3];
    r.f_safe = parts[4];
    r.evaluations = evals_;
    for (int i = first_; i <= last_; ++i) {
      for (int k = 0; k < 3; ++k) {ctrl(i, k) = x[3 * (i - first_) + k];}
    }
    r.ok = std::isfinite(r.cost_after) && r.cost_after <= r.cost_before + 1e-9;
    if (!r.ok) {ctrl = Q_;}
    return r;
  }

private:
  static double nloptCost(const std::vector<double> & x, std::vector<double> & grad, void * data)
  {
    auto * self = static_cast<BsplineOptimizer *>(data);
    std::vector<double> g(x.size());
    const double c = self->cost(x.data(), g.data(), nullptr);
    if (!grad.empty()) {grad = g;}
    ++self->evals_;
    if (c < self->best_cost_) {
      self->best_cost_ = c;
      self->best_x_ = x;
    }
    return c;
  }

  // cost + gradient for decision vector x (free control points)
  double cost(const double * x, double * grad, double * parts)
  {
    const int n = static_cast<int>(Q_.rows());
    std::vector<Eigen::Vector3d> q(n);
    for (int i = 0; i < n; ++i) {
      q[i] = (i >= first_ && i <= last_) ?
        Eigen::Vector3d(x[3 * (i - first_)], x[3 * (i - first_) + 1], x[3 * (i - first_) + 2]) :
        Eigen::Vector3d(Q_.row(i).transpose());
    }
    std::vector<Eigen::Vector3d> gs(n, Eigen::Vector3d::Zero()), gc = gs, gf = gs;
    double fs = 0, fc = 0, ff = 0;

    // smoothness: every second difference that touches a free point
    for (int i = 1; i + 1 < n; ++i) {
      if (i + 1 < first_ || i - 1 > last_) {continue;}
      const Eigen::Vector3d a = q[i + 1] - 2.0 * q[i] + q[i - 1];
      fs += a.squaredNorm();
      gs[i - 1] += 2.0 * a;
      gs[i] += -4.0 * a;
      gs[i + 1] += 2.0 * a;
    }

    // collision (+ fence) on the free control points
    for (int i = first_; i <= last_; ++i) {
      Eigen::Vector3d dg;
      const double d = dist_(q[i], dg);
      if (d < P_.dist_threshold) {
        const double e = d - P_.dist_threshold;
        fc += e * e;
        gc[i] += 2.0 * e * dg;
      }
      if (P_.use_fence) {
        for (int k = 0; k < 3; ++k) {
          const double lo = q[i](k) - P_.fence_min(k);
          const double hi = P_.fence_max(k) - q[i](k);
          if (lo < 0) {fc += lo * lo; gc[i](k) += 2.0 * lo;}
          if (hi < 0) {fc += hi * hi; gc[i](k) -= 2.0 * hi;}
        }
      }
    }

    // feasibility: velocity and acceleration control points (uniform dt)
    const double vm2 = P_.max_vel * P_.max_vel, am2 = P_.max_acc * P_.max_acc;
    const double idt = 1.0 / dt_, idt2 = idt * idt;
    for (int i = 0; i + 1 < n; ++i) {
      if (i + 1 < first_ || i > last_) {continue;}
      const Eigen::Vector3d v = (q[i + 1] - q[i]) * idt;
      for (int k = 0; k < 3; ++k) {
        const double e = v(k) * v(k) - vm2;
        if (e > 0) {
          ff += e * e;
          const double dv = 4.0 * e * v(k) * idt;   // d/dv (e^2) * dv/dQ
          gf[i + 1](k) += dv;
          gf[i](k) -= dv;
        }
      }
    }
    for (int i = 0; i + 2 < n; ++i) {
      if (i + 2 < first_ || i > last_) {continue;}
      const Eigen::Vector3d a = (q[i + 2] - 2.0 * q[i + 1] + q[i]) * idt2;
      for (int k = 0; k < 3; ++k) {
        const double e = a(k) * a(k) - am2;
        if (e > 0) {
          ff += e * e;
          const double da = 4.0 * e * a(k) * idt2;
          gf[i + 2](k) += da;
          gf[i + 1](k) -= 2.0 * da;
          gf[i](k) += da;
        }
      }
    }

    // RAPTOR view (f_dv) + reaction distance (f_dsf) at t_s
    double fv = 0, fsd = 0;
    std::vector<Eigen::Vector3d> gr(n, Eigen::Vector3d::Zero());
    if (vc_.enabled) {
      int j;
      double w[4];
      uniformCubicBasis(vc_.t_s, dt_, n, j, w);
      Eigen::Vector3d p = Eigen::Vector3d::Zero();
      for (int m = 0; m < 4; ++m) {p += w[m] * q[j + m];}
      const Eigen::Vector3d r = p - vc_.p_f;
      const Eigen::Matrix3d Pperp = Eigen::Matrix3d::Identity() - vc_.v_c * vc_.v_c.transpose();
      const Eigen::Vector3d dv = Pperp * r;                       // Eq. 8
      const double ndv = dv.norm();
      Eigen::Vector3d gp = Eigen::Vector3d::Zero();               // d f / d p(t_s)
      if (ndv > vc_.delta_v) {                                    // Eq. 10, 12
        fv = (ndv - vc_.delta_v) * (ndv - vc_.delta_v);
        gp += 2.0 * (ndv - vc_.delta_v) * (Pperp * dv) / ndv;
      }
      const double nr = r.norm();
      if (nr < vc_.d_s && nr > 1e-9) {                            // Eq. 11, 13
        fsd = (nr - vc_.d_s) * (nr - vc_.d_s);
        gp += vc_.w_r * 2.0 * (nr - vc_.d_s) * r / nr;
      }
      for (int m = 0; m < 4; ++m) {gr[j + m] = vc_.lambda_r * w[m] * gp;}
    }

    const double total = P_.lambda_smooth * fs + P_.lambda_collision * fc +
      P_.lambda_feasibility * ff + vc_.lambda_r * (fv + vc_.w_r * fsd);   // Eq. 14
    if (grad) {
      for (int i = first_; i <= last_; ++i) {
        const Eigen::Vector3d g = P_.lambda_smooth * gs[i] + P_.lambda_collision * gc[i] +
          P_.lambda_feasibility * gf[i] + gr[i];
        for (int k = 0; k < 3; ++k) {grad[3 * (i - first_) + k] = g(k);}
      }
    }
    if (parts) {parts[0] = fs; parts[1] = fc; parts[2] = ff; parts[3] = fv; parts[4] = fsd;}
    return total;
  }

  OptParams P_;
  DistFn dist_;
  ViewConstraint vc_;
  Eigen::MatrixXd Q_;
  double dt_{0.1};
  const int pb_ = 3;
  int first_{3}, last_{0};
  double best_cost_{0};
  std::vector<double> best_x_;
  int evals_{0};
};

}  // namespace da3_metric_scale
