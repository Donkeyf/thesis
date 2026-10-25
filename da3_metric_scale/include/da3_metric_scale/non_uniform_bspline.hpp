// non_uniform_bspline.hpp - ROS-free clamped-domain B-spline of degree p (Zhou et al. 2019, Sec. IV-A / V).
//
// Control points Q_0..Q_N (rows of an (N+1)x3 matrix), knot vector u_0..u_M with M = N + p + 1.
// The curve is defined on [u_p, u_{M-p}]; it starts uniform (u_i = (i - p) * ts) and becomes
// non-uniform when the time-adjustment step (Algorithm 2) stretches individual knot spans.
//
// Derivative of a degree-p spline is a degree-(p-1) spline with control points
//   Q'_i = p (Q_{i+1} - Q_i) / (u_{i+p+1} - u_{i+1})      on knots u_1..u_{M-1}
// so velocity / acceleration control points (Eq. 13, 14) are just the derivative's control points
// and, by the convex-hull property, bound the curve's velocity / acceleration.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

namespace da3_metric_scale
{

class NonUniformBspline
{
public:
  NonUniformBspline() = default;

  // uniform knots u_i = (i - p) * ts, curve domain [0, (N + 1 - p) * ts]
  NonUniformBspline(const Eigen::MatrixXd & ctrl, int p, double ts)
  : ctrl_(ctrl), p_(p)
  {
    const int M = static_cast<int>(ctrl.rows()) + p;  // index of last knot
    u_.resize(M + 1);
    for (int i = 0; i <= M; ++i) {u_(i) = (i - p) * ts;}
  }

  NonUniformBspline(const Eigen::MatrixXd & ctrl, int p, const Eigen::VectorXd & knots)
  : ctrl_(ctrl), u_(knots), p_(p) {}

  bool valid() const {return ctrl_.rows() > p_ && u_.size() == ctrl_.rows() + p_ + 1;}
  int degree() const {return p_;}
  const Eigen::MatrixXd & ctrl() const {return ctrl_;}
  const Eigen::VectorXd & knots() const {return u_;}
  double tStart() const {return u_(p_);}
  double tEnd() const {return u_(u_.size() - 1 - p_);}
  double duration() const {return tEnd() - tStart();}

  // de Boor evaluation at domain parameter u (clamped to the domain)
  Eigen::Vector3d evaluate(double u) const
  {
    const int M = static_cast<int>(u_.size()) - 1;
    u = std::min(std::max(u, tStart()), tEnd());
    // span k with u_k <= u < u_{k+1}, k in [p, M-p-1]
    int k = p_;
    while (k < M - p_ - 1 && u >= u_(k + 1)) {++k;}
    std::vector<Eigen::Vector3d> d(p_ + 1);
    for (int i = 0; i <= p_; ++i) {d[i] = ctrl_.row(k - p_ + i).transpose();}
    for (int r = 1; r <= p_; ++r) {
      for (int i = p_; i >= r; --i) {
        const int j = k - p_ + i;
        const double den = u_(j + p_ + 1 - r) - u_(j);
        const double a = den > 1e-12 ? (u - u_(j)) / den : 0.0;
        d[i] = (1.0 - a) * d[i - 1] + a * d[i];
      }
    }
    return d[p_];
  }

  // evaluate at time t measured from the start of the curve
  Eigen::Vector3d at(double t) const {return evaluate(tStart() + t);}

  NonUniformBspline derivative() const
  {
    const int N = static_cast<int>(ctrl_.rows()) - 1;
    Eigen::MatrixXd d(N, 3);
    for (int i = 0; i < N; ++i) {
      const double den = u_(i + p_ + 1) - u_(i + 1);
      d.row(i) = p_ * (ctrl_.row(i + 1) - ctrl_.row(i)) / den;
    }
    return NonUniformBspline(d, p_ - 1, u_.segment(1, u_.size() - 2));
  }

  // largest per-axis |value| over the control points (convex-hull bound)
  double maxAbsCtrl() const {return ctrl_.cwiseAbs().maxCoeff();}

  // ---------------------------------------------------------------- time adjustment (Algorithm 2)
  // One pass over the velocity / acceleration control points. Every V_i with |V_i| > vmax
  // (per axis) gets its span stretched by mu_v = min(alpha, v_m / vmax); every A_i with
  // |A_i| > amax by mu_a = min(alpha, sqrt(a_m / amax)). Later knots are shifted so the rest
  // of the curve keeps its shape; control points never move, so the curve geometry (and the
  // collision clearance) is unchanged. Knots below `first_free_knot` are left alone, which
  // keeps the start state (position, velocity, acceleration at u_p) exactly as optimised.
  // Returns true if the spline was already feasible.
  bool adjustTime(double vmax, double amax, double alpha, int first_free_knot)
  {
    bool feasible = true;
    const int N = static_cast<int>(ctrl_.rows()) - 1;
    const int M = static_cast<int>(u_.size()) - 1;
    const double tol = 1e-4;

    for (int i = 0; i < N; ++i) {  // velocity control points
      const double span = u_(i + p_ + 1) - u_(i + 1);
      const Eigen::Vector3d v = p_ * (ctrl_.row(i + 1) - ctrl_.row(i)).transpose() / span;
      const double vm = v.cwiseAbs().maxCoeff();
      if (vm <= vmax + tol) {continue;}
      feasible = false;
      if (i + 2 < first_free_knot) {continue;}  // span touches the fixed start
      const double mu = std::min(alpha, vm / vmax + tol);
      const double delta = (mu - 1.0) * span;
      const double inc = delta / p_;
      for (int j = i + 2; j <= i + p_ + 1; ++j) {u_(j) += (j - i - 1) * inc;}
      for (int j = i + p_ + 2; j <= M; ++j) {u_(j) += delta;}
    }
    for (int i = 0; i + 1 < N; ++i) {  // acceleration control points
      const double s0 = u_(i + p_ + 1) - u_(i + 1);
      const double s1 = u_(i + p_ + 2) - u_(i + 2);
      const double sa = u_(i + p_ + 1) - u_(i + 2);
      const Eigen::Vector3d a = p_ * (p_ - 1) *
        ((ctrl_.row(i + 2) - ctrl_.row(i + 1)).transpose() / s1 -
        (ctrl_.row(i + 1) - ctrl_.row(i)).transpose() / s0) / sa;
      const double am = a.cwiseAbs().maxCoeff();
      if (am <= amax + tol) {continue;}
      feasible = false;
      if (i + 3 < first_free_knot) {continue;}
      const double mu = std::min(alpha, std::sqrt(am / amax) + tol);
      const double delta = (mu - 1.0) * sa;
      const double inc = delta / (p_ - 1);
      for (int j = i + 3; j <= i + p_ + 1; ++j) {u_(j) += (j - i - 2) * inc;}
      for (int j = i + p_ + 2; j <= M; ++j) {u_(j) += delta;}
    }
    return feasible;
  }

  // uniformly stretch every knot span after `first_free_knot` by `ratio`
  void lengthen(double ratio, int first_free_knot)
  {
    const int M = static_cast<int>(u_.size()) - 1;
    const int k0 = std::max(1, first_free_knot);
    Eigen::VectorXd nu = u_;
    for (int j = k0; j <= M; ++j) {nu(j) = nu(j - 1) + ratio * (u_(j) - u_(j - 1));}
    u_ = nu;
  }

  // shift the whole parameterisation so the curve starts at t = 0
  void rebase()
  {
    u_.array() -= tStart();
  }

private:
  Eigen::MatrixXd ctrl_;
  Eigen::VectorXd u_;
  int p_{3};
};

// ---------------------------------------------------------------------- fitting
// Uniform cubic B-spline through samples P_0..P_K taken ts apart (Fast-Planner
// parameterizeToBspline). For a uniform cubic, at knot k:
//   p = (Q_k + 4 Q_{k+1} + Q_{k+2}) / 6,  v = (Q_{k+2} - Q_k) / (2 ts),  a = (Q_k - 2 Q_{k+1} + Q_{k+2}) / ts^2
// The first and last three control points are solved exactly from the boundary states
// (so a replanned trajectory starts exactly on the previous one, and these are the p_b = 3
// points the optimiser keeps fixed); the K-3 interior points are a least-squares fit to the
// interior samples. Needs K >= 4 (at least one free control point).
inline void boundaryCtrl(
  const Eigen::Vector3d & p, const Eigen::Vector3d & v, const Eigen::Vector3d & a, double ts,
  Eigen::Vector3d & q0, Eigen::Vector3d & q1, Eigen::Vector3d & q2)
{
  q0 = p - ts * v + ts * ts * a / 3.0;
  q1 = p - ts * ts * a / 6.0;
  q2 = p + ts * v + ts * ts * a / 3.0;
}

inline bool fitUniformCubic(
  const std::vector<Eigen::Vector3d> & pts, double ts,
  const Eigen::Vector3d & v0, const Eigen::Vector3d & a0,
  const Eigen::Vector3d & v1, const Eigen::Vector3d & a1, Eigen::MatrixXd & ctrl)
{
  const int K = static_cast<int>(pts.size()) - 1;
  if (K < 4 || ts <= 0) {return false;}
  const int n = K + 3;           // control points Q_0..Q_{K+2}
  ctrl.resize(n, 3);
  Eigen::Vector3d q0, q1, q2;
  boundaryCtrl(pts.front(), v0, a0, ts, q0, q1, q2);
  ctrl.row(0) = q0.transpose();
  ctrl.row(1) = q1.transpose();
  ctrl.row(2) = q2.transpose();
  boundaryCtrl(pts.back(), v1, a1, ts, q0, q1, q2);
  ctrl.row(K) = q0.transpose();
  ctrl.row(K + 1) = q1.transpose();
  ctrl.row(K + 2) = q2.transpose();

  const int nf = K - 3;          // free Q_3..Q_{K-1}
  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(K - 1, nf);
  Eigen::MatrixXd b(K - 1, 3);
  for (int k = 1; k <= K - 1; ++k) {
    Eigen::RowVector3d rhs = 6.0 * pts[k].transpose();
    const double c[3] = {1.0, 4.0, 1.0};
    for (int m = 0; m < 3; ++m) {
      const int j = k + m;       // control point index
      if (j >= 3 && j <= K - 1) {
        A(k - 1, j - 3) = c[m];
      } else {
        rhs -= c[m] * ctrl.row(j);
      }
    }
    b.row(k - 1) = rhs;
  }
  const Eigen::MatrixXd x = A.colPivHouseholderQr().solve(b);
  ctrl.block(3, 0, nf, 3) = x;
  return ctrl.allFinite();
}

}  // namespace da3_metric_scale
