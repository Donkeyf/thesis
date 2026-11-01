// yaw_planner.hpp - perception-aware yaw planning for a forward-facing camera.
//
// Follows RAPTOR Sec. VII (Zhou et al., T-RO 2021), "Yaw angle planning":
//   A. Graph search. Positions p_r,i are taken uniformly in time along the position trajectory
//      (i = 0 is the current state, its yaw fixed). At each position, J candidate yaw angles
//      xi_i,j form one layer of a directed graph; edges join adjacent layers. The sequence that
//      minimises  c = sum_i [ -g_i,j + mu (xi_i - xi_i-1)^2 ]  (Eq. 15) is found exactly by dynamic
//      programming over the layered graph (equivalent to Dijkstra on this DAG).
//      Information gain g (Eq. 16): unknown voxels inside the camera FOV that are visible (the ray
//      from the camera is not blocked by an occupied voxel), each weighted by
//      exp(-w_l D_l - w_s D_s), D_l / D_s = lateral / longitudinal distance to the trajectory.
//      Here D_s is measured forward from the position of the layer being evaluated, and space the
//      vehicle has already passed at that layer is ignored (our reading of "closer to the
//      trajectory and the current position ... have higher influence").
//      Accelerations from the paper: visibility memorised per position (each voxel ray-cast once,
//      shared by all yaw candidates), and voxel sub-sampling.
//   B. Yaw optimisation (Eq. 17): uniform cubic B-spline phi(t) minimising
//      int jerk^2 + gamma1 sum (phi(t_i) - xi_i)^2 + gamma2 (feasibility penalties on the control
//      points of phi_dot and phi_ddot, Eq. 18), first three control points fixed by the current yaw
//      and yaw rate.
//
// Extension beyond the paper (feature-aware terms, for a camera used by visual odometry):
//   node cost  + lambda_f (1 - min(1, F_i,j / F_des))^2      F = summed quality of landmarks in view
//   edge cost  + lambda_c (1 - min(1, C / C_des))^2          C = landmarks in view at BOTH ends of the
//                                                                edge (covisibility -> trackable)
//   node cost  + lambda_v (1 - cos(xi - psi_vel)) / 2        mild preference to look where it flies
//   node cost  + w_o max(0, |xi - psi_vel| - offset_max)^2   optional cap on looking away from it
// IG is normalised by its maximum over the graph so the weights are dimensionless.
//
// ROS-free (Eigen only), unit-tested in test/test_yaw_planner.cpp.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

#include "da3_metric_scale/esdf_map.hpp"

namespace da3_metric_scale
{

struct Landmark
{
  Eigen::Vector3d p{Eigen::Vector3d::Zero()};
  double score{1.0};            // quality in [0, 1]
};

struct YawPlannerParams
{
  // camera (horizontal / vertical FOV of the undistorted image, optical axis pitch: + = up)
  double hfov = 60.0 * M_PI / 180.0;
  double vfov = 47.0 * M_PI / 180.0;
  double cam_pitch = 0.0;
  double min_range = 0.3;       // m
  double max_range = 4.0;       // m (= nvblox integration range)

  // A. graph search
  double key_dt = 0.5;          // s between layers
  double horizon = 5.0;         // s of trajectory covered by the yaw plan (yaw is held after it)
  int num_yaw = 36;             // J candidate angles per layer (360 / J deg apart)
  double mu = 1.0;              // smoothness weight [1/rad^2] (Eq. 15)
  double max_yaw_rate = 45.0 * M_PI / 180.0;   // rad/s (edges faster than this are not created)
  double max_yaw_acc = 90.0 * M_PI / 180.0;    // rad/s^2 (B. feasibility)

  // information gain (Eq. 16)
  double ig_weight = 1.0;
  double ig_stride = 0.2;       // m, voxel sub-sampling for IG
  double w_lat = 1.0;           // 1/m, lateral distance weight w_l
  double w_lon = 0.3;           // 1/m, longitudinal distance weight w_s
  double occupied_dist = 0.6;   // known voxel with ESDF <= occupied_dist * voxel blocks rays
  double behind_tol = 0.3;      // m, unknown space whose closest path point is this far behind the
                                // layer position (already flown past) does not count

  // features (extension)
  double feat_weight = 0.5;     // lambda_f
  double feat_desired = 20.0;   // F_des, summed landmark quality considered "enough"
  double covis_weight = 0.5;    // lambda_c
  double covis_desired = 10.0;  // C_des
  double feat_margin = 5.0 * M_PI / 180.0;    // landmarks must be this far inside the FOV edge
  double feat_max_range = 4.0;  // m
  bool feat_occlusion = true;   // ray-cast landmarks against the ESDF

  // velocity prior (extension)
  double vel_weight = 0.2;      // lambda_v
  double vel_min_speed = 0.2;   // m/s
  double max_vel_offset = 0.0;  // rad, > 0: penalise looking further than this from the flight
                                // direction (keeps the path ahead near the image); 0 = off
  double offset_weight = 10.0;  // [1/rad^2] for the excess beyond max_vel_offset

  // B. optimisation (Eq. 17)
  double ctrl_dt = 0.25;        // s, knot span of the yaw B-spline
  double w_smooth = 1.0;        // jerk
  double gamma1 = 100.0;        // waypoints
  double gamma2 = 1000.0;       // feasibility
  double w_end = 10.0;          // soft zero yaw rate at the end of the plan (yaw is held after it)
};

struct YawPlanInput
{
  std::function<Eigen::Vector3d(double)> pos;   // position at t in [0, duration]
  std::function<Eigen::Vector3d(double)> vel;
  double duration{0};
  double yaw0{0};               // current yaw (continuous)
  double yaw_rate0{0};
  const EsdfMap * map{nullptr};
  const std::vector<Landmark> * landmarks{nullptr};
};

struct YawPlanResult
{
  bool ok{false};
  std::string message;
  // graph search
  std::vector<double> key_t, key_yaw;            // waypoints xi_i (continuous), i = 0..M
  std::vector<double> key_gain, key_feat, key_covis;   // raw IG, F, C along the chosen sequence
  int ig_voxels{0};             // visible unknown voxels counted (all layers)
  // B-spline
  std::vector<double> ctrl;
  double dt{0.25};
  double t_end{0};
  double max_rate{0};           // max |phi_dot| on the control polygon [rad/s]
  // timing [ms]
  double t_ig{0}, t_feat{0}, t_search{0}, t_opt{0};

  double yaw(double t) const {return eval(t, 0);}
  double rate(double t) const {return t >= t_end ? 0.0 : eval(t, 1);}

  double eval(double t, int deriv) const
  {
    if (ctrl.size() < 4) {return key_yaw.empty() ? 0.0 : key_yaw.back();}
    const int n = static_cast<int>(ctrl.size()) - 3;
    t = std::min(std::max(t, 0.0), t_end);
    int k = std::min(static_cast<int>(t / dt), n - 1);
    const double u = t / dt - k;
    double b[4];
    if (deriv == 0) {
      b[0] = (1 - u) * (1 - u) * (1 - u) / 6.0;
      b[1] = (3 * u * u * u - 6 * u * u + 4) / 6.0;
      b[2] = (-3 * u * u * u + 3 * u * u + 3 * u + 1) / 6.0;
      b[3] = u * u * u / 6.0;
    } else {
      b[0] = -(1 - u) * (1 - u) / 2.0 / dt;
      b[1] = (3 * u * u - 4 * u) / 2.0 / dt;
      b[2] = (-3 * u * u + 2 * u + 1) / 2.0 / dt;
      b[3] = u * u / 2.0 / dt;
    }
    return b[0] * ctrl[k] + b[1] * ctrl[k + 1] + b[2] * ctrl[k + 2] + b[3] * ctrl[k + 3];
  }
};

class YawPlanner
{
public:
  explicit YawPlanner(const YawPlannerParams & p = {})
  : P_(p) {}
  YawPlannerParams & params() {return P_;}
  const YawPlannerParams & params() const {return P_;}

  static double wrap(double a) {return std::atan2(std::sin(a), std::cos(a));}

  YawPlanResult plan(const YawPlanInput & in) const
  {
    using clk = std::chrono::steady_clock;
    auto ms = [](clk::time_point a) {
        return std::chrono::duration<double, std::milli>(clk::now() - a).count();
      };
    YawPlanResult R;
    R.dt = P_.ctrl_dt;
    if (!in.pos || in.duration <= 1e-3) {R.message = "empty trajectory"; return R;}
    const double T = std::min(in.duration, P_.horizon);
    const int M = std::max(1, static_cast<int>(std::ceil(T / P_.key_dt - 1e-6)));
    const double kdt = T / M;
    const int J = std::max(4, P_.num_yaw);
    const double dxi = 2.0 * M_PI / J;

    std::vector<Eigen::Vector3d> kp(M + 1);
    for (int i = 0; i <= M; ++i) {
      R.key_t.push_back(i * kdt);
      kp[i] = in.pos(i * kdt);
    }
    // candidate angle j of every layer: yaw0 + j * dxi (lattice aligned with the current yaw)
    auto angle = [&](int j) {return in.yaw0 + j * dxi;};

    // ---------------------------------------------------------------- information gain
    auto t0 = clk::now();
    const int B = 360;                                   // azimuth bins
    std::vector<std::vector<double>> gain(M + 1, std::vector<double>(J, 0.0));
    const bool have_map = in.map && !in.map->empty();
    if (have_map && P_.ig_weight > 0) {
      const auto traj = sampleTraj(in, 0.1);
      std::unordered_map<long long, Eigen::Vector2d> cache;   // voxel -> (D_l, arc length s)
      for (int i = 1; i <= M; ++i) {
        std::vector<double> bins(B, 0.0);
        const double s_i = arcLength(traj, R.key_t[i]);
        R.ig_voxels += igBins(*in.map, kp[i], s_i, traj, cache, bins);
        windowSums(bins, in.yaw0, dxi, J, 0.5 * P_.hfov, gain[i]);
      }
    }
    R.t_ig = ms(t0);

    // ---------------------------------------------------------------- features
    t0 = clk::now();
    // vis[i][l] = azimuth of landmark l from layer i, or NaN if not visible
    const int L = in.landmarks ? static_cast<int>(in.landmarks->size()) : 0;
    std::vector<std::vector<double>> feat(M + 1, std::vector<double>(J, 0.0));
    std::vector<std::vector<double>> vis(M + 1);
    const double half_f = 0.5 * P_.hfov - P_.feat_margin;
    const bool use_feat = L > 0 && (P_.feat_weight > 0 || P_.covis_weight > 0);
    if (use_feat) {
      for (int i = 0; i <= M; ++i) {
        vis[i].assign(L, std::numeric_limits<double>::quiet_NaN());
        for (int l = 0; l < L; ++l) {
          const auto & lm = (*in.landmarks)[l];
          const Eigen::Vector3d d = lm.p - kp[i];
          const double r = d.norm();
          if (r < P_.min_range || r > P_.feat_max_range) {continue;}
          const double el = std::atan2(d.z(), d.head<2>().norm());
          if (std::fabs(el - P_.cam_pitch) > 0.5 * P_.vfov - P_.feat_margin) {continue;}
          if (P_.feat_occlusion && have_map && blocked(*in.map, kp[i], lm.p, 2.0)) {continue;}
          vis[i][l] = std::atan2(d.y(), d.x());
        }
        if (i == 0) {continue;}
        for (int l = 0; l < L; ++l) {
          if (std::isnan(vis[i][l])) {continue;}
          for (int j = 0; j < J; ++j) {
            if (std::fabs(wrap(vis[i][l] - angle(j))) <= half_f) {
              feat[i][j] += (*in.landmarks)[l].score;
            }
          }
        }
      }
    }
    R.t_feat = ms(t0);

    // ---------------------------------------------------------------- A. graph search (DP)
    t0 = clk::now();
    double gmax = 0;
    for (int i = 1; i <= M; ++i) {for (double g : gain[i]) {gmax = std::max(gmax, g);}}
    std::vector<double> psi_v(M + 1, std::numeric_limits<double>::quiet_NaN());
    if (in.vel) {
      for (int i = 0; i <= M; ++i) {
        const Eigen::Vector3d v = in.vel(R.key_t[i]);
        if (v.head<2>().norm() > P_.vel_min_speed) {psi_v[i] = std::atan2(v.y(), v.x());}
      }
    }
    auto sat = [](double x, double des) {
        const double r = 1.0 - std::min(1.0, x / std::max(des, 1e-9));
        return r * r;
      };
    auto nodeCost = [&](int i, int j) {
        double c = 0;
        if (gmax > 1e-9) {c -= P_.ig_weight * gain[i][j] / gmax;}
        if (use_feat) {c += P_.feat_weight * sat(feat[i][j], P_.feat_desired);}
        if (!std::isnan(psi_v[i])) {
          c += P_.vel_weight * 0.5 * (1.0 - std::cos(angle(j) - psi_v[i]));
          if (P_.max_vel_offset > 0) {
            const double ex = std::fabs(wrap(angle(j) - psi_v[i])) - P_.max_vel_offset;
            if (ex > 0) {c += P_.offset_weight * ex * ex;}
          }
        }
        return c;
      };
    // covisibility of node (i-1, j1) and (i, j2)
    auto covis = [&](int i, int j1, int j2) {
        double c = 0;
        if (!use_feat) {return c;}
        for (int l = 0; l < L; ++l) {
          const double a = vis[i - 1][l], b = vis[i][l];
          if (std::isnan(a) || std::isnan(b)) {continue;}
          if (std::fabs(wrap(a - angle(j1))) <= half_f && std::fabs(wrap(b - angle(j2))) <= half_f) {
            c += (*in.landmarks)[l].score;
          }
        }
        return c;
      };
    const double max_step = P_.max_yaw_rate * kdt + 1e-9;
    const double INF = std::numeric_limits<double>::infinity();
    std::vector<std::vector<double>> cost(M + 1, std::vector<double>(J, INF));
    std::vector<std::vector<int>> parent(M + 1, std::vector<int>(J, -1));
    cost[0][0] = 0;                                   // layer 0: the current yaw only
    for (int i = 1; i <= M; ++i) {
      for (int j2 = 0; j2 < J; ++j2) {
        const double nc = nodeCost(i, j2);
        for (int j1 = 0; j1 < J; ++j1) {
          if (cost[i - 1][j1] == INF) {continue;}
          const double d = wrap(angle(j2) - angle(j1));
          if (std::fabs(d) > max_step) {continue;}
          double c = cost[i - 1][j1] + nc + P_.mu * d * d;
          if (use_feat && P_.covis_weight > 0) {
            c += P_.covis_weight * sat(covis(i, j1, j2), P_.covis_desired);
          }
          if (c < cost[i][j2]) {cost[i][j2] = c; parent[i][j2] = j1;}
        }
      }
    }
    int jb = 0;
    for (int j = 1; j < J; ++j) {if (cost[M][j] < cost[M][jb]) {jb = j;}}
    if (cost[M][jb] == INF) {R.message = "yaw graph: no feasible sequence"; return R;}
    std::vector<int> seq(M + 1);
    for (int i = M, j = jb; i >= 0; --i) {seq[i] = j; j = i > 0 ? parent[i][j] : 0;}
    R.key_yaw.assign(M + 1, in.yaw0);
    for (int i = 1; i <= M; ++i) {
      R.key_yaw[i] = R.key_yaw[i - 1] + wrap(angle(seq[i]) - angle(seq[i - 1]));
    }
    for (int i = 0; i <= M; ++i) {
      R.key_gain.push_back(i > 0 ? gain[i][seq[i]] : 0.0);
      R.key_feat.push_back(i > 0 ? feat[i][seq[i]] : 0.0);
      R.key_covis.push_back(i > 0 ? covis(i, seq[i - 1], seq[i]) : 0.0);
    }
    R.t_search = ms(t0);

    // ---------------------------------------------------------------- B. optimisation
    t0 = clk::now();
    optimize(in, R);
    R.t_opt = ms(t0);
    R.ok = true;
    char buf[160];
    std::snprintf(buf, sizeof(buf), "%d layers x %d yaws, IG voxels %d, landmarks %d", M, J,
      R.ig_voxels, L);
    R.message = buf;
    return R;
  }

private:
  struct TrajSample {Eigen::Vector3d p; double s;};

  static std::vector<TrajSample> sampleTraj(const YawPlanInput & in, double dt)
  {
    std::vector<TrajSample> out;
    double s = 0;
    Eigen::Vector3d prev = in.pos(0);
    for (double t = 0; t <= in.duration + 1e-9; t += dt) {
      const Eigen::Vector3d p = in.pos(std::min(t, in.duration));
      s += (p - prev).norm();
      prev = p;
      out.push_back({p, s});
    }
    return out;
  }

  static double arcLength(const std::vector<TrajSample> & traj, double t, double dt = 0.1)
  {
    const size_t k = std::min(traj.size() - 1, static_cast<size_t>(std::lround(t / dt)));
    return traj[k].s;
  }

  bool isBlocking(const EsdfMap & m, const Eigen::Vector3i & ix) const
  {
    if (!m.inside(ix.x(), ix.y(), ix.z())) {return false;}
    const float v = m.raw(ix.x(), ix.y(), ix.z());
    return !m.unobserved(v) && v <= P_.occupied_dist * m.voxelSize();
  }

  // true if an occupied voxel lies between a and b (the last `skip_voxels` before b are ignored,
  // e.g. the surface a landmark sits on)
  bool blocked(const EsdfMap & m, const Eigen::Vector3d & a, const Eigen::Vector3d & b,
    double skip_voxels) const
  {
    const double vs = m.voxelSize();
    const Eigen::Vector3d d = b - a;
    const double L = d.norm() - skip_voxels * vs;
    if (L <= 0) {return false;}
    const Eigen::Vector3d u = d / d.norm();
    for (double s = 0.5 * vs; s < L; s += vs) {
      if (isBlocking(m, m.index(a + s * u))) {return true;}
    }
    return false;
  }

  // Sub-sampled unknown voxels around position c, visible from c, binned by azimuth with the
  // Eq. 16 weight. Each voxel is ray-cast once and shared by all yaw candidates (memorised
  // visibility). Returns the number of visible unknown voxels.
  int igBins(const EsdfMap & m, const Eigen::Vector3d & c, double s_c,
    const std::vector<TrajSample> & traj, std::unordered_map<long long, Eigen::Vector2d> & cache,
    std::vector<double> & bins) const
  {
    const double vs = m.voxelSize();
    const int stride = std::max(1, static_cast<int>(std::lround(P_.ig_stride / vs)));
    const Eigen::Vector3i sz = m.size();
    const Eigen::Vector3i lo = m.index(c - Eigen::Vector3d::Constant(P_.max_range));
    const Eigen::Vector3i hi = m.index(c + Eigen::Vector3d::Constant(P_.max_range));
    // align the lattice to the grid so the same voxels are sampled from every position
    auto first = [stride](int a) {a = std::max(a, 0); return ((a + stride - 1) / stride) * stride;};
    const int B = static_cast<int>(bins.size());
    int count = 0;
    for (int ix = first(lo.x()); ix <= std::min(hi.x(), sz.x() - 1); ix += stride) {
      for (int iy = first(lo.y()); iy <= std::min(hi.y(), sz.y() - 1); iy += stride) {
        for (int iz = first(lo.z()); iz <= std::min(hi.z(), sz.z() - 1); iz += stride) {
          if (!m.unobserved(m.raw(ix, iy, iz))) {continue;}
          const Eigen::Vector3d p = m.center(ix, iy, iz);
          const Eigen::Vector3d d = p - c;
          const double r = d.norm();
          if (r < P_.min_range || r > P_.max_range) {continue;}
          const double el = std::atan2(d.z(), d.head<2>().norm());
          if (std::fabs(el - P_.cam_pitch) > 0.5 * P_.vfov) {continue;}
          if (blocked(m, c, p, 0.5)) {continue;}
          const long long key = (static_cast<long long>(ix) * sz.y() + iy) * sz.z() + iz;
          auto it = cache.find(key);
          if (it == cache.end()) {
            double best = std::numeric_limits<double>::infinity(), s = 0;
            for (const auto & q : traj) {
              const double dd = (q.p - p).squaredNorm();
              if (dd < best) {best = dd; s = q.s;}
            }
            it = cache.emplace(key, Eigen::Vector2d(std::sqrt(best), s)).first;
          }
          // D_s is measured forward from this layer's position: space whose closest trajectory
          // point has already been passed is irrelevant to the rest of the flight
          const double D_s = it->second.y() - s_c;
          if (D_s < -P_.behind_tol) {continue;}
          const double w = std::exp(-P_.w_lat * it->second.x() - P_.w_lon * std::max(0.0, D_s));
          const double az = std::atan2(d.y(), d.x());
          int b = static_cast<int>(std::floor((az + M_PI) / (2 * M_PI) * B));
          bins[std::min(std::max(b, 0), B - 1)] += w;
          ++count;
        }
      }
    }
    return count;
  }

  // sum of the azimuth bins within +-half of each candidate angle
  static void windowSums(const std::vector<double> & bins, double yaw0, double dxi, int J,
    double half, std::vector<double> & out)
  {
    const int B = static_cast<int>(bins.size());
    std::vector<double> pre(2 * B + 1, 0.0);           // prefix sums over two turns (wrap-around)
    for (int k = 0; k < 2 * B; ++k) {pre[k + 1] = pre[k] + bins[k % B];}
    const double bw = 2 * M_PI / B;
    for (int j = 0; j < J; ++j) {
      const double a = wrap(yaw0 + j * dxi) + M_PI;    // [0, 2 pi]
      const int b_start = static_cast<int>(std::floor((a - half) / bw));
      const int b_end = static_cast<int>(std::floor((a + half) / bw));
      const int len = std::min(std::max(b_end - b_start + 1, 1), B);
      const int start = ((b_start % B) + B) % B;
      out[j] = pre[start + len] - pre[start];
    }
  }

  // Eq. 17 as a linear least-squares problem in the free control points; the feasibility
  // penalties are piecewise quadratic and handled with an active set (re-solved until stable).
  void optimize(const YawPlanInput & in, YawPlanResult & R) const
  {
    const double Ty = R.key_t.back();
    const int n = std::max(1, static_cast<int>(std::ceil(Ty / P_.ctrl_dt - 1e-6)));
    const double dt = Ty / n;
    const int Nc = n + 3;
    R.dt = dt;
    R.t_end = Ty;
    // fixed: phi(0) = yaw0, phi_dot(0) = yaw_rate0, phi_ddot(0) = 0
    const double y0 = in.yaw0, r0 = in.yaw_rate0;
    std::vector<double> c(Nc, R.key_yaw.back());
    c[0] = y0 - r0 * dt;
    c[1] = y0;
    c[2] = y0 + r0 * dt;
    const int F = Nc - 3;                              // free control points 3..Nc-1

    struct Row {std::vector<std::pair<int, double>> a; double b; double w;};
    std::vector<Row> base;
    // smoothness: jerk on every span, (-c_k + 3c_k+1 - 3c_k+2 + c_k+3) / dt^3, integrated over dt
    for (int k = 0; k < n; ++k) {
      const double s = 1.0 / (dt * dt * dt);
      base.push_back({{{k, -s}, {k + 1, 3 * s}, {k + 2, -3 * s}, {k + 3, s}}, 0.0,
          P_.w_smooth * dt});
    }
    // waypoints
    auto basisRow = [&](double t, int deriv) {
        const int k = std::min(static_cast<int>(t / dt), n - 1);
        const double u = t / dt - k;
        double b[4];
        if (deriv == 0) {
          b[0] = (1 - u) * (1 - u) * (1 - u) / 6.0;
          b[1] = (3 * u * u * u - 6 * u * u + 4) / 6.0;
          b[2] = (-3 * u * u * u + 3 * u * u + 3 * u + 1) / 6.0;
          b[3] = u * u * u / 6.0;
        } else {
          b[0] = -(1 - u) * (1 - u) / 2.0 / dt;
          b[1] = (3 * u * u - 4 * u) / 2.0 / dt;
          b[2] = (-3 * u * u + 2 * u + 1) / 2.0 / dt;
          b[3] = u * u / 2.0 / dt;
        }
        return std::vector<std::pair<int, double>>{{k, b[0]}, {k + 1, b[1]}, {k + 2, b[2]},
          {k + 3, b[3]}};
      };
    for (size_t i = 1; i < R.key_t.size(); ++i) {
      base.push_back({basisRow(R.key_t[i], 0), R.key_yaw[i], P_.gamma1});
    }
    base.push_back({basisRow(Ty, 1), 0.0, P_.w_end});

    std::vector<Row> active;
    auto solve = [&]() {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(F, F);
        Eigen::VectorXd g = Eigen::VectorXd::Zero(F);
        auto add = [&](const Row & r) {
            // residual = sum a_k c_k - b, split fixed / free
            double rhs = r.b;
            std::vector<std::pair<int, double>> fr;
            for (const auto & e : r.a) {
              if (e.first < 3) {rhs -= e.second * c[e.first];} else {fr.push_back(e);}
            }
            for (const auto & e1 : fr) {
              g[e1.first - 3] += r.w * e1.second * rhs;
              for (const auto & e2 : fr) {H(e1.first - 3, e2.first - 3) += r.w * e1.second * e2.second;}
            }
          };
        for (const auto & r : base) {add(r);}
        for (const auto & r : active) {add(r);}
        H.diagonal().array() += 1e-9;
        const Eigen::VectorXd x = H.ldlt().solve(g);
        for (int k = 0; k < F; ++k) {c[k + 3] = x[k];}
      };
    // semi-smooth Newton on the piecewise-quadratic penalties: the active set is the set of
    // bounds violated by the current solution; re-solve until it no longer changes
    solve();
    std::vector<char> prev_set;
    for (int it = 0; it < 20; ++it) {
      active.clear();
      std::vector<char> set;
      for (int j = 0; j + 1 < Nc; ++j) {
        const double v = (c[j + 1] - c[j]) / dt;
        const bool on = j + 1 >= 3 && std::fabs(v) > P_.max_yaw_rate;
        set.push_back(on);
        if (on) {
          const double s = v > 0 ? 1.0 : -1.0;
          active.push_back({{{j, -s / dt}, {j + 1, s / dt}}, P_.max_yaw_rate, P_.gamma2});
        }
      }
      for (int j = 0; j + 2 < Nc; ++j) {
        const double a = (c[j + 2] - 2 * c[j + 1] + c[j]) / (dt * dt);
        const bool on = j + 2 >= 3 && std::fabs(a) > P_.max_yaw_acc;
        set.push_back(on);
        if (on) {
          const double k = (a > 0 ? 1.0 : -1.0) / (dt * dt);
          active.push_back({{{j, k}, {j + 1, -2 * k}, {j + 2, k}}, P_.max_yaw_acc, P_.gamma2});
        }
      }
      if (active.empty() || set == prev_set) {break;}
      prev_set = set;
      solve();
    }
    R.ctrl = c;
    R.max_rate = 0;
    for (int j = 0; j + 1 < Nc; ++j) {R.max_rate = std::max(R.max_rate, std::fabs(c[j + 1] - c[j]) / dt);}
  }

  YawPlannerParams P_;
};

}  // namespace da3_metric_scale
