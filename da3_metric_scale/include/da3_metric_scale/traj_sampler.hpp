// traj_sampler.hpp - ROS-free trajectory container + sampler.
//
// Points (t, p, optional v, optional a, yaw, optional yaw_rate) are joined with
// cubic Hermite segments, so position, velocity and acceleration are continuous
// within a segment and consistent with each other. Missing vertex velocities are
// estimated by finite differences (non-uniform central differences; one-sided at
// the start; zero at the end if stop_at_end). If both endpoints of a segment carry
// accelerations they are interpolated linearly, otherwise the Hermite 2nd derivative
// is used. Yaw is unwrapped and interpolated linearly.
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace da3_metric_scale
{

struct Vec3
{
  double x{0}, y{0}, z{0};
  Vec3() = default;
  Vec3(double x_, double y_, double z_)
  : x(x_), y(y_), z(z_) {}
  Vec3 operator+(const Vec3 & o) const {return {x + o.x, y + o.y, z + o.z};}
  Vec3 operator-(const Vec3 & o) const {return {x - o.x, y - o.y, z - o.z};}
  Vec3 operator*(double s) const {return {x * s, y * s, z * s};}
  double norm() const {return std::sqrt(x * x + y * y + z * z);}
};
inline Vec3 operator*(double s, const Vec3 & v) {return v * s;}

inline double wrapPi(double a)
{
  return std::atan2(std::sin(a), std::cos(a));
}

struct TrajPoint
{
  double t{0};          // absolute time [s]
  Vec3 p, v, a;
  bool has_v{false}, has_a{false};
  double yaw{0};        // rad, ENU
  double yaw_rate{0};   // rad/s
  bool has_yaw_rate{false};
};

struct RefState
{
  Vec3 p, v, a;
  double yaw{0}, yaw_rate{0};
  bool before_start{false}, after_end{false};
};

class Trajectory
{
public:
  // Validates and completes the points. Returns false (and err) if unusable.
  bool set(std::vector<TrajPoint> pts, bool stop_at_end, std::string & err)
  {
    if (pts.empty()) {err = "empty trajectory"; return false;}
    for (size_t i = 1; i < pts.size(); ++i) {
      if (!(pts[i].t > pts[i - 1].t)) {
        err = "point times must strictly increase (index " + std::to_string(i) + ")";
        return false;
      }
    }
    // unwrap yaw
    for (size_t i = 1; i < pts.size(); ++i) {
      pts[i].yaw = pts[i - 1].yaw + wrapPi(pts[i].yaw - pts[i - 1].yaw);
    }
    const size_t n = pts.size();
    // vertex velocities
    for (size_t i = 0; i < n; ++i) {
      if (pts[i].has_v) {continue;}
      if (n == 1) {
        pts[i].v = Vec3();
      } else if (i == 0) {
        pts[i].v = (pts[1].p - pts[0].p) * (1.0 / (pts[1].t - pts[0].t));
      } else if (i == n - 1) {
        pts[i].v = stop_at_end ? Vec3() :
          (pts[i].p - pts[i - 1].p) * (1.0 / (pts[i].t - pts[i - 1].t));
      } else {
        const double h0 = pts[i].t - pts[i - 1].t, h1 = pts[i + 1].t - pts[i].t;
        const Vec3 d0 = (pts[i].p - pts[i - 1].p) * (1.0 / h0);
        const Vec3 d1 = (pts[i + 1].p - pts[i].p) * (1.0 / h1);
        // weighted (non-uniform) central difference
        pts[i].v = (d0 * h1 + d1 * h0) * (1.0 / (h0 + h1));
      }
    }
    // yaw rates
    for (size_t i = 0; i < n; ++i) {
      if (pts[i].has_yaw_rate) {continue;}
      if (n == 1) {
        pts[i].yaw_rate = 0;
      } else if (i == n - 1) {
        pts[i].yaw_rate = stop_at_end ? 0.0 :
          (pts[i].yaw - pts[i - 1].yaw) / (pts[i].t - pts[i - 1].t);
      } else {
        pts[i].yaw_rate = (pts[i + 1].yaw - pts[i].yaw) / (pts[i + 1].t - pts[i].t);
      }
    }
    pts_ = std::move(pts);
    return true;
  }

  bool empty() const {return pts_.empty();}
  size_t size() const {return pts_.size();}
  double startTime() const {return pts_.front().t;}
  double endTime() const {return pts_.back().t;}
  const TrajPoint & back() const {return pts_.back();}

  RefState sample(double t) const
  {
    RefState r;
    if (pts_.size() == 1 || t >= pts_.back().t) {
      const auto & e = pts_.back();
      r.p = e.p;
      r.yaw = e.yaw;
      r.after_end = true;          // hold the end point, no feedforward
      return r;
    }
    if (t <= pts_.front().t) {
      const auto & s = pts_.front();
      r.p = s.p;
      r.yaw = s.yaw;
      r.before_start = true;       // not started yet: hold the start point
      return r;
    }
    auto it = std::upper_bound(
      pts_.begin(), pts_.end(), t, [](double tv, const TrajPoint & q) {return tv < q.t;});
    const TrajPoint & b = *it;
    const TrajPoint & a = *(it - 1);
    const double h = b.t - a.t;
    const double s = (t - a.t) / h;
    const double s2 = s * s, s3 = s2 * s;

    const double h00 = 2 * s3 - 3 * s2 + 1, h10 = s3 - 2 * s2 + s;
    const double h01 = -2 * s3 + 3 * s2, h11 = s3 - s2;
    const double d00 = 6 * s2 - 6 * s, d10 = 3 * s2 - 4 * s + 1;
    const double d01 = -6 * s2 + 6 * s, d11 = 3 * s2 - 2 * s;
    const double dd00 = 12 * s - 6, dd10 = 6 * s - 4, dd01 = -12 * s + 6, dd11 = 6 * s - 2;

    r.p = h00 * a.p + (h10 * h) * a.v + h01 * b.p + (h11 * h) * b.v;
    r.v = (d00 * a.p + (d10 * h) * a.v + d01 * b.p + (d11 * h) * b.v) * (1.0 / h);
    if (a.has_a && b.has_a) {
      r.a = (1 - s) * a.a + s * b.a;
    } else {
      r.a = (dd00 * a.p + (dd10 * h) * a.v + dd01 * b.p + (dd11 * h) * b.v) * (1.0 / (h * h));
    }
    r.yaw = wrapPi((1 - s) * a.yaw + s * b.yaw);
    r.yaw_rate = (a.has_yaw_rate && b.has_yaw_rate) ?
      (1 - s) * a.yaw_rate + s * b.yaw_rate : (b.yaw - a.yaw) / h;
    return r;
  }

  // Give untimed points times by arc length at constant speed, starting at t0.
  static void timeByArcLength(std::vector<TrajPoint> & pts, double speed, double t0)
  {
    if (pts.empty()) {return;}
    pts[0].t = t0;
    for (size_t i = 1; i < pts.size(); ++i) {
      const double d = (pts[i].p - pts[i - 1].p).norm();
      pts[i].t = pts[i - 1].t + std::max(d / std::max(speed, 1e-3), 1e-3);
    }
  }

private:
  std::vector<TrajPoint> pts_;
};

}  // namespace da3_metric_scale
