// esdf_map.hpp - ROS-free dense ESDF with trilinear distance + gradient queries.
//
// Holds one da3_metric_scale/msg/EsdfGrid (nvblox 3D ESDF around the vehicle):
//   voxel (ix,iy,iz) centre = origin + (i + 0.5) * voxel_size, value = d[(ix*ny + iy)*nz + iz]
// Distances are interpolated trilinearly between voxel centres (Zhou et al. 2019, Sec. VII-C:
// "trilinear interpolation is used to improve the accuracy of the distance and gradient").
// Unobserved voxels and points outside the grid return `unknown_distance` (or a negative
// value if `unknown_is_occupied`), so the planner's behaviour in unknown space is explicit.
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <vector>

namespace da3_metric_scale
{

class EsdfMap
{
public:
  // configuration
  double unknown_distance = 0.6;   // distance assumed for unobserved / out-of-grid space [m]
  bool unknown_is_occupied = false;

  void set(const Eigen::Vector3d & origin, double voxel, int nx, int ny, int nz,
    std::vector<float> data, float unobserved_value)
  {
    origin_ = origin;
    vs_ = voxel;
    nx_ = nx;
    ny_ = ny;
    nz_ = nz;
    d_ = std::move(data);
    unobs_ = unobserved_value;
  }

  bool empty() const {return d_.empty() || vs_ <= 0;}
  double voxelSize() const {return vs_;}
  const Eigen::Vector3d & origin() const {return origin_;}
  Eigen::Vector3i size() const {return {nx_, ny_, nz_};}

  // Raw voxel access (no unknown handling) for ray casting / information gain
  bool inside(int ix, int iy, int iz) const
  {
    return ix >= 0 && iy >= 0 && iz >= 0 && ix < nx_ && iy < ny_ && iz < nz_;
  }
  float raw(int ix, int iy, int iz) const
  {
    return d_[(static_cast<size_t>(ix) * ny_ + iy) * nz_ + iz];
  }
  bool unobserved(float v) const {return std::fabs(v - unobs_) < 1e-2f;}
  Eigen::Vector3i index(const Eigen::Vector3d & p) const
  {
    const Eigen::Vector3d q = (p - origin_) / vs_;
    return {static_cast<int>(std::floor(q.x())), static_cast<int>(std::floor(q.y())),
      static_cast<int>(std::floor(q.z()))};
  }
  Eigen::Vector3d center(int ix, int iy, int iz) const
  {
    return origin_ + vs_ * Eigen::Vector3d(ix + 0.5, iy + 0.5, iz + 0.5);
  }

  // Value at voxel index (unknown handling applied)
  double voxel(int ix, int iy, int iz) const
  {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_) {return unknown();}
    const float v = d_[(static_cast<size_t>(ix) * ny_ + iy) * nz_ + iz];
    if (std::fabs(v - unobs_) < 1e-2f) {return unknown();}
    return v;
  }

  // True if the voxel containing p was never observed, or p is outside the grid
  // (RAPTOR Sec. VI-A: the trajectory "exits the known-free space and enters the unknown space")
  bool isUnknown(const Eigen::Vector3d & p) const
  {
    if (empty()) {return true;}
    const Eigen::Vector3d q = (p - origin_) / vs_;
    const int ix = static_cast<int>(std::floor(q.x()));
    const int iy = static_cast<int>(std::floor(q.y()));
    const int iz = static_cast<int>(std::floor(q.z()));
    if (ix < 0 || iy < 0 || iz < 0 || ix >= nx_ || iy >= ny_ || iz >= nz_) {return true;}
    const float v = d_[(static_cast<size_t>(ix) * ny_ + iy) * nz_ + iz];
    return std::fabs(v - unobs_) < 1e-2f;
  }

  // Trilinear distance; optional gradient (d distance / d position)
  double distance(const Eigen::Vector3d & p, Eigen::Vector3d * grad = nullptr) const
  {
    if (empty()) {
      if (grad) {grad->setZero();}
      return unknown();
    }
    const Eigen::Vector3d q = (p - origin_) / vs_ - Eigen::Vector3d::Constant(0.5);
    const Eigen::Vector3d fl = q.array().floor();
    const int x0 = static_cast<int>(fl.x()), y0 = static_cast<int>(fl.y()),
      z0 = static_cast<int>(fl.z());
    const Eigen::Vector3d f = q - fl;
    double c[2][2][2];
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          c[i][j][k] = voxel(x0 + i, y0 + j, z0 + k);
        }
      }
    }
    // interpolate along x, then y, then z
    double cx[2][2];
    for (int j = 0; j < 2; ++j) {
      for (int k = 0; k < 2; ++k) {
        cx[j][k] = c[0][j][k] * (1 - f.x()) + c[1][j][k] * f.x();
      }
    }
    double cxy[2];
    for (int k = 0; k < 2; ++k) {cxy[k] = cx[0][k] * (1 - f.y()) + cx[1][k] * f.y();}
    const double dist = cxy[0] * (1 - f.z()) + cxy[1] * f.z();

    if (grad) {
      // partial derivatives of the trilinear interpolant
      double gx = 0, gy = 0;
      for (int j = 0; j < 2; ++j) {
        for (int k = 0; k < 2; ++k) {
          const double wy = j ? f.y() : 1 - f.y();
          const double wz = k ? f.z() : 1 - f.z();
          gx += (c[1][j][k] - c[0][j][k]) * wy * wz;
        }
      }
      for (int i = 0; i < 2; ++i) {
        for (int k = 0; k < 2; ++k) {
          const double wx = i ? f.x() : 1 - f.x();
          const double wz = k ? f.z() : 1 - f.z();
          gy += (c[i][1][k] - c[i][0][k]) * wx * wz;
        }
      }
      const double gz = cxy[1] - cxy[0];
      *grad = Eigen::Vector3d(gx, gy, gz) / vs_;
    }
    return dist;
  }

private:
  double unknown() const {return unknown_is_occupied ? -vs_ : unknown_distance;}

  Eigen::Vector3d origin_{Eigen::Vector3d::Zero()};
  double vs_{0.0};
  int nx_{0}, ny_{0}, nz_{0};
  std::vector<float> d_;
  float unobs_{-1000.0f};
};

}  // namespace da3_metric_scale
