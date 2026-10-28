#include <chrono>
#include <iostream>

#include "da3_metric_scale/kino_astar.hpp"

int main() {
  kino::Params p;
  p.max_vel = 3.0;
  p.max_acc = 2.0;
  p.resolution = 0.2;
  p.horizon = 50.0;

  // Two walls with gaps; replace this lambda with an nvblox ESDF query (distance > clearance).
  auto is_free = [](const Eigen::Vector3d& x) {
    if (x.z() < 0.0 || x.z() > 3.0) return false;
    if (x.x() > 4.9 && x.x() < 5.3 && !(x.y() > 1.0 && x.y() < 2.0)) return false;
    if (x.x() > 9.9 && x.x() < 10.3 && !(x.y() > -2.0 && x.y() < -1.0)) return false;
    return true;
  };

  kino::KinodynamicAStar planner(p, is_free);
  auto t0 = std::chrono::steady_clock::now();
  auto res = planner.search({0, 0, 1}, {0, 0, 0}, {15, 0, 1});
  double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

  std::cout << "status=" << static_cast<int>(res.status) << " expanded=" << res.expanded
            << " samples=" << res.traj.size() << " time=" << ms << " ms\n";
  if (!res.traj.empty()) {
    const auto& e = res.traj.back();
    std::cout << "end pos=" << e.pos.transpose() << " vel=" << e.vel.transpose() << " t=" << e.t << "\n";
    double vmax = 0;
    bool ok = true;
    for (const auto& s : res.traj) {
      vmax = std::max(vmax, s.vel.cwiseAbs().maxCoeff());
      ok &= is_free(s.pos);
    }
    std::cout << "max |v| axis=" << vmax << " collision_free=" << ok << "\n";
  }
  return 0;
}
