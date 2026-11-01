// Unit tests for the perception-aware yaw planner (RAPTOR Sec. VII + feature terms).
//   g++ -O2 -std=c++17 -Iinclude -I/usr/include/eigen3 test/test_yaw_planner.cpp -o yawtest && ./yawtest
#include <cstdio>
#include <vector>

#include "da3_metric_scale/yaw_planner.hpp"

using da3_metric_scale::EsdfMap;
using da3_metric_scale::Landmark;
using da3_metric_scale::YawPlanner;
using da3_metric_scale::YawPlanInput;
using da3_metric_scale::YawPlanResult;
using V3 = Eigen::Vector3d;

static int fails = 0;
#define CHECK(c, ...) do {if (!(c)) {std::printf("FAIL: " __VA_ARGS__); std::printf("\n"); ++fails;}} while (0)

constexpr float UNOBS = -1000.f;
constexpr double DEG = 180.0 / M_PI;

// 8 x 8 x 2 m grid at 5 cm, origin (-2, -4, 0); fill(p) gives the value of each voxel
template<class Fn> EsdfMap makeMap(Fn fill)
{
  const double vs = 0.05;
  const int nx = 160, ny = 160, nz = 40;
  const V3 o(-2, -4, 0);
  std::vector<float> d(static_cast<size_t>(nx) * ny * nz);
  for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j) for (int k = 0; k < nz; ++k) {
    d[(static_cast<size_t>(i) * ny + j) * nz + k] = fill(o + vs * V3(i + 0.5, j + 0.5, k + 0.5));
  }
  EsdfMap m;
  m.set(o, vs, nx, ny, nz, d, UNOBS);
  return m;
}

// straight line along +x at 1 m height, 0.75 m/s
YawPlanInput straight(const EsdfMap * m, const std::vector<Landmark> * lm, double yaw0 = 0)
{
  YawPlanInput in;
  in.pos = [](double t) {return V3(0.75 * t, 0, 1.0);};
  in.vel = [](double) {return V3(0.75, 0, 0);};
  in.duration = 4.0;
  in.yaw0 = yaw0;
  in.map = m;
  in.landmarks = lm;
  return in;
}

double maxAbsKey(const YawPlanResult & r)
{
  double a = 0;
  for (double y : r.key_yaw) {a = std::max(a, std::fabs(y));}
  return a;
}
double meanKey(const YawPlanResult & r)
{
  double s = 0;
  for (size_t i = 1; i < r.key_yaw.size(); ++i) {s += r.key_yaw[i];}
  return s / std::max<size_t>(1, r.key_yaw.size() - 1);
}
void print(const char * name, const YawPlanResult & r)
{
  std::printf("%-34s ok %d | key yaw [deg]:", name, r.ok);
  for (double y : r.key_yaw) {std::printf(" %5.0f", y * DEG);}
  std::printf(" | max rate %4.1f deg/s | IG %.1f ms feat %.1f ms search %.1f ms opt %.2f ms | %s\n",
    r.max_rate * DEG, r.t_ig, r.t_feat, r.t_search, r.t_opt, r.message.c_str());
}

int main()
{
  YawPlanner yp;
  const auto & P = yp.params();

  // 1) everything known and free: yaw follows the flight direction (velocity prior, smoothness)
  {
    auto m = makeMap([](const V3 &) {return 1.0f;});
    auto in = straight(&m, nullptr);
    auto r = yp.plan(in);
    print("known free", r);
    CHECK(r.ok, "known free: plan failed");
    CHECK(maxAbsKey(r) < 1e-6, "known free: yaw should stay at 0 (max %.1f deg)", maxAbsKey(r) * DEG);
  }

  // 2) unknown region on the left of the path (+y): the camera turns left to observe it
  auto leftUnknown = [](const V3 & p) {
      return (p.y() > 1.0 && p.x() > 0.5 && p.x() < 3.5) ? UNOBS : 1.0f;
    };
  {
    auto m = makeMap(leftUnknown);
    auto r = yp.plan(straight(&m, nullptr));
    print("unknown on the left", r);
    CHECK(r.ok && meanKey(r) > 20.0 / DEG, "unknown left: expected a left turn (mean %.1f deg)",
      meanKey(r) * DEG);
    // same on the right -> turns right
    auto m2 = makeMap([](const V3 & p) {
        return (p.y() < -1.0 && p.x() > 0.5 && p.x() < 3.5) ? UNOBS : 1.0f;
      });
    auto r2 = yp.plan(straight(&m2, nullptr));
    print("unknown on the right", r2);
    CHECK(r2.ok && meanKey(r2) < -20.0 / DEG, "unknown right: expected a right turn (mean %.1f deg)",
      meanKey(r2) * DEG);
  }

  // 3) the same unknown region hidden behind a wall: not visible -> no gain -> no turn
  {
    auto m = makeMap([&](const V3 & p) {
        if (p.y() > 0.6 && p.y() < 0.7) {return 0.0f;}           // occupied wall at y = 0.65
        return leftUnknown(p);
      });
    auto r = yp.plan(straight(&m, nullptr));
    print("unknown behind a wall", r);
    CHECK(r.ok && r.ig_voxels == 0, "wall: hidden unknown voxels should not count (%d)", r.ig_voxels);
    CHECK(maxAbsKey(r) < 1e-6, "wall: yaw should stay at 0 (max %.1f deg)", maxAbsKey(r) * DEG);
  }

  // 4) unknown far from the trajectory counts less (Eq. 16): left region near, right region far
  {
    auto m = makeMap([](const V3 & p) {
        if (p.y() > 1.0 && p.y() < 1.6 && p.x() > 0.5 && p.x() < 3.5) {return UNOBS;}   // near, small
        if (p.y() < -2.8 && p.x() > 0.5 && p.x() < 3.5) {return UNOBS;}                 // far, large
        return 1.0f;
      });
    auto r = yp.plan(straight(&m, nullptr));
    print("near (left) vs far (right) unknown", r);
    CHECK(r.ok && meanKey(r) > 0, "lateral weighting: expected to favour the near region (mean %.1f deg)",
      meanKey(r) * DEG);
  }

  // 5) features: map known, good landmarks only on the right wall -> camera keeps them in view
  {
    auto m = makeMap([](const V3 & p) {return (p.y() < -1.95 && p.y() > -2.05) ? 0.0f : 1.0f;});
    std::vector<Landmark> lm;
    for (double x = 0.0; x <= 4.0; x += 0.25) {
      for (double z = 0.6; z <= 1.4; z += 0.2) {lm.push_back({V3(x, -1.9, z), 0.9});}
    }
    auto r = yp.plan(straight(&m, &lm));
    print("landmarks on the right", r);
    CHECK(r.ok && meanKey(r) < -20.0 / DEG, "features: expected to look right (mean %.1f deg)",
      meanKey(r) * DEG);
    CHECK(r.key_feat.back() > 0, "features: none in view at the end");
    // landmarks straight ahead -> no reason to turn
    std::vector<Landmark> ahead;
    for (double y = -0.5; y <= 0.5; y += 0.25) {
      for (double z = 0.6; z <= 1.4; z += 0.2) {ahead.push_back({V3(5.0, y, z), 0.9});}
    }
    auto r2 = yp.plan(straight(&m, &ahead));
    print("landmarks ahead", r2);
    CHECK(r2.ok && maxAbsKey(r2) < 1e-6, "features ahead: yaw should stay at 0 (max %.1f deg)",
      maxAbsKey(r2) * DEG);
  }

  // 6) feasibility + boundary conditions of the optimised yaw spline (start turning, yaw0 = 1 rad)
  {
    auto m = makeMap(leftUnknown);
    auto in = straight(&m, nullptr, 1.0);
    in.yaw_rate0 = 0.3;
    auto r = yp.plan(in);
    print("start yaw 57 deg, rate 0.3", r);
    CHECK(std::fabs(r.yaw(0) - 1.0) < 1e-9, "start yaw %.4f", r.yaw(0));
    CHECK(std::fabs(r.rate(0) - 0.3) < 1e-9, "start rate %.4f", r.rate(0));
    double vmax = 0, emax = 0;
    for (double t = 0; t <= r.t_end; t += 0.01) {vmax = std::max(vmax, std::fabs(r.rate(t)));}
    for (size_t i = 1; i < r.key_t.size(); ++i) {
      emax = std::max(emax, std::fabs(r.yaw(r.key_t[i]) - r.key_yaw[i]));
    }
    std::printf("   spline: max |yaw rate| %.1f deg/s (limit %.0f), max waypoint error %.1f deg\n",
      vmax * DEG, P.max_yaw_rate * DEG, emax * DEG);
    CHECK(vmax < 1.15 * P.max_yaw_rate, "yaw rate %.1f deg/s exceeds the limit", vmax * DEG);
    CHECK(emax < 12.0 / DEG, "waypoint error %.1f deg", emax * DEG);
    // yaw is held after the plan horizon
    CHECK(std::fabs(r.yaw(r.t_end + 3.0) - r.yaw(r.t_end)) < 1e-12 && r.rate(r.t_end + 1) == 0,
      "yaw not held after t_end");
  }

  // 7) wrap-around: start yaw near +pi, unknown behind-left -> continuous (unwrapped) sequence
  {
    auto m = makeMap(leftUnknown);
    YawPlanInput in = straight(&m, nullptr, 3.0);
    in.vel = {};                                         // no velocity prior
    auto r = yp.plan(in);
    print("start yaw 172 deg (wrap)", r);
    bool cont = true;
    for (size_t i = 1; i < r.key_yaw.size(); ++i) {
      cont = cont && std::fabs(r.key_yaw[i] - r.key_yaw[i - 1]) <= P.max_yaw_rate * P.key_dt + 1e-6;
    }
    CHECK(r.ok && cont, "wrap: key yaws not continuous");
  }

  // 8) timing with a realistic mixed map + 1000 landmarks
  {
    auto m = makeMap([](const V3 & p) {
        if (std::fabs(p.x() - 2.0) < 0.05 && p.y() > -0.5 && p.y() < 0.5) {return 0.0f;}
        return (p.x() > 1.5 || std::fabs(p.y()) > 1.5) ? UNOBS : 1.0f;
      });
    std::vector<Landmark> lm;
    for (int k = 0; k < 1000; ++k) {
      lm.push_back({V3(-1.5 + 0.0055 * k, (k % 2 ? 1.8 : -1.8), 0.5 + 0.001 * k), 0.5});
    }
    auto r = yp.plan(straight(&m, &lm));
    print("mixed map, 1000 landmarks", r);
    const double tot = r.t_ig + r.t_feat + r.t_search + r.t_opt;
    std::printf("   total %.1f ms\n", tot);
    CHECK(r.ok && tot < 150.0, "too slow: %.1f ms", tot);
  }

  std::printf("%s (%d failures)\n", fails ? "YAW PLANNER TEST FAILED" : "YAW PLANNER TEST OK", fails);
  return fails;
}
