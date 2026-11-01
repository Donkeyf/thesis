// landmark_map.hpp - persistent map of visual landmarks ("good features") in the map frame.
//
// Each observation is a 3D corner position (from the metric depth) with a quality in [0, 1]
// (Shi-Tomasi response relative to the strongest corner of the frame). Observations are merged
// per voxel; a landmark's score combines its average quality with how often it was re-detected,
// so features that are strong AND repeatable (i.e. trackable) score highest:
//     score = quality_avg * min(1, n_obs / n_full)
// Landmarks with fewer than min_obs observations are not exported (single-frame outliers from
// monocular depth). Landmarks not re-observed for forget_time are dropped; beyond max_landmarks the
// lowest-scoring ones are dropped.
//
// ROS-free (Eigen only).
#pragma once

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include "da3_metric_scale/yaw_planner.hpp"   // Landmark

namespace da3_metric_scale
{

struct LandmarkMapParams
{
  double voxel = 0.10;          // m, observations within one voxel are the same landmark
  int min_obs = 3;              // observations before a landmark is exported
  int n_full = 10;              // observations for full score
  double forget_time = 120.0;   // s
  int max_landmarks = 2000;
};

class LandmarkMap
{
public:
  explicit LandmarkMap(const LandmarkMapParams & p = {})
  : P_(p) {}
  LandmarkMapParams & params() {return P_;}

  void insert(const Eigen::Vector3d & p, double quality, double t)
  {
    const long long k = key(p);
    auto it = lm_.find(k);
    if (it == lm_.end()) {
      Entry e;
      e.p = p;
      e.q_sum = quality;
      e.n = 1;
      e.last = t;
      lm_.emplace(k, e);
      return;
    }
    Entry & e = it->second;
    if (t - e.last < 1e-6) {return;}             // one observation per frame
    e.p = (e.p * e.n + p) / (e.n + 1);
    e.q_sum += quality;
    ++e.n;
    e.last = t;
  }

  // drop stale landmarks and enforce the capacity
  void prune(double t)
  {
    for (auto it = lm_.begin(); it != lm_.end(); ) {
      if (t - it->second.last > P_.forget_time) {it = lm_.erase(it);} else {++it;}
    }
    if (static_cast<int>(lm_.size()) <= P_.max_landmarks) {return;}
    std::vector<std::pair<double, long long>> s;
    s.reserve(lm_.size());
    for (const auto & kv : lm_) {s.emplace_back(score(kv.second), kv.first);}
    std::nth_element(s.begin(), s.begin() + (s.size() - P_.max_landmarks), s.end());
    for (size_t i = 0; i < s.size() - P_.max_landmarks; ++i) {lm_.erase(s[i].second);}
  }

  std::vector<Landmark> exportLandmarks() const
  {
    std::vector<Landmark> out;
    for (const auto & kv : lm_) {
      if (kv.second.n < P_.min_obs) {continue;}
      out.push_back({kv.second.p, score(kv.second)});
    }
    return out;
  }

  size_t size() const {return lm_.size();}

private:
  struct Entry {Eigen::Vector3d p; double q_sum{0}; int n{0}; double last{0};};

  double score(const Entry & e) const
  {
    return (e.q_sum / e.n) * std::min(1.0, static_cast<double>(e.n) / std::max(1, P_.n_full));
  }
  long long key(const Eigen::Vector3d & p) const
  {
    const long long x = static_cast<long long>(std::floor(p.x() / P_.voxel)) + (1 << 20);
    const long long y = static_cast<long long>(std::floor(p.y() / P_.voxel)) + (1 << 20);
    const long long z = static_cast<long long>(std::floor(p.z() / P_.voxel)) + (1 << 20);
    return (x << 42) | (y << 21) | z;
  }

  LandmarkMapParams P_;
  std::unordered_map<long long, Entry> lm_;
};

}  // namespace da3_metric_scale
