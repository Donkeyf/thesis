// Closed-loop test of feature_map_node with mocked ROS (synthetic checkerboard wall at 2 m).
//   g++ -O2 -std=c++17 -Itest/mock -Iinclude -I/usr/include/eigen3 $(pkg-config --cflags opencv4) test/test_feature_map.cpp $(pkg-config --libs opencv4) -o fmap && ./fmap
#define main feature_map_main
#include "../src/feature_map_node.cpp"
#undef main

#include <cstdio>

using rclcpp::Publisher;
using rclcpp::Subscription;

template<class T> std::shared_ptr<T> get(rclcpp::Node & n, const std::string & k, bool sub)
{
  auto & m = sub ? n.subs : n.pubs;
  auto it = m.find(k);
  if (it == m.end()) {std::printf("missing %s\n", k.c_str()); std::exit(2);}
  return std::static_pointer_cast<T>(it->second);
}

int main()
{
  int fails = 0;
  auto check = [&](bool ok, const char * w) {if (!ok) {std::printf("FAIL: %s\n", w); ++fails;}};
  rclcpp::g_quiet = true;

  // camera at (0, 0, 1) in map, looking along +x: optical (x right, y down, z fwd) -> map
  Eigen::Matrix3d R;
  R << 0, 0, 1,
    -1, 0, 0,
    0, -1, 0;
  const Eigen::Quaterniond q(R);
  tf2_ros::g_tf.transform.rotation.x = q.x();
  tf2_ros::g_tf.transform.rotation.y = q.y();
  tf2_ros::g_tf.transform.rotation.z = q.z();
  tf2_ros::g_tf.transform.rotation.w = q.w();
  tf2_ros::g_tf.transform.translation.z = 1.0;

  FeatureMapNode node;
  auto img_sub = get<Subscription<Image>>(node, "/image_da3", true);
  auto depth_sub = get<Subscription<Image>>(node, "/da3_metric/depth", true);
  auto info_sub = get<Subscription<CameraInfo>>(node, "/da3_metric/camera_info", true);
  auto out = get<Publisher<PointCloud2>>(node, "~/landmarks", false);

  const int W = 640, H = 480, SQ = 40;
  const double fx = 550, fy = 550, cx = 320, cy = 240, Z = 2.0;
  CameraInfo ci;
  ci.width = W;
  ci.height = H;
  ci.k = {fx, 0, cx, 0, fy, cy, 0, 0, 1};
  ci.d = {0, 0, 0, 0, 0};
  info_sub->inject(ci);

  // checkerboard image (mono8) and constant depth, except an invalid (edge-masked) band
  Image img;
  img.width = W; img.height = H; img.encoding = "mono8"; img.step = W;
  img.data.resize(W * H);
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {img.data[v * W + u] = ((u / SQ + v / SQ) % 2) ? 220 : 30;}
  }
  Image dep;
  dep.width = W; dep.height = H; dep.encoding = "32FC1"; dep.step = W * 4;
  dep.header.frame_id = "camera_optical";
  dep.data.resize(W * H * 4);
  for (int v = 0; v < H; ++v) {
    for (int u = 0; u < W; ++u) {
      const float z = (u >= 300 && u < 340) ? 0.0f : static_cast<float>(Z);   // masked column band
      std::memcpy(&dep.data[(v * W + u) * 4], &z, 4);
    }
  }

  // 4 frames (same view) -> every landmark observed 4 times (min_observations = 3)
  for (int k = 0; k < 4; ++k) {
    rclcpp::g_now += 0.1;
    img.header.stamp = rclcpp::Time::fromSec(rclcpp::g_now);
    dep.header.stamp = img.header.stamp;
    img_sub->inject(img);
    depth_sub->inject(dep);
  }
  // a depth frame whose image never arrived is ignored
  dep.header.stamp = rclcpp::Time::fromSec(rclcpp::g_now + 5.0);
  depth_sub->inject(dep);
  node.spinUntil(rclcpp::g_now + 1.0);

  check(!out->sent.empty(), "no landmarks published");
  if (out->sent.empty()) {return 1;}
  const auto & pc = out->sent.back();
  std::printf("published %u landmarks (frame %s)\n", pc.width, pc.header.frame_id.c_str());
  check(pc.width > 40, "too few landmarks");
  check(pc.header.frame_id == "map", "wrong frame");
  int bad = 0, in_band = 0;
  double max_err = 0, min_score = 1, max_score = 0;
  for (uint32_t i = 0; i < pc.width; ++i) {
    float p[4];
    std::memcpy(p, &pc.data[i * 16], 16);
    // wall at map x = 2; back to pixels: u = cx - y fx / Z, v = cy - (z - 1) fy / Z
    const double u = cx - p[1] * fx / Z, v = cy - (p[2] - 1.0) * fy / Z;
    const double eu = std::fabs(u - std::round(u / SQ) * SQ), ev = std::fabs(v - std::round(v / SQ) * SQ);
    max_err = std::max({max_err, std::fabs(p[0] - Z), eu, ev});
    if (std::fabs(p[0] - Z) > 0.02 || eu > 2.0 || ev > 2.0) {++bad;}
    if (u >= 299 && u < 341) {++in_band;}
    min_score = std::min<double>(min_score, p[3]);
    max_score = std::max<double>(max_score, p[3]);
  }
  std::printf("landmarks off a checker corner: %d, in the masked band: %d, max err %.3f, score %.2f..%.2f\n",
    bad, in_band, max_err, min_score, max_score);
  check(bad == 0, "landmark not on a checkerboard corner at x = 2 m");
  check(in_band == 0, "landmark from invalid (masked) depth");
  check(min_score > 0 && max_score <= 1.0, "score outside (0, 1]");

  // lost TF -> frame skipped, nothing new stored
  tf2_ros::g_tf_ok = false;
  rclcpp::g_now += 0.1;
  img.header.stamp = rclcpp::Time::fromSec(rclcpp::g_now);
  dep.header.stamp = img.header.stamp;
  img_sub->inject(img);
  depth_sub->inject(dep);

  // landmark map: forgetting and capacity
  da3_metric_scale::LandmarkMapParams lp;
  lp.forget_time = 10; lp.max_landmarks = 5; lp.min_obs = 1;
  da3_metric_scale::LandmarkMap lm(lp);
  for (int i = 0; i < 10; ++i) {lm.insert(Eigen::Vector3d(i, 0, 0), 0.1 * (i + 1), 0.0);}
  lm.insert(Eigen::Vector3d(0.02, 0.01, 0), 0.5, 1.0);           // same voxel -> merged
  lm.prune(5.0);
  check(lm.size() == 5, "capacity not enforced");
  lm.prune(20.0);
  check(lm.size() == 0, "stale landmarks not forgotten");

  std::printf("%s (%d failures)\n", fails ? "FEATURE MAP TEST FAILED" : "FEATURE MAP TEST OK", fails);
  return fails;
}
