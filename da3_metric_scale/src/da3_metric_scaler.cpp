// da3_metric_scaler.cpp
//
// Metricises Depth Anything 3 depth using OptiTrack mocap poses.
//
//   Method A (floor):  floor is a known plane in the mocap frame. Rays from
//                      floor pixels are intersected with it to get metric depth.
//   Method B (tri):    features are tracked between a keyframe and the current
//                      frame. They are triangulated with the metric mocap camera
//                      poses, which gives sparse metric depth.
//
// Both methods produce (DA3 value, metric target) pairs. Each set is fitted
// robustly (RANSAC + LS refit). The two fits are fused, weighted by inlier
// count, and EMA-smoothed. The fused scale is applied to the whole DA3 image.
//
// Camera pose:  T_W_C = T_W_B(mocap) * T_B_I * inv(Kalibr T_cam_imu)
// Kalibr's camera frame is already an optical frame (z fwd, x right, y down).
// Mocap defaults: world Z-up; rigid body x = left, -y = nose forward, z = up.

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
// message_filters switched to .hpp headers (and rclcpp::QoS subscribe) after Jazzy
#if __has_include(<message_filters/subscriber.hpp>)
#include <message_filters/subscriber.hpp>
#include <message_filters/sync_policies/approximate_time.hpp>
#include <message_filters/synchronizer.hpp>
#define DA3_MF_RCLCPP_QOS 1
#else
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#define DA3_MF_RCLCPP_QOS 0
#endif
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#if __has_include(<tf2_ros/transform_broadcaster.hpp>)
#include <tf2_ros/transform_broadcaster.hpp>
#else
#include <tf2_ros/transform_broadcaster.h>
#endif

using CameraInfo = sensor_msgs::msg::CameraInfo;
using Image = sensor_msgs::msg::Image;
using PoseStamped = geometry_msgs::msg::PoseStamped;

namespace
{

struct Sample
{
  double x;  // DA3 value (depth or disparity)
  double y;  // metric target in the same space (depth, or inverse depth)
};

struct Fit
{
  bool ok{false};
  double s{1.0};
  double b{0.0};
  int inliers{0};
};

struct PoseSample
{
  double t;
  Eigen::Isometry3d T;
};

std::vector<double> identity16()
{
  return {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
}

cv::Mat projFromIso(const Eigen::Isometry3d & T_C_W)
{
  cv::Mat P(3, 4, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 4; ++c) {
      P.at<double>(r, c) = T_C_W.matrix()(r, c);
    }
  }
  return P;
}

}  // namespace

class Da3MetricScaler : public rclcpp::Node
{
public:
  Da3MetricScaler()
  : Node("da3_metric_scaler"), rng_(42)
  {
    // ---------------- Topics ----------------
    const auto image_topic = declare_parameter<std::string>("image_topic", "/image_throttled");
    const auto depth_topic =
      declare_parameter<std::string>("depth_topic", "/depth_anything_3/depth");
    const auto info_topic = declare_parameter<std::string>("camera_info_topic", "/camera_info");
    const auto pose_topic =
      declare_parameter<std::string>("mocap_pose_topic", "/vrpn_mocap/dilon/pose");
    const auto out_depth_topic =
      declare_parameter<std::string>("output_depth_topic", "/da3_metric/depth");
    const auto out_info_topic =
      declare_parameter<std::string>("output_camera_info_topic", "/da3_metric/camera_info");
    const auto stats_topic = declare_parameter<std::string>("stats_topic", "/da3_metric/scale");
    const int sync_queue = declare_parameter<int>("sync_queue_size", 30);
    const double sync_max_dt = declare_parameter<double>("sync_max_interval", 0.01);

    // ---------------- Frames / extrinsics ----------------
    world_frame_ = declare_parameter<std::string>("world_frame", "mocap");
    camera_frame_ = declare_parameter<std::string>("camera_frame", "camera_optical");
    publish_tf_ = declare_parameter<bool>("publish_tf", true);
    const auto T_cam_imu_v =
      declare_parameter<std::vector<double>>("kalibr_T_cam_imu", identity16());
    const auto R_body_imu_v = declare_parameter<std::vector<double>>(
      "R_body_imu", std::vector<double>{0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0});
    const auto p_body_imu_v = declare_parameter<std::vector<double>>(
      "p_body_imu", std::vector<double>{0.0, 0.0, 0.0});
    mocap_time_offset_ = declare_parameter<double>("mocap_time_offset", 0.0);
    max_pose_gap_ = declare_parameter<double>("max_pose_gap", 0.05);

    // ---------------- Model / output ----------------
    const auto method = declare_parameter<std::string>("method", "both");
    use_floor_ = (method == "floor" || method == "both");
    use_tri_ = (method == "triangulation" || method == "both");
    if (!use_floor_ && !use_tri_) {
      throw std::runtime_error("method must be 'floor', 'triangulation' or 'both'");
    }
    da3_is_disparity_ = declare_parameter<bool>("da3_is_disparity", false);
    fit_shift_ = declare_parameter<bool>("fit_shift", false);
    ema_alpha_ = declare_parameter<double>("ema_alpha", 0.1);
    min_depth_ = declare_parameter<double>("min_depth", 0.1);
    max_depth_ = declare_parameter<double>("max_depth", 10.0);

    // ---------------- RANSAC ----------------
    ransac_iters_ = declare_parameter<int>("ransac_iterations", 200);
    ransac_rel_thresh_ = declare_parameter<double>("ransac_rel_threshold", 0.10);
    min_samples_ = declare_parameter<int>("min_samples", 20);
    min_inlier_ratio_ = declare_parameter<double>("min_inlier_ratio", 0.4);

    // ---------------- Method A: floor ----------------
    const auto up_v = declare_parameter<std::vector<double>>(
      "world_up", std::vector<double>{0.0, 0.0, 1.0});
    floor_height_ = declare_parameter<double>("floor_height", 0.0);
    floor_roi_frac_ = declare_parameter<double>("floor_roi_fraction", 0.5);
    floor_grid_step_ = declare_parameter<int>("floor_grid_step", 16);
    min_grazing_deg_ = declare_parameter<double>("floor_min_grazing_deg", 15.0);
    max_floor_range_ = declare_parameter<double>("floor_max_depth", 6.0);
    min_cam_height_ = declare_parameter<double>("min_camera_height", 0.15);

    // ---------------- Method B: triangulation ----------------
    max_features_ = declare_parameter<int>("max_features", 300);
    min_feature_dist_ = declare_parameter<double>("min_feature_distance", 12.0);
    min_tracks_ = declare_parameter<int>("min_tracks", 30);
    min_baseline_ = declare_parameter<double>("min_baseline", 0.12);
    max_kf_frames_ = declare_parameter<int>("max_keyframe_frames", 60);
    min_parallax_deg_ = declare_parameter<double>("min_parallax_deg", 2.0);
    max_reproj_px_ = declare_parameter<double>("max_reproj_px", 1.5);
    fb_thresh_px_ = declare_parameter<double>("lk_fb_threshold_px", 1.0);
    subpix_enable_ = declare_parameter<bool>("subpix_enable", true);
    subpix_win_ = declare_parameter<int>("subpix_window", 5);
    subpix_iters_ = declare_parameter<int>("subpix_max_iterations", 30);
    subpix_eps_ = declare_parameter<double>("subpix_epsilon", 0.01);

    // ---------------- DA3 depth-edge mask ----------------
    edge_enable_ = declare_parameter<bool>("edge_mask_enable", true);
    edge_rel_grad_ = declare_parameter<double>("edge_rel_gradient", 0.05);
    edge_dilate_px_ = declare_parameter<int>("edge_dilate_px", 2);
    edge_mask_output_ = declare_parameter<bool>("edge_mask_output", true);
    edge_mask_detection_ = declare_parameter<bool>("edge_mask_detection", true);

    // ---------------- Build T_body_cam ----------------
    if (T_cam_imu_v.size() != 16 || R_body_imu_v.size() != 9 || p_body_imu_v.size() != 3 ||
      up_v.size() != 3)
    {
      throw std::runtime_error(
              "kalibr_T_cam_imu needs 16, R_body_imu 9, p_body_imu 3, world_up 3 values");
    }
    Eigen::Isometry3d T_cam_imu = Eigen::Isometry3d::Identity();
    for (int i = 0; i < 16; ++i) {
      T_cam_imu.matrix()(i / 4, i % 4) = T_cam_imu_v[i];
    }
    // Kalibr imu-cam chain file overrides the inline matrix (and optionally camera_info)
    const auto kalibr_file = declare_parameter<std::string>("kalibr_file", "");
    const auto kalibr_cam = declare_parameter<std::string>("kalibr_camera", "cam0");
    const bool kalibr_intr = declare_parameter<bool>("use_kalibr_intrinsics", true);
    if (!kalibr_file.empty()) {
      T_cam_imu = loadKalibr(kalibr_file, kalibr_cam, kalibr_intr);
    }
    if (T_cam_imu.matrix().isIdentity(1e-9)) {
      RCLCPP_WARN(get_logger(), "T_cam_imu is identity - set kalibr_file or kalibr_T_cam_imu");
    }
    Eigen::Matrix3d R_bi;
    for (int i = 0; i < 9; ++i) {
      R_bi(i / 3, i % 3) = R_body_imu_v[i];
    }
    if (std::abs(R_bi.determinant() - 1.0) > 1e-3) {
      throw std::runtime_error("R_body_imu is not a proper rotation (det != 1)");
    }
    Eigen::Isometry3d T_body_imu = Eigen::Isometry3d::Identity();
    T_body_imu.linear() = R_bi;
    T_body_imu.translation() = Eigen::Vector3d(p_body_imu_v[0], p_body_imu_v[1], p_body_imu_v[2]);
    T_body_cam_ = T_body_imu * T_cam_imu.inverse();
    world_up_ = Eigen::Vector3d(up_v[0], up_v[1], up_v[2]).normalized();

    // ---------------- ROS I/O ----------------
    depth_pub_ = create_publisher<Image>(out_depth_topic, rclcpp::QoS(10));
    info_pub_ = create_publisher<CameraInfo>(out_info_topic, rclcpp::QoS(10));
    stats_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(stats_topic, 10);
    tf_br_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    info_sub_ = create_subscription<CameraInfo>(
      info_topic, rclcpp::SensorDataQoS(),
      std::bind(&Da3MetricScaler::onInfo, this, std::placeholders::_1));
    pose_sub_ = create_subscription<PoseStamped>(
      pose_topic, rclcpp::SensorDataQoS(),
      std::bind(&Da3MetricScaler::onPose, this, std::placeholders::_1));

#if DA3_MF_RCLCPP_QOS
    img_sub_.subscribe(this, image_topic, rclcpp::SensorDataQoS());
    depth_sub_.subscribe(this, depth_topic, rclcpp::SensorDataQoS());
#else
    img_sub_.subscribe(this, image_topic, rmw_qos_profile_sensor_data);
    depth_sub_.subscribe(this, depth_topic, rmw_qos_profile_sensor_data);
#endif
    // Pass the policy as a temporary: newer message_filters (Kilted+) also has a
    // Synchronizer(F0&, F1&, ...) filter-only constructor that would otherwise
    // grab a named policy as the first input. Synchronizer derives from the
    // policy, so the interval is set on it afterwards.
    sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
      SyncPolicy(static_cast<uint32_t>(sync_queue)), img_sub_, depth_sub_);
    sync_->setMaxIntervalDuration(rclcpp::Duration::from_seconds(sync_max_dt));
    sync_->registerCallback(
      std::bind(&Da3MetricScaler::onFrame, this, std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(
      get_logger(), "da3_metric_scaler up: method=%s, disparity=%d, fit_shift=%d",
      method.c_str(), da3_is_disparity_, fit_shift_);
  }

private:
  using SyncPolicy = message_filters::sync_policies::ApproximateTime<Image, Image>;

  // ======================================================================
  // Kalibr file
  // ======================================================================
  // Reads a Kalibr camchain-imucam YAML with cv::FileStorage (needs the
  // "%YAML:1.0" first line, which the OpenVINS copy already has). Accepts
  // T_cam_imu (Kalibr) or T_imu_cam (OpenVINS convention). Optionally fills
  // camera intrinsics so camera_info isn't needed.
  Eigen::Isometry3d loadKalibr(const std::string & path, const std::string & cam, bool intr)
  {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      throw std::runtime_error("cannot open kalibr_file " + path);
    }
    const cv::FileNode c = fs[cam];
    if (c.empty()) {
      throw std::runtime_error("kalibr_file has no '" + cam + "' entry");
    }

    auto readMat4 = [](const cv::FileNode & n, Eigen::Isometry3d & T) {
        if (n.empty() || !n.isSeq() || n.size() != 4) {return false;}
        T = Eigen::Isometry3d::Identity();
        for (int r = 0; r < 4; ++r) {
          const cv::FileNode row = n[r];
          if (!row.isSeq() || row.size() != 4) {return false;}
          for (int k = 0; k < 4; ++k) {
            T.matrix()(r, k) = static_cast<double>(row[k]);
          }
        }
        return true;
      };
    auto readVec = [](const cv::FileNode & n) {
        std::vector<double> v;
        if (n.isSeq()) {
          for (const auto & e : n) {
            v.push_back(static_cast<double>(e));
          }
        }
        return v;
      };

    Eigen::Isometry3d T_cam_imu;
    if (readMat4(c["T_cam_imu"], T_cam_imu)) {
      RCLCPP_INFO(get_logger(), "kalibr: using %s/T_cam_imu from %s", cam.c_str(), path.c_str());
    } else {
      Eigen::Isometry3d T_imu_cam;
      if (!readMat4(c["T_imu_cam"], T_imu_cam)) {
        throw std::runtime_error("kalibr_file: no 4x4 T_cam_imu or T_imu_cam under " + cam);
      }
      T_cam_imu = T_imu_cam.inverse();
      RCLCPP_INFO(get_logger(), "kalibr: using inverse of %s/T_imu_cam", cam.c_str());
    }
    if (!(T_cam_imu.linear() * T_cam_imu.linear().transpose()).isIdentity(1e-3)) {
      throw std::runtime_error("kalibr_file: T_cam_imu rotation is not orthonormal");
    }

    const double ts = c["timeshift_cam_imu"].empty() ? 0.0 :
      static_cast<double>(c["timeshift_cam_imu"]);
    RCLCPP_INFO(
      get_logger(), "kalibr: timeshift_cam_imu = %.4f s (camera<->FC IMU, not applied to mocap)",
      ts);

    if (intr) {
      const auto in = readVec(c["intrinsics"]);
      const auto dc = readVec(c["distortion_coeffs"]);
      const auto res = readVec(c["resolution"]);
      std::string model, dmodel;
      if (!c["camera_model"].empty()) {c["camera_model"] >> model;}
      if (!c["distortion_model"].empty()) {c["distortion_model"] >> dmodel;}
      if (in.size() != 4 || res.size() != 2) {
        throw std::runtime_error("kalibr_file: need intrinsics[4] and resolution[2]");
      }
      if (model != "pinhole") {
        throw std::runtime_error("kalibr_file: only pinhole camera_model supported, got " + model);
      }
      CameraInfo ci;
      ci.width = static_cast<uint32_t>(res[0]);
      ci.height = static_cast<uint32_t>(res[1]);
      ci.k = {in[0], 0.0, in[2], 0.0, in[1], in[3], 0.0, 0.0, 1.0};
      ci.r = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
      ci.p = {in[0], 0.0, in[2], 0.0, 0.0, in[1], in[3], 0.0, 0.0, 0.0, 1.0, 0.0};
      if (dmodel == "radtan") {
        ci.distortion_model = "plumb_bob";
        ci.d = dc;
        ci.d.resize(5, 0.0);  // k1 k2 p1 p2 [k3]
      } else if (dmodel == "equidistant") {
        ci.distortion_model = "equidistant";
        ci.d = dc;
      } else {
        throw std::runtime_error("kalibr_file: unsupported distortion_model " + dmodel);
      }
      setIntrinsics(ci);
      RCLCPP_INFO(
        get_logger(), "kalibr: intrinsics %ux%u fx=%.1f fy=%.1f cx=%.1f cy=%.1f (%s)",
        ci.width, ci.height, in[0], in[1], in[2], in[3], dmodel.c_str());
    }
    return T_cam_imu;
  }

  void setIntrinsics(const CameraInfo & msg)
  {
    info_ = msg;
    const auto & k = msg.k;
    K_ = (cv::Mat_<double>(3, 3) << k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7], k[8]);
    fisheye_ = (msg.distortion_model == "equidistant");
    D_ = cv::Mat();
    if (!msg.d.empty()) {
      const size_t n = fisheye_ ? std::min<size_t>(4, msg.d.size()) : msg.d.size();
      D_ = cv::Mat(1, static_cast<int>(n), CV_64F);
      for (size_t i = 0; i < n; ++i) {
        D_.at<double>(0, static_cast<int>(i)) = msg.d[i];
      }
    }
    fx_ = k[0];
    buildFloorGrid();
    have_info_ = true;
  }

  // ======================================================================
  // Callbacks
  // ======================================================================
  void onInfo(const CameraInfo::ConstSharedPtr msg)
  {
    if (have_info_) {
      return;  // intrinsics are static (or came from kalibr_file)
    }
    setIntrinsics(*msg);
    RCLCPP_INFO(
      get_logger(), "camera_info: %ux%u, model=%s", msg->width, msg->height,
      msg->distortion_model.c_str());
  }

  void onPose(const PoseStamped::ConstSharedPtr msg)
  {
    PoseSample ps;
    ps.t = rclcpp::Time(msg->header.stamp).seconds();
    const auto & o = msg->pose.orientation;
    const auto & p = msg->pose.position;
    Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
    q.normalize();
    ps.T = Eigen::Isometry3d::Identity();
    ps.T.linear() = q.toRotationMatrix();
    ps.T.translation() = Eigen::Vector3d(p.x, p.y, p.z);
    if (!poses_.empty() && ps.t <= poses_.back().t) {
      return;
    }
    poses_.push_back(ps);
    while (poses_.size() > 2 && poses_.front().t < ps.t - 3.0) {
      poses_.pop_front();
    }
  }

  void onFrame(const Image::ConstSharedPtr & img_msg, const Image::ConstSharedPtr & depth_msg)
  {
    if (!have_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "waiting for camera_info");
      return;
    }

    cv::Mat gray, d;
    try {
      gray = cv_bridge::toCvShare(img_msg, "mono8")->image;
      auto dcv = cv_bridge::toCvShare(depth_msg);
      if (dcv->image.channels() != 1) {
        RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "depth must be single channel");
        return;
      }
      dcv->image.convertTo(d, CV_32F);
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 2000, "cv_bridge: %s", e.what());
      return;
    }
    if (gray.cols != static_cast<int>(info_.width) || gray.rows != static_cast<int>(info_.height)) {
      RCLCPP_ERROR_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "image %dx%d does not match camera_info %ux%u", gray.cols, gray.rows,
        info_.width, info_.height);
      return;
    }

    // Camera pose at the image time
    const double t = rclcpp::Time(depth_msg->header.stamp).seconds() + mocap_time_offset_;
    std::optional<Eigen::Isometry3d> T_W_C;
    if (auto T_W_B = lookupPose(t)) {
      T_W_C = (*T_W_B) * T_body_cam_;
    } else {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "no mocap pose bracketing t=%.3f", t);
    }

    // Depth-edge mask on the DA3 image (used for sampling, detection and output)
    computeEdgeMask(d);

    // Scale estimation
    Fit fa, fb;
    if (use_floor_ && T_W_C) {
      fa = robustFit(floorSamples(d, *T_W_C));
    }
    if (use_tri_) {
      fb = robustFit(triangulationSamples(gray, d, T_W_C));
    }
    fuse(fa, fb);
    publishStats(fa, fb);

    if (!scale_init_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "no scale estimate yet");
      return;
    }

    publishDepth(d, depth_msg->header);
    if (publish_tf_ && T_W_C) {
      publishTf(*T_W_C, depth_msg->header.stamp);
    }
  }

  // ======================================================================
  // Pose handling
  // ======================================================================
  std::optional<Eigen::Isometry3d> lookupPose(double t) const
  {
    if (poses_.size() < 2) {
      return std::nullopt;
    }
    auto it = std::lower_bound(
      poses_.begin(), poses_.end(), t,
      [](const PoseSample & p, double v) {return p.t < v;});
    if (it == poses_.begin() || it == poses_.end()) {
      return std::nullopt;  // no extrapolation
    }
    const PoseSample & b = *it;
    const PoseSample & a = *(it - 1);
    if (b.t - a.t > max_pose_gap_) {
      return std::nullopt;
    }
    const double s = (t - a.t) / (b.t - a.t);
    const Eigen::Quaterniond qa(a.T.linear()), qb(b.T.linear());
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = qa.slerp(s, qb).toRotationMatrix();
    T.translation() = (1.0 - s) * a.T.translation() + s * b.T.translation();
    return T;
  }

  // ======================================================================
  // Helpers
  // ======================================================================
  std::vector<cv::Point2f> normalize(const std::vector<cv::Point2f> & px) const
  {
    std::vector<cv::Point2f> out;
    if (px.empty()) {
      return out;
    }
    if (fisheye_) {
      cv::fisheye::undistortPoints(px, out, K_, D_);
    } else {
      cv::undistortPoints(px, out, K_, D_);
    }
    return out;
  }

  // Depth-edge mask: 255 where DA3 depth changes fast relative to its value
  // (|grad d| / d), or is invalid. Relative gradient is the same for depth and
  // disparity, so this works for either DA3 output. Dilated so samples near an
  // edge are rejected too.
  void computeEdgeMask(const cv::Mat & d)
  {
    if (!edge_enable_) {
      edge_mask_.release();
      edge_fraction_ = 0.0;
      return;
    }
    cv::Mat ds = d.clone();
    cv::patchNaNs(ds, 0.0);
    cv::Mat invalid = ds <= 0.f;

    cv::Mat gx, gy, mag;
    cv::Sobel(ds, gx, CV_32F, 1, 0, 3, 0.125);  // 0.125 -> per-pixel derivative
    cv::Sobel(ds, gy, CV_32F, 0, 1, 3, 0.125);
    cv::magnitude(gx, gy, mag);
    cv::Mat rel = mag / cv::max(ds, 1e-6f);

    edge_mask_ = (rel > edge_rel_grad_) | invalid;
    if (edge_dilate_px_ > 0) {
      const int k = 2 * edge_dilate_px_ + 1;
      cv::dilate(
        edge_mask_, edge_mask_,
        cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k)));
    }
    edge_fraction_ = cv::countNonZero(edge_mask_) / static_cast<double>(edge_mask_.total());
  }

  // Detection mask at camera resolution (non-zero = corners allowed).
  cv::Mat detectionMask(const cv::Size & img_size) const
  {
    if (!edge_mask_detection_ || edge_mask_.empty()) {
      return cv::Mat();
    }
    cv::Mat m;
    cv::resize(edge_mask_, m, img_size, 0, 0, cv::INTER_NEAREST);
    return ~m;
  }

  // Sample DA3 at an image pixel; the depth image may be a different size.
  bool sampleDA3(const cv::Mat & d, const cv::Point2f & p, double & x) const
  {
    const int c = static_cast<int>(std::lround(p.x * d.cols / static_cast<double>(info_.width)));
    const int r = static_cast<int>(std::lround(p.y * d.rows / static_cast<double>(info_.height)));
    if (c < 0 || r < 0 || c >= d.cols || r >= d.rows) {
      return false;
    }
    if (!edge_mask_.empty() && edge_mask_.at<uchar>(r, c)) {
      return false;  // on or near a DA3 depth edge
    }
    const float v = d.at<float>(r, c);
    if (!std::isfinite(v) || v <= 0.f) {
      return false;
    }
    x = v;
    return true;
  }

  double toTarget(double z) const {return da3_is_disparity_ ? 1.0 / z : z;}

  // ======================================================================
  // Method A: floor plane
  // ======================================================================
  void buildFloorGrid()
  {
    floor_px_.clear();
    floor_rays_.clear();
    const int W = static_cast<int>(info_.width);
    const int H = static_cast<int>(info_.height);
    const int step = std::max(2, floor_grid_step_);
    const int v0 = std::clamp(static_cast<int>(H * (1.0 - floor_roi_frac_)), 0, H - 1);
    std::vector<cv::Point2f> px;
    for (int v = v0 + step / 2; v < H; v += step) {
      for (int u = step / 2; u < W; u += step) {
        px.emplace_back(static_cast<float>(u), static_cast<float>(v));
      }
    }
    const auto n = normalize(px);
    for (size_t i = 0; i < px.size(); ++i) {
      floor_px_.push_back(px[i]);
      floor_rays_.emplace_back(n[i].x, n[i].y, 1.0);  // z = 1 -> ray param == optical depth
    }
  }

  std::vector<Sample> floorSamples(const cv::Mat & d, const Eigen::Isometry3d & T_W_C) const
  {
    std::vector<Sample> out;
    const Eigen::Matrix3d R = T_W_C.linear();
    const Eigen::Vector3d c = T_W_C.translation();
    const double h = world_up_.dot(c) - floor_height_;  // camera height above floor
    if (h < min_cam_height_) {
      return out;
    }
    const double sin_min = std::sin(min_grazing_deg_ * M_PI / 180.0);
    out.reserve(floor_rays_.size());
    for (size_t i = 0; i < floor_rays_.size(); ++i) {
      const Eigen::Vector3d rw = R * floor_rays_[i];
      const double denom = world_up_.dot(rw);
      if (denom >= 0.0 || -denom / rw.norm() < sin_min) {
        continue;  // ray points up or hits floor too obliquely
      }
      const double z = h / -denom;  // optical-axis depth, since ray has z = 1
      if (z < min_depth_ || z > max_floor_range_) {
        continue;
      }
      double x;
      if (!sampleDA3(d, floor_px_[i], x)) {
        continue;
      }
      out.push_back({x, toTarget(z)});
    }
    return out;
  }

  // ======================================================================
  // Method B: mocap-pose triangulation
  // ======================================================================
  void resetKeyframe(const cv::Mat & gray, const Eigen::Isometry3d & T_W_C)
  {
    std::vector<cv::Point2f> pts;
    cv::goodFeaturesToTrack(
      gray, pts, max_features_, 0.01, min_feature_dist_, detectionMask(gray.size()));
    if (subpix_enable_ && !pts.empty()) {
      cv::cornerSubPix(
        gray, pts, cv::Size(subpix_win_, subpix_win_), cv::Size(-1, -1),
        cv::TermCriteria(
          cv::TermCriteria::EPS + cv::TermCriteria::COUNT, subpix_iters_, subpix_eps_));
    }
    kf_pts_ = pts;
    trk_pts_ = pts;
    kf_T_W_C_ = T_W_C;
    kf_age_ = 0;
    has_kf_ = !pts.empty();
    prev_gray_ = gray.clone();
  }

  std::vector<Sample> triangulationSamples(
    const cv::Mat & gray, const cv::Mat & d, const std::optional<Eigen::Isometry3d> & T_W_C)
  {
    std::vector<Sample> out;
    if (!T_W_C) {  // lose the chain if mocap drops out
      has_kf_ = false;
      return out;
    }
    if (!has_kf_ || prev_gray_.empty() || prev_gray_.size() != gray.size()) {
      resetKeyframe(gray, *T_W_C);
      return out;
    }

    // Track previous -> current with a forward/backward consistency check
    std::vector<cv::Point2f> nxt, back;
    std::vector<uchar> st1, st2;
    std::vector<float> err;
    const cv::Size win(21, 21);
    cv::calcOpticalFlowPyrLK(prev_gray_, gray, trk_pts_, nxt, st1, err, win, 3);
    cv::calcOpticalFlowPyrLK(gray, prev_gray_, nxt, back, st2, err, win, 3);

    std::vector<cv::Point2f> kf_keep, trk_keep;
    for (size_t i = 0; i < trk_pts_.size(); ++i) {
      if (!st1[i] || !st2[i]) {continue;}
      if (cv::norm(back[i] - trk_pts_[i]) > fb_thresh_px_) {continue;}
      if (nxt[i].x < 0 || nxt[i].y < 0 || nxt[i].x >= gray.cols || nxt[i].y >= gray.rows) {
        continue;
      }
      kf_keep.push_back(kf_pts_[i]);
      trk_keep.push_back(nxt[i]);
    }
    kf_pts_.swap(kf_keep);
    trk_pts_.swap(trk_keep);
    prev_gray_ = gray.clone();
    ++kf_age_;

    if (static_cast<int>(trk_pts_.size()) < min_tracks_ || kf_age_ > max_kf_frames_) {
      resetKeyframe(gray, *T_W_C);
      return out;
    }

    const double baseline = (T_W_C->translation() - kf_T_W_C_.translation()).norm();
    if (baseline < min_baseline_) {
      return out;  // keep tracking until enough baseline
    }

    // Triangulate in normalized coordinates with metric mocap poses
    const auto n1 = normalize(kf_pts_);
    const auto n2 = normalize(trk_pts_);
    const int N = static_cast<int>(n1.size());
    cv::Mat m1(2, N, CV_64F), m2(2, N, CV_64F);
    for (int i = 0; i < N; ++i) {
      m1.at<double>(0, i) = n1[i].x;
      m1.at<double>(1, i) = n1[i].y;
      m2.at<double>(0, i) = n2[i].x;
      m2.at<double>(1, i) = n2[i].y;
    }
    const Eigen::Isometry3d T1 = kf_T_W_C_.inverse();  // C_kf <- W
    const Eigen::Isometry3d T2 = T_W_C->inverse();     // C_cur <- W
    cv::Mat X;
    cv::triangulatePoints(projFromIso(T1), projFromIso(T2), m1, m2, X);
    X.convertTo(X, CV_64F);

    const double min_par = min_parallax_deg_ * M_PI / 180.0;
    const Eigen::Vector3d c1 = kf_T_W_C_.translation();
    const Eigen::Vector3d c2 = T_W_C->translation();
    out.reserve(N);
    for (int i = 0; i < N; ++i) {
      const double w = X.at<double>(3, i);
      if (std::abs(w) < 1e-12) {continue;}
      const Eigen::Vector3d Xw(X.at<double>(0, i) / w, X.at<double>(1, i) / w,
        X.at<double>(2, i) / w);
      const Eigen::Vector3d p1 = T1 * Xw;
      const Eigen::Vector3d p2 = T2 * Xw;
      if (p1.z() <= 0.0 || p2.z() <= 0.0) {continue;}

      const double e1 = fx_ * std::hypot(p1.x() / p1.z() - n1[i].x, p1.y() / p1.z() - n1[i].y);
      const double e2 = fx_ * std::hypot(p2.x() / p2.z() - n2[i].x, p2.y() / p2.z() - n2[i].y);
      if (std::max(e1, e2) > max_reproj_px_) {continue;}

      const Eigen::Vector3d a = Xw - c1, b = Xw - c2;
      const double cosang = std::clamp(a.dot(b) / (a.norm() * b.norm()), -1.0, 1.0);
      if (std::acos(cosang) < min_par) {continue;}

      const double z = p2.z();
      if (z < min_depth_ || z > max_depth_) {continue;}

      double x;
      if (!sampleDA3(d, trk_pts_[i], x)) {continue;}
      out.push_back({x, toTarget(z)});
    }

    resetKeyframe(gray, *T_W_C);  // current frame becomes the next keyframe
    return out;
  }

  // ======================================================================
  // Robust fit  y = s*x (+ b)
  // ======================================================================
  bool isInlier(const Sample & p, double s, double b) const
  {
    return std::abs(s * p.x + b - p.y) <= ransac_rel_thresh_ * std::abs(p.y);
  }

  Fit robustFit(const std::vector<Sample> & smp)
  {
    Fit f;
    const int n = static_cast<int>(smp.size());
    if (n < min_samples_) {
      return f;
    }
    std::uniform_int_distribution<int> U(0, n - 1);
    int best = 0;
    double bs = 1.0, bb = 0.0;
    for (int it = 0; it < ransac_iters_; ++it) {
      double s, b = 0.0;
      if (!fit_shift_) {
        const auto & a = smp[U(rng_)];
        s = a.y / a.x;
      } else {
        const auto & a = smp[U(rng_)];
        const auto & c = smp[U(rng_)];
        if (std::abs(c.x - a.x) < 1e-9) {continue;}
        s = (c.y - a.y) / (c.x - a.x);
        b = a.y - s * a.x;
      }
      if (!std::isfinite(s) || s <= 0.0) {continue;}
      int cnt = 0;
      for (const auto & p : smp) {
        cnt += isInlier(p, s, b);
      }
      if (cnt > best) {
        best = cnt;
        bs = s;
        bb = b;
      }
    }
    if (best < min_samples_ || best < min_inlier_ratio_ * n) {
      return f;
    }

    // Least-squares refit on inliers
    double Sx = 0, Sy = 0, Sxx = 0, Sxy = 0;
    int m = 0;
    for (const auto & p : smp) {
      if (!isInlier(p, bs, bb)) {continue;}
      Sx += p.x; Sy += p.y; Sxx += p.x * p.x; Sxy += p.x * p.y; ++m;
    }
    double s = bs, b = bb;
    if (fit_shift_) {
      const double den = m * Sxx - Sx * Sx;
      if (std::abs(den) > 1e-12) {
        s = (m * Sxy - Sx * Sy) / den;
        b = (Sy - s * Sx) / m;
      }
    } else if (Sxx > 0) {
      s = Sxy / Sxx;
    }
    if (!std::isfinite(s) || s <= 0.0) {
      return f;
    }
    f.ok = true;
    f.s = s;
    f.b = b;
    f.inliers = m;
    return f;
  }

  void fuse(const Fit & fa, const Fit & fb)
  {
    double w = 0, s = 0, b = 0;
    for (const Fit * f : {&fa, &fb}) {
      if (!f->ok) {continue;}
      w += f->inliers;
      s += f->inliers * f->s;
      b += f->inliers * f->b;
    }
    if (w <= 0) {
      return;  // hold last estimate
    }
    s /= w;
    b /= w;
    if (!scale_init_) {
      s_ = s;
      b_ = b;
      scale_init_ = true;
    } else {
      s_ = (1.0 - ema_alpha_) * s_ + ema_alpha_ * s;
      b_ = (1.0 - ema_alpha_) * b_ + ema_alpha_ * b;
    }
  }

  // ======================================================================
  // Output
  // ======================================================================
  void publishDepth(const cv::Mat & d, const std_msgs::msg::Header & hdr_in)
  {
    cv::Mat out(d.size(), CV_32F);
    const bool mask_out = edge_mask_output_ && !edge_mask_.empty();
    for (int r = 0; r < d.rows; ++r) {
      const float * src = d.ptr<float>(r);
      float * dst = out.ptr<float>(r);
      for (int c = 0; c < d.cols; ++c) {
        float o = 0.f;  // 0 = invalid for nvblox
        const double x = src[c];
        const bool edge = mask_out && edge_mask_.at<uchar>(r, c);
        if (!edge && std::isfinite(x) && x > 0.0) {
          const double y = s_ * x + b_;
          const double z = da3_is_disparity_ ? (y > 0.0 ? 1.0 / y : 0.0) : y;
          if (z >= min_depth_ && z <= max_depth_) {
            o = static_cast<float>(z);
          }
        }
        dst[c] = o;
      }
    }

    std_msgs::msg::Header hdr = hdr_in;
    if (!camera_frame_.empty()) {
      hdr.frame_id = camera_frame_;
    }
    depth_pub_->publish(*cv_bridge::CvImage(hdr, "32FC1", out).toImageMsg());

    // camera_info rescaled to the depth image resolution
    CameraInfo ci = info_;
    ci.header = hdr;
    const double sx = d.cols / static_cast<double>(info_.width);
    const double sy = d.rows / static_cast<double>(info_.height);
    ci.width = d.cols;
    ci.height = d.rows;
    ci.k[0] *= sx; ci.k[2] *= sx; ci.k[4] *= sy; ci.k[5] *= sy;
    ci.p[0] *= sx; ci.p[2] *= sx; ci.p[3] *= sx; ci.p[5] *= sy; ci.p[6] *= sy;
    info_pub_->publish(ci);
  }

  void publishTf(const Eigen::Isometry3d & T_W_C, const builtin_interfaces::msg::Time & stamp)
  {
    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = stamp;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id = camera_frame_;
    const Eigen::Quaterniond q(T_W_C.linear());
    tf.transform.translation.x = T_W_C.translation().x();
    tf.transform.translation.y = T_W_C.translation().y();
    tf.transform.translation.z = T_W_C.translation().z();
    tf.transform.rotation.w = q.w();
    tf.transform.rotation.x = q.x();
    tf.transform.rotation.y = q.y();
    tf.transform.rotation.z = q.z();
    tf_br_->sendTransform(tf);
  }

  void publishStats(const Fit & fa, const Fit & fb)
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std_msgs::msg::Float64MultiArray m;
    // [fused_s, fused_b, floor_s, floor_inliers, tri_s, tri_inliers, edge_fraction]
    m.data = {scale_init_ ? s_ : nan, scale_init_ ? b_ : nan,
      fa.ok ? fa.s : nan, static_cast<double>(fa.inliers),
      fb.ok ? fb.s : nan, static_cast<double>(fb.inliers), edge_fraction_};
    stats_pub_->publish(m);
  }

  // ======================================================================
  // Members
  // ======================================================================
  // ROS
  message_filters::Subscriber<Image> img_sub_, depth_sub_;
  std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Publisher<Image>::SharedPtr depth_pub_;
  rclcpp::Publisher<CameraInfo>::SharedPtr info_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr stats_pub_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_br_;

  // Config
  std::string world_frame_, camera_frame_;
  bool publish_tf_{true}, use_floor_{true}, use_tri_{true};
  bool da3_is_disparity_{false}, fit_shift_{false};
  double mocap_time_offset_{0}, max_pose_gap_{0.05};
  double ema_alpha_{0.1}, min_depth_{0.1}, max_depth_{10.0};
  int ransac_iters_{200}, min_samples_{20};
  double ransac_rel_thresh_{0.1}, min_inlier_ratio_{0.4};
  double floor_height_{0}, floor_roi_frac_{0.5}, min_grazing_deg_{15}, max_floor_range_{6};
  double min_cam_height_{0.15};
  int floor_grid_step_{16};
  int max_features_{300}, min_tracks_{30}, max_kf_frames_{60};
  double min_feature_dist_{12}, min_baseline_{0.12}, min_parallax_deg_{2};
  double max_reproj_px_{1.5}, fb_thresh_px_{1.0};
  bool subpix_enable_{true};
  int subpix_win_{5}, subpix_iters_{30};
  double subpix_eps_{0.01};
  bool edge_enable_{true}, edge_mask_output_{true}, edge_mask_detection_{true};
  double edge_rel_grad_{0.05};
  int edge_dilate_px_{2};
  cv::Mat edge_mask_;  // CV_8U at DA3 resolution, 255 = edge/invalid
  double edge_fraction_{0.0};

  // Geometry
  Eigen::Isometry3d T_body_cam_{Eigen::Isometry3d::Identity()};
  Eigen::Vector3d world_up_{0, 0, 1};

  // Camera
  bool have_info_{false}, fisheye_{false};
  CameraInfo info_;
  cv::Mat K_, D_;
  double fx_{1.0};
  std::vector<cv::Point2f> floor_px_;
  std::vector<Eigen::Vector3d> floor_rays_;

  // Mocap buffer
  std::deque<PoseSample> poses_;

  // Method B state
  bool has_kf_{false};
  int kf_age_{0};
  cv::Mat prev_gray_;
  std::vector<cv::Point2f> kf_pts_, trk_pts_;
  Eigen::Isometry3d kf_T_W_C_{Eigen::Isometry3d::Identity()};

  // Scale state
  bool scale_init_{false};
  double s_{1.0}, b_{0.0};
  std::mt19937 rng_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Da3MetricScaler>());
  rclcpp::shutdown();
  return 0;
}
