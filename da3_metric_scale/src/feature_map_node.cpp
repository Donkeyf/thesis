// feature_map_node.cpp - map of "good features" (visual landmarks) for perception-aware yaw.
//
// For every metric depth frame from da3_metric_scaler, Shi-Tomasi corners are detected on the
// matching camera image (same stamp), lifted to 3D with the metric depth and the camera pose (TF
// map <- camera_optical at the image stamp), and merged into a persistent LandmarkMap. Landmarks
// that were re-detected several times are published as a PointCloud2 (x, y, z, score) for the
// raptor_planner yaw planner, which keeps them in view.
//
// Inputs:
//   image_topic       sensor_msgs/Image  (the image DA3 consumed, /image_da3)
//   depth_topic       sensor_msgs/Image  32FC1 metric depth (/da3_metric/depth)
//   depth_info_topic  sensor_msgs/CameraInfo of the depth (/da3_metric/camera_info)
//   kalibr_file       raw camera intrinsics + distortion (needed when the depth is rectified)
//   TF                map_frame <- depth frame at the image stamp
// Output:
//   ~/landmarks       sensor_msgs/PointCloud2 (x, y, z, score), frame map_frame

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cv_bridge/cv_bridge.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#if __has_include(<tf2_ros/buffer.hpp>)
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#else
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#endif

#include "da3_metric_scale/landmark_map.hpp"

using Image = sensor_msgs::msg::Image;
using CameraInfo = sensor_msgs::msg::CameraInfo;
using PointCloud2 = sensor_msgs::msg::PointCloud2;
using da3_metric_scale::LandmarkMap;

class FeatureMapNode : public rclcpp::Node
{
public:
  FeatureMapNode()
  : Node("feature_map")
  {
    const auto image_topic = declare_parameter<std::string>("image_topic", "/image_da3");
    const auto depth_topic = declare_parameter<std::string>("depth_topic", "/da3_metric/depth");
    const auto info_topic =
      declare_parameter<std::string>("depth_info_topic", "/da3_metric/camera_info");
    map_frame_ = declare_parameter<std::string>("map_frame", "map");
    max_corners_ = declare_parameter<int>("max_corners", 150);
    quality_level_ = declare_parameter<double>("quality_level", 0.01);
    min_distance_ = declare_parameter<double>("min_distance_px", 15.0);
    block_size_ = declare_parameter<int>("block_size", 3);
    border_ = declare_parameter<int>("border_px", 10);
    min_depth_ = declare_parameter<double>("min_depth", 0.3);
    max_depth_ = declare_parameter<double>("max_depth", 4.0);
    depth_jump_ = declare_parameter<double>("max_depth_spread", 0.05);   // relative, in 3x3
    tf_timeout_ = declare_parameter<double>("tf_timeout", 0.05);
    max_stamp_diff_ = declare_parameter<double>("max_stamp_diff", 0.02);   // s, image <-> depth
    const double rate = declare_parameter<double>("publish_rate", 2.0);
    auto & lp = lmap_.params();
    lp.voxel = declare_parameter<double>("landmark_voxel", 0.10);
    lp.min_obs = declare_parameter<int>("min_observations", 3);
    lp.n_full = declare_parameter<int>("full_score_observations", 10);
    lp.forget_time = declare_parameter<double>("forget_time", 120.0);
    lp.max_landmarks = declare_parameter<int>("max_landmarks", 2000);

    const auto kalibr_file = declare_parameter<std::string>("kalibr_file", "");
    const auto kalibr_cam = declare_parameter<std::string>("kalibr_camera", "cam0");
    if (!kalibr_file.empty()) {loadKalibr(kalibr_file, kalibr_cam);}

    tf_buf_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buf_);

    cloud_pub_ = create_publisher<PointCloud2>("~/landmarks", rclcpp::QoS(1));
    img_sub_ = create_subscription<Image>(
      image_topic, rclcpp::SensorDataQoS(), [this](Image::ConstSharedPtr m) {
        imgs_.push_back(m);
        while (imgs_.size() > 15) {imgs_.pop_front();}
      });
    info_sub_ = create_subscription<CameraInfo>(
      info_topic, rclcpp::SensorDataQoS(), [this](CameraInfo::ConstSharedPtr m) {info_ = m;});
    depth_sub_ = create_subscription<Image>(
      depth_topic, rclcpp::SensorDataQoS(), [this](Image::ConstSharedPtr m) {onDepth(m);});
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / std::max(0.1, rate)), [this]() {publish();});
    RCLCPP_INFO(
      get_logger(), "feature_map: corners on %s, depth %s -> %s/landmarks (frame %s)%s",
      image_topic.c_str(), depth_topic.c_str(), get_fully_qualified_name(), map_frame_.c_str(),
      have_raw_ ? ", raw intrinsics from kalibr" : "");
  }

private:
  void loadKalibr(const std::string & path, const std::string & cam)
  {
    cv::FileStorage fs(path, cv::FileStorage::READ);
    if (!fs.isOpened()) {throw std::runtime_error("cannot open kalibr_file " + path);}
    const cv::FileNode c = fs[cam];
    std::vector<double> in, dc;
    for (const auto & e : c["intrinsics"]) {in.push_back(static_cast<double>(e));}
    for (const auto & e : c["distortion_coeffs"]) {dc.push_back(static_cast<double>(e));}
    if (in.size() != 4) {throw std::runtime_error("kalibr_file: need intrinsics[4]");}
    raw_K_ = (cv::Mat_<double>(3, 3) << in[0], 0, in[2], 0, in[1], in[3], 0, 0, 1);
    raw_D_ = cv::Mat(dc, true).reshape(1, 1);
    have_raw_ = true;
  }

  void onDepth(const Image::ConstSharedPtr & dmsg)
  {
    if (!info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for depth camera_info");
      return;
    }
    // camera image the depth was computed from (same stamp; nearest within max_stamp_diff)
    Image::ConstSharedPtr img;
    const double td = rclcpp::Time(dmsg->header.stamp).seconds();
    double best = max_stamp_diff_;
    for (const auto & im : imgs_) {
      const double e = std::fabs(rclcpp::Time(im->header.stamp).seconds() - td);
      if (e <= best) {best = e; img = im;}
    }
    if (!img) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
        "no camera image within %.0f ms of the depth stamp", 1e3 * max_stamp_diff_);
      return;
    }
    Eigen::Isometry3d T_map_cam;
    try {
      const auto tf = tf_buf_->lookupTransform(
        map_frame_, dmsg->header.frame_id, rclcpp::Time(dmsg->header.stamp),
        rclcpp::Duration::from_seconds(tf_timeout_));
      const auto & q = tf.transform.rotation;
      const auto & t = tf.transform.translation;
      T_map_cam = Eigen::Isometry3d::Identity();
      T_map_cam.linear() = Eigen::Quaterniond(q.w, q.x, q.y, q.z).normalized().toRotationMatrix();
      T_map_cam.translation() = Eigen::Vector3d(t.x, t.y, t.z);
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "TF %s <- %s: %s",
        map_frame_.c_str(), dmsg->header.frame_id.c_str(), e.what());
      return;
    }

    cv::Mat gray, depth;
    try {
      gray = cv_bridge::toCvShare(img, "mono8")->image;
      depth = cv_bridge::toCvShare(dmsg)->image;
    } catch (const std::exception & e) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge: %s", e.what());
      return;
    }
    if (depth.type() != CV_32FC1) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "depth must be 32FC1");
      return;
    }

    // Shi-Tomasi corners + their response, normalised by the strongest corner of the frame
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(0));
    mask(cv::Rect(border_, border_, gray.cols - 2 * border_, gray.rows - 2 * border_)).setTo(255);
    std::vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(gray, corners, max_corners_, quality_level_, min_distance_, mask,
      block_size_, false);
    if (corners.empty()) {return;}
    cv::Mat eig;
    cv::cornerMinEigenVal(gray, eig, block_size_, 3);
    double emax = 0;
    auto resp = [&eig](const cv::Point2f & c) {
        return static_cast<double>(eig.at<float>(cvRound(c.y), cvRound(c.x)));
      };
    for (const auto & c : corners) {emax = std::max(emax, resp(c));}

    // pixel in the depth image + normalised ray for every corner
    const bool rectified = std::all_of(info_->d.begin(), info_->d.end(),
        [](double v) {return std::fabs(v) < 1e-12;});
    const double fx = info_->k[0], fy = info_->k[4], cx = info_->k[2], cy = info_->k[5];
    std::vector<cv::Point2f> norm;
    if (rectified && have_raw_) {
      cv::undistortPoints(corners, norm, raw_K_, raw_D_);     // raw pixel -> normalised
    } else {
      // depth in the raw image geometry: undistort with the depth camera model
      cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
      cv::Mat D(info_->d, true);
      const double sx = static_cast<double>(depth.cols) / gray.cols;
      const double sy = static_cast<double>(depth.rows) / gray.rows;
      std::vector<cv::Point2f> scaled;
      for (const auto & c : corners) {scaled.emplace_back(c.x * sx, c.y * sy);}
      cv::undistortPoints(scaled, norm, K, D.reshape(1, 1));
      if (rectified && !have_raw_) {
        RCLCPP_WARN_ONCE(get_logger(), "depth is rectified but kalibr_file is not set: corners "
          "are not undistorted (set kalibr_file to the same file as da3_metric_scaler)");
      }
    }

    const double t = rclcpp::Time(dmsg->header.stamp).seconds();
    int added = 0;
    for (size_t i = 0; i < corners.size(); ++i) {
      const double xn = norm[i].x, yn = norm[i].y;
      const int u = static_cast<int>(std::lround(fx * xn + cx));
      const int v = static_cast<int>(std::lround(fy * yn + cy));
      if (u < 1 || v < 1 || u >= depth.cols - 1 || v >= depth.rows - 1) {continue;}
      std::vector<float> zs;
      for (int dv = -1; dv <= 1; ++dv) {
        for (int du = -1; du <= 1; ++du) {
          const float z = depth.at<float>(v + dv, u + du);
          if (std::isfinite(z) && z > 0) {zs.push_back(z);}
        }
      }
      if (zs.size() < 5) {continue;}                    // edge-masked or invalid depth
      std::sort(zs.begin(), zs.end());
      const double z = zs[zs.size() / 2];
      if (z < min_depth_ || z > max_depth_) {continue;}
      if (zs.back() - zs.front() > depth_jump_ * z) {continue;}   // depth discontinuity
      const Eigen::Vector3d pc(z * xn, z * yn, z);
      const double q = emax > 0 ? resp(corners[i]) / emax : 0.0;
      lmap_.insert(T_map_cam * pc, std::min(1.0, std::max(0.0, q)), t);
      ++added;
    }
    last_t_ = t;
    ++frames_;
    added_ += added;
  }

  void publish()
  {
    if (last_t_ > 0) {lmap_.prune(last_t_);}
    const auto lm = lmap_.exportLandmarks();
    PointCloud2 pc;
    pc.header.stamp = now();
    pc.header.frame_id = map_frame_;
    pc.height = 1;
    pc.width = static_cast<uint32_t>(lm.size());
    pc.is_bigendian = false;
    pc.is_dense = true;
    const char * names[4] = {"x", "y", "z", "score"};
    for (uint32_t i = 0; i < 4; ++i) {
      sensor_msgs::msg::PointField f;
      f.name = names[i];
      f.offset = 4 * i;
      f.datatype = sensor_msgs::msg::PointField::FLOAT32;
      f.count = 1;
      pc.fields.push_back(f);
    }
    pc.point_step = 16;
    pc.row_step = pc.point_step * pc.width;
    pc.data.resize(pc.row_step);
    for (size_t i = 0; i < lm.size(); ++i) {
      const float v[4] = {static_cast<float>(lm[i].p.x()), static_cast<float>(lm[i].p.y()),
        static_cast<float>(lm[i].p.z()), static_cast<float>(lm[i].score)};
      std::memcpy(&pc.data[i * 16], v, 16);
    }
    cloud_pub_->publish(pc);
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 10000,
      "%d frames, %d corner observations, %zu landmarks stored, %zu published",
      frames_, added_, lmap_.size(), lm.size());
  }

  std::string map_frame_;
  int max_corners_, block_size_, border_;
  double quality_level_, min_distance_, min_depth_, max_depth_, depth_jump_, tf_timeout_;
  double max_stamp_diff_{0.02};
  bool have_raw_{false};
  cv::Mat raw_K_, raw_D_;
  LandmarkMap lmap_;
  double last_t_{0};
  int frames_{0}, added_{0};
  std::deque<Image::ConstSharedPtr> imgs_;
  CameraInfo::ConstSharedPtr info_;
  std::shared_ptr<tf2_ros::Buffer> tf_buf_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Subscription<Image>::SharedPtr img_sub_, depth_sub_;
  rclcpp::Subscription<CameraInfo>::SharedPtr info_sub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<FeatureMapNode>());
  rclcpp::shutdown();
  return 0;
}
