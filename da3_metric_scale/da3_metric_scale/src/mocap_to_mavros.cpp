// mocap_to_mavros.cpp
//
// OptiTrack (vrpn_mocap / mocap4r2 PoseStamped) -> MAVROS vision_pose -> ArduPilot
// VISION_POSITION_ESTIMATE.
//
//   * Converts the mocap world frame (default Motive Y-up) to ENU, and the
//     rigid-body frame (default z fwd, x left, y up) to FLU base_link.
//     MAVROS then converts ENU/FLU -> NED/FRD itself.
//   * Optionally shifts the published position from the rigid-body pivot to
//     the flight controller, so VISO_POS_X/Y/Z can be left at 0.
//   * Rate-limits output (ArduPilot doesn't need 120 Hz over serial).
//   * Stops publishing when mocap goes stale, so the EKF times out cleanly
//     instead of fusing a frozen pose.
//   * Sets the EKF origin (SET_GPS_GLOBAL_ORIGIN) once MAVROS is connected.

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Geometry>

#include <geographic_msgs/msg/geo_point_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>

using PoseStamped = geometry_msgs::msg::PoseStamped;

namespace
{
Eigen::Matrix3d mat3(const std::vector<double> & v, const std::string & name)
{
  if (v.size() != 9) {
    throw std::runtime_error(name + " needs 9 values (row-major)");
  }
  Eigen::Matrix3d R;
  for (int i = 0; i < 9; ++i) {
    R(i / 3, i % 3) = v[i];
  }
  if (std::abs(R.determinant() - 1.0) > 1e-3 ||
    !(R * R.transpose()).isIdentity(1e-3))
  {
    throw std::runtime_error(name + " is not a proper rotation");
  }
  return R;
}
}  // namespace

class MocapToMavros : public rclcpp::Node
{
public:
  MocapToMavros()
  : Node("mocap_to_mavros")
  {
    const auto in_topic =
      declare_parameter<std::string>("mocap_pose_topic", "/vrpn_mocap/dilon/pose");
    const auto out_topic =
      declare_parameter<std::string>("vision_pose_topic", "/mavros/mavros/vision_pose/pose");
    const auto state_topic =
      declare_parameter<std::string>("mavros_state_topic", "/mavros/mavros/state");
    const auto origin_topic = declare_parameter<std::string>(
      "origin_topic", "/mavros/mavros/global_position/set_gp_origin");

    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    rate_hz_ = declare_parameter<double>("publish_rate", 50.0);
    stale_timeout_ = declare_parameter<double>("stale_timeout", 0.2);
    max_jump_ = declare_parameter<double>("max_jump", 0.5);
    use_mocap_stamp_ = declare_parameter<bool>("use_mocap_stamp", false);

    // World: rows = ENU axes expressed in mocap world. Default for Motive Y-up:
    // E = x_w, N = -z_w, U = y_w.  For Motive Z-up use identity.
    R_enu_w_ = mat3(
      declare_parameter<std::vector<double>>(
        "R_enu_world", {1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0}), "R_enu_world");
    // Body: columns = FLU axes expressed in rigid-body frame. Default for
    // z = nose, x = left, y = up (same matrix as R_body_imu in the scaler).
    R_b_flu_ = mat3(
      declare_parameter<std::vector<double>>(
        "R_body_flu", {0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0}), "R_body_flu");
    const auto p = declare_parameter<std::vector<double>>("p_body_fc", {0.0, 0.0, 0.0});
    if (p.size() != 3) {
      throw std::runtime_error("p_body_fc needs 3 values");
    }
    p_b_fc_ = Eigen::Vector3d(p[0], p[1], p[2]);

    set_origin_ = declare_parameter<bool>("set_ekf_origin", true);
    origin_lat_ = declare_parameter<double>("origin_lat", -35.363261);  // any fixed point works (ArduPilot SITL default)
    origin_lon_ = declare_parameter<double>("origin_lon", 149.165230);
    origin_alt_ = declare_parameter<double>("origin_alt", 584.0);
    origin_repeats_ = declare_parameter<int>("origin_repeats", 5);

    pub_ = create_publisher<PoseStamped>(out_topic, rclcpp::QoS(10));
    origin_pub_ = create_publisher<geographic_msgs::msg::GeoPointStamped>(
      origin_topic, rclcpp::QoS(10));

    sub_ = create_subscription<PoseStamped>(
      in_topic, rclcpp::SensorDataQoS(),
      std::bind(&MocapToMavros::onPose, this, std::placeholders::_1));
    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      state_topic, rclcpp::QoS(10),
      [this](mavros_msgs::msg::State::ConstSharedPtr s) {connected_ = s->connected;});

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz_),
      std::bind(&MocapToMavros::onTimer, this));
    origin_timer_ = create_wall_timer(
      std::chrono::seconds(1), std::bind(&MocapToMavros::onOriginTimer, this));

    RCLCPP_INFO(
      get_logger(), "mocap_to_mavros: %s -> %s @ %.0f Hz", in_topic.c_str(),
      out_topic.c_str(), rate_hz_);
  }

private:
  void onPose(const PoseStamped::ConstSharedPtr msg)
  {
    const auto & o = msg->pose.orientation;
    const Eigen::Quaterniond q_in(o.w, o.x, o.y, o.z);
    if (q_in.norm() < 0.5) {
      return;  // untracked / garbage
    }
    const Eigen::Matrix3d R_w_b = q_in.normalized().toRotationMatrix();
    const Eigen::Vector3d p_w_b(msg->pose.position.x, msg->pose.position.y,
      msg->pose.position.z);
    if (!p_w_b.allFinite()) {
      return;
    }

    // Shift to the FC and convert to ENU / FLU
    const Eigen::Vector3d p_enu = R_enu_w_ * (p_w_b + R_w_b * p_b_fc_);
    const Eigen::Quaterniond q_enu_flu(R_enu_w_ * R_w_b * R_b_flu_);

    const rclcpp::Time now = this->now();
    if (have_pose_ && max_jump_ > 0.0 &&
      (p_enu - last_p_).norm() > max_jump_ &&
      (now - last_rx_).seconds() < stale_timeout_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "mocap jump of %.2f m rejected (rigid body swap?)", (p_enu - last_p_).norm());
      return;
    }

    last_p_ = p_enu;
    last_q_ = q_enu_flu.normalized();
    last_stamp_ = msg->header.stamp;
    last_rx_ = now;
    have_pose_ = true;
    fresh_ = true;
  }

  void onTimer()
  {
    if (!have_pose_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000, "no mocap pose yet");
      return;
    }
    if ((now() - last_rx_).seconds() > stale_timeout_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, "mocap stale - not publishing");
      return;
    }
    if (!fresh_) {
      return;  // don't resend the same sample
    }
    fresh_ = false;

    PoseStamped out;
    out.header.frame_id = frame_id_;
    out.header.stamp = use_mocap_stamp_ ? last_stamp_ : builtin_interfaces::msg::Time(last_rx_);
    out.pose.position.x = last_p_.x();
    out.pose.position.y = last_p_.y();
    out.pose.position.z = last_p_.z();
    out.pose.orientation.w = last_q_.w();
    out.pose.orientation.x = last_q_.x();
    out.pose.orientation.y = last_q_.y();
    out.pose.orientation.z = last_q_.z();
    pub_->publish(out);
  }

  void onOriginTimer()
  {
    if (!set_origin_ || origin_sent_ >= origin_repeats_ || !connected_) {
      return;
    }
    geographic_msgs::msg::GeoPointStamped g;
    g.header.stamp = now();
    g.header.frame_id = "map";
    g.position.latitude = origin_lat_;
    g.position.longitude = origin_lon_;
    g.position.altitude = origin_alt_;
    origin_pub_->publish(g);
    if (++origin_sent_ == origin_repeats_) {
      RCLCPP_INFO(get_logger(), "EKF origin sent (%.5f, %.5f)", origin_lat_, origin_lon_);
    }
  }

  // ROS
  rclcpp::Publisher<PoseStamped>::SharedPtr pub_;
  rclcpp::Publisher<geographic_msgs::msg::GeoPointStamped>::SharedPtr origin_pub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::TimerBase::SharedPtr timer_, origin_timer_;

  // Config
  std::string frame_id_;
  double rate_hz_{50}, stale_timeout_{0.2}, max_jump_{0.5};
  bool use_mocap_stamp_{false};
  Eigen::Matrix3d R_enu_w_, R_b_flu_;
  Eigen::Vector3d p_b_fc_{Eigen::Vector3d::Zero()};
  bool set_origin_{true};
  double origin_lat_, origin_lon_, origin_alt_;
  int origin_repeats_{5}, origin_sent_{0};

  // State
  bool connected_{false}, have_pose_{false}, fresh_{false};
  Eigen::Vector3d last_p_{Eigen::Vector3d::Zero()};
  Eigen::Quaterniond last_q_{Eigen::Quaterniond::Identity()};
  builtin_interfaces::msg::Time last_stamp_;
  rclcpp::Time last_rx_{0, 0, RCL_ROS_TIME};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<MocapToMavros>());
  rclcpp::shutdown();
  return 0;
}
