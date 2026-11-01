// trajectory_tracker.cpp
//
// Streams a continuously re-planned trajectory (spline updated at ~5-10 Hz) to
// ArduCopter GUIDED as position + velocity + acceleration + yaw + yaw-rate setpoints
// via MAVROS setpoint_raw/local (SET_POSITION_TARGET_LOCAL_NED). The latest
// trajectory is sampled at `rate` Hz at (now + lookahead).
//
// Inputs (EKF local ENU frame = /mavros/.../local_position/pose, frame_id "map"):
//   ~/trajectory  trajectory_msgs/MultiDOFJointTrajectory  (preferred: carries the
//                 spline's own velocities/accelerations; header.stamp = t0,
//                 point time = header.stamp + time_from_start)
//   ~/path        nav_msgs/Path  (pose stamps used as times if increasing,
//                 otherwise timed by arc length at `path_speed` from receipt)
//   srv ~/enable, ~/disable (std_srvs/Trigger)
// Output:
//   <mavros_prefix>/setpoint_raw/local (mavros_msgs/PositionTarget), ~/reference, ~/status
//
// Safety: setpoints are only streamed while enabled AND the FC is connected, armed,
// in GUIDED, airborne, with a fresh EKF pose. Leaving GUIDED disables the tracker.
// Stale trajectory (> traj_timeout) -> hold position (zero velocity/accel).
// Reference outside the fence or too far from the vehicle -> hold. Velocity and
// acceleration feedforward are norm-clamped.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/extended_state.hpp>
#include <mavros_msgs/msg/position_target.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "da3_metric_scale/traj_sampler.hpp"

using namespace std::chrono_literals;
using da3_metric_scale::RefState;
using da3_metric_scale::TrajPoint;
using da3_metric_scale::Trajectory;
using da3_metric_scale::Vec3;
using da3_metric_scale::wrapPi;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using PositionTarget = mavros_msgs::msg::PositionTarget;

namespace
{
template<typename Q>
double yawOf(const Q & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
double toSec(const builtin_interfaces::msg::Time & t) {return t.sec + t.nanosec * 1e-9;}
double toSec(const builtin_interfaces::msg::Duration & d) {return d.sec + d.nanosec * 1e-9;}
Vec3 clampNorm(const Vec3 & v, double maxn)
{
  const double n = v.norm();
  return (maxn > 0 && n > maxn) ? v * (maxn / n) : v;
}
}  // namespace

class TrajectoryTracker : public rclcpp::Node
{
public:
  TrajectoryTracker()
  : Node("trajectory_tracker")
  {
    prefix_ = declare_parameter<std::string>("mavros_prefix", "/mavros/mavros");
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    rate_ = declare_parameter<double>("rate", 50.0);
    lookahead_ = declare_parameter<double>("lookahead", 0.05);
    traj_timeout_ = declare_parameter<double>("traj_timeout", 0.5);
    pose_timeout_ = declare_parameter<double>("pose_timeout", 0.5);
    max_vel_ = declare_parameter<double>("max_vel", 1.0);
    max_acc_ = declare_parameter<double>("max_acc", 2.0);
    max_yaw_rate_ = declare_parameter<double>("max_yaw_rate_deg", 90.0) * M_PI / 180.0;
    max_err_ = declare_parameter<double>("max_tracking_error", 1.0);
    send_vel_ = declare_parameter<bool>("send_velocity", true);
    send_acc_ = declare_parameter<bool>("send_acceleration", true);
    send_yaw_ = declare_parameter<bool>("send_yaw", true);
    send_yaw_rate_ = declare_parameter<bool>("send_yaw_rate", true);
    yaw_mode_ = declare_parameter<std::string>("yaw_mode", "trajectory");  // trajectory|velocity|hold
    yaw_min_speed_ = declare_parameter<double>("yaw_velocity_min_speed", 0.2);
    path_speed_ = declare_parameter<double>("path_speed", 0.5);
    stop_at_end_ = declare_parameter<bool>("stop_at_end", true);
    auto_enable_ = declare_parameter<bool>("auto_enable", false);
    z_offset_ = declare_parameter<double>("mocap_z_offset", 0.0);
    fence_min_ = declare_parameter<std::vector<double>>("fence_min", {-2.0, -2.0, 0.3});
    fence_max_ = declare_parameter<std::vector<double>>("fence_max", {2.0, 2.0, 1.8});
    if (fence_min_.size() != 3 || fence_max_.size() != 3) {
      throw std::runtime_error("fence_min / fence_max need 3 values");
    }
    if (yaw_mode_ != "trajectory" && yaw_mode_ != "velocity" && yaw_mode_ != "hold") {
      throw std::runtime_error("yaw_mode must be trajectory, velocity or hold");
    }

    sp_pub_ = create_publisher<PositionTarget>(prefix_ + "/setpoint_raw/local", 10);
    ref_pub_ = create_publisher<PoseStamped>("~/reference", 10);
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", 10);

    const auto qos = rclcpp::SensorDataQoS();
    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      prefix_ + "/state", qos,
      [this](mavros_msgs::msg::State::ConstSharedPtr m) {onState(*m);});
    ext_sub_ = create_subscription<mavros_msgs::msg::ExtendedState>(
      prefix_ + "/extended_state", qos,
      [this](mavros_msgs::msg::ExtendedState::ConstSharedPtr m) {
        landed_state_ = m->landed_state;
        last_ext_ = now();
      });
    pose_sub_ = create_subscription<PoseStamped>(
      prefix_ + "/local_position/pose", qos,
      [this](PoseStamped::ConstSharedPtr m) {pose_ = *m; last_pose_ = now();});
    traj_sub_ = create_subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      "~/trajectory", 10,
      [this](trajectory_msgs::msg::MultiDOFJointTrajectory::ConstSharedPtr m) {onTraj(*m);});
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "~/path", 10, [this](nav_msgs::msg::Path::ConstSharedPtr m) {onPath(*m);});

    en_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/enable", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        res->success = enable(res->message);
      });
    dis_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/disable", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        disable("disabled by service");
        res->success = true;
        res->message = "disabled; FC holds the last position target in GUIDED";
      });

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_), std::bind(&TrajectoryTracker::tick, this));
    status_timer_ = create_wall_timer(1s, std::bind(&TrajectoryTracker::publishStatus, this));
    RCLCPP_INFO(
      get_logger(), "trajectory_tracker ready (%s, %.0f Hz, lookahead %.3f s). Take off, switch "
      "to GUIDED, then: ros2 service call /trajectory_tracker/enable std_srvs/srv/Trigger",
      prefix_.c_str(), rate_, lookahead_);
  }

private:
  // ---------------------------------------------------------------- inputs
  void onTraj(const trajectory_msgs::msg::MultiDOFJointTrajectory & m)
  {
    if (m.points.empty()) {return;}
    checkFrame(m.header.frame_id);
    const double t0 = toSec(m.header.stamp) > 0 ? toSec(m.header.stamp) : now().seconds();
    std::vector<TrajPoint> pts;
    pts.reserve(m.points.size());
    for (const auto & pt : m.points) {
      if (pt.transforms.empty()) {continue;}
      TrajPoint q;
      q.t = t0 + toSec(pt.time_from_start);
      const auto & tr = pt.transforms[0];
      q.p = {tr.translation.x, tr.translation.y, tr.translation.z + z_offset_};
      q.yaw = yawOf(tr.rotation);
      if (!pt.velocities.empty()) {
        const auto & v = pt.velocities[0];
        q.v = {v.linear.x, v.linear.y, v.linear.z};
        q.has_v = true;
        q.yaw_rate = v.angular.z;
        q.has_yaw_rate = true;
      }
      if (!pt.accelerations.empty()) {
        const auto & a = pt.accelerations[0];
        q.a = {a.linear.x, a.linear.y, a.linear.z};
        q.has_a = true;
      }
      pts.push_back(q);
    }
    accept(std::move(pts), "trajectory");
  }

  void onPath(const nav_msgs::msg::Path & m)
  {
    if (m.poses.empty()) {return;}
    checkFrame(m.header.frame_id);
    std::vector<TrajPoint> pts;
    pts.reserve(m.poses.size());
    bool timed = true;
    for (size_t i = 0; i < m.poses.size(); ++i) {
      const auto & ps = m.poses[i];
      TrajPoint q;
      q.t = toSec(ps.header.stamp);
      q.p = {ps.pose.position.x, ps.pose.position.y, ps.pose.position.z + z_offset_};
      q.yaw = yawOf(ps.pose.orientation);
      if (q.t <= 0 || (i > 0 && q.t <= pts.back().t)) {timed = false;}
      pts.push_back(q);
    }
    if (!timed) {Trajectory::timeByArcLength(pts, path_speed_, now().seconds());}
    accept(std::move(pts), timed ? "path(timed)" : "path(arc-length)");
  }

  void accept(std::vector<TrajPoint> pts, const char * src)
  {
    std::string err;
    Trajectory t;
    if (!t.set(std::move(pts), stop_at_end_, err)) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "%s rejected: %s", src, err.c_str());
      return;
    }
    traj_ = std::move(t);
    last_traj_rx_ = now();
    RCLCPP_INFO_ONCE(get_logger(), "first %s received (%zu points)", src, traj_.size());
  }

  void checkFrame(const std::string & f)
  {
    if (!f.empty() && f != frame_id_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "input frame '%s' != '%s' - treated as '%s' (no TF)",
        f.c_str(), frame_id_.c_str(), frame_id_.c_str());
    }
  }

  // ---------------------------------------------------------------- gating
  bool poseFresh() const
  {
    return last_pose_.nanoseconds() > 0 && (now() - last_pose_).seconds() < pose_timeout_;
  }
  bool trajFresh() const
  {
    return !traj_.empty() && (now() - last_traj_rx_).seconds() < traj_timeout_;
  }
  bool onGround() const
  {
    if (last_ext_.nanoseconds() > 0 && (now() - last_ext_).seconds() < 2.0) {
      return landed_state_ == mavros_msgs::msg::ExtendedState::LANDED_STATE_ON_GROUND;
    }
    return pose_.pose.position.z - arm_z_ < 0.15;
  }
  bool ready(std::string & why) const
  {
    if (!state_.connected) {why = "FC not connected"; return false;}
    if (state_.mode != "GUIDED") {why = "not GUIDED (" + state_.mode + ")"; return false;}
    if (!state_.armed) {why = "not armed"; return false;}
    if (!poseFresh()) {why = "no fresh EKF local_position/pose"; return false;}
    if (onGround()) {why = "on the ground - take off first"; return false;}
    return true;
  }

  bool enable(std::string & msg)
  {
    if (!ready(msg)) {
      RCLCPP_WARN(get_logger(), "enable refused: %s", msg.c_str());
      return false;
    }
    enabled_ = true;
    hold_.reset();
    msg = trajFresh() ? "enabled, tracking" : "enabled, holding until a trajectory arrives";
    RCLCPP_INFO(get_logger(), "%s", msg.c_str());
    return true;
  }

  void disable(const std::string & why)
  {
    if (enabled_) {RCLCPP_WARN(get_logger(), "tracker disabled: %s", why.c_str());}
    enabled_ = false;
    hold_.reset();
  }

  void onState(const mavros_msgs::msg::State & s)
  {
    const bool was_guided = state_.mode == "GUIDED";
    if (s.armed && !state_.armed) {arm_z_ = pose_.pose.position.z;}
    state_ = s;
    if (enabled_ && (s.mode != "GUIDED" || !s.armed)) {
      disable(!s.armed ? "disarmed" : "left GUIDED (now " + s.mode + ")");
    }
    if (!was_guided && s.mode == "GUIDED" && auto_enable_ && !enabled_) {
      std::string msg;
      enable(msg);
    }
  }

  // ------------------------------------------------------------------ loop
  void tick()
  {
    if (!enabled_) {return;}
    std::string why;
    if (!ready(why)) {disable(why); return;}

    const auto & cp = pose_.pose.position;
    const Vec3 cur{cp.x, cp.y, cp.z};
    const double cur_yaw = yawOf(pose_.pose.orientation);

    if (!trajFresh()) {
      holdAt(cur, cur_yaw, traj_.empty() ? "no trajectory yet" : "trajectory stale");
      return;
    }

    RefState r = traj_.sample(now().seconds() + lookahead_);
    if (!inFence(r.p)) {
      holdAt(cur, cur_yaw, "reference outside fence (ref z " + std::to_string(r.p.z) + ", vehicle z " + std::to_string(cur.z) + ")");
      return;
    }
    if ((r.p - cur).norm() > max_err_) {
      holdAt(cur, cur_yaw, "reference too far from vehicle (" +
        std::to_string((r.p - cur).norm()) + " m)");
      return;
    }
    hold_.reset();

    r.v = clampNorm(r.v, max_vel_);
    r.a = clampNorm(r.a, max_acc_);
    if (yaw_mode_ == "velocity") {
      const double sp = std::hypot(r.v.x, r.v.y);
      if (sp > yaw_min_speed_) {
        last_cmd_yaw_ = std::atan2(r.v.y, r.v.x);
      } else if (!have_cmd_yaw_) {
        last_cmd_yaw_ = cur_yaw;
      }
      r.yaw = last_cmd_yaw_;
      r.yaw_rate = 0.0;
    } else if (yaw_mode_ == "hold") {
      if (!have_cmd_yaw_) {last_cmd_yaw_ = cur_yaw;}
      r.yaw = last_cmd_yaw_;
      r.yaw_rate = 0.0;
    }
    r.yaw_rate = std::clamp(r.yaw_rate, -max_yaw_rate_, max_yaw_rate_);
    last_cmd_yaw_ = r.yaw;
    have_cmd_yaw_ = true;
    send(r, true);
  }

  void holdAt(const Vec3 & cur, double cur_yaw, const std::string & why)
  {
    if (!hold_) {
      // hold where we are now (not the last reference) and remember it
      hold_ = RefState();
      hold_->p = cur;
      hold_->yaw = have_cmd_yaw_ ? last_cmd_yaw_ : cur_yaw;
      RCLCPP_WARN(get_logger(), "holding position: %s", why.c_str());
    } else if (why.rfind(hold_status_.substr(0, 20), 0) != 0) {
      // already holding, but for a new reason (e.g. "no trajectory yet" -> "outside fence")
      RCLCPP_WARN(get_logger(), "still holding position: %s", why.c_str());
    }
    hold_status_ = why;
    send(*hold_, false);
  }

  void send(const RefState & r, bool feedforward)
  {
    PositionTarget sp;
    sp.header.stamp = now();
    sp.header.frame_id = frame_id_;
    sp.coordinate_frame = PositionTarget::FRAME_LOCAL_NED;  // MAVROS converts ENU -> NED
    uint16_t mask = 0;
    const bool use_v = feedforward && send_vel_;
    const bool use_a = feedforward && send_acc_;
    if (!use_v) {mask |= PositionTarget::IGNORE_VX | PositionTarget::IGNORE_VY |
        PositionTarget::IGNORE_VZ;}
    if (!use_a) {mask |= PositionTarget::IGNORE_AFX | PositionTarget::IGNORE_AFY |
        PositionTarget::IGNORE_AFZ;}
    if (!send_yaw_) {mask |= PositionTarget::IGNORE_YAW;}
    if (!(feedforward && send_yaw_rate_)) {mask |= PositionTarget::IGNORE_YAW_RATE;}
    sp.type_mask = mask;
    sp.position.x = r.p.x;
    sp.position.y = r.p.y;
    sp.position.z = r.p.z;
    sp.velocity.x = use_v ? r.v.x : 0.0;
    sp.velocity.y = use_v ? r.v.y : 0.0;
    sp.velocity.z = use_v ? r.v.z : 0.0;
    sp.acceleration_or_force.x = use_a ? r.a.x : 0.0;
    sp.acceleration_or_force.y = use_a ? r.a.y : 0.0;
    sp.acceleration_or_force.z = use_a ? r.a.z : 0.0;
    sp.yaw = static_cast<float>(r.yaw);
    sp.yaw_rate = static_cast<float>(feedforward ? r.yaw_rate : 0.0);
    sp_pub_->publish(sp);

    PoseStamped ref;
    ref.header = sp.header;
    ref.pose.position.x = r.p.x;
    ref.pose.position.y = r.p.y;
    ref.pose.position.z = r.p.z;
    ref.pose.orientation.z = std::sin(r.yaw / 2);
    ref.pose.orientation.w = std::cos(r.yaw / 2);
    ref_pub_->publish(ref);
  }

  bool inFence(const Vec3 & p) const
  {
    return p.x >= fence_min_[0] && p.x <= fence_max_[0] && p.y >= fence_min_[1] &&
           p.y <= fence_max_[1] && p.z >= fence_min_[2] && p.z <= fence_max_[2];
  }

  void publishStatus()
  {
    std::string why = "ok";
    ready(why);
    std_msgs::msg::String s;
    std::string st = !enabled_ ? "DISABLED" : (hold_ ? "HOLD (" + hold_status_ + ")" : "TRACKING");
    s.data = st + " | traj " + (traj_.empty() ? std::string("none") :
      (trajFresh() ? "fresh" : "stale")) + " | mode " + state_.mode +
      (state_.armed ? " ARMED" : " disarmed") + " | " + why;
    status_pub_->publish(s);
  }

  // ---------------------------------------------------------------- members
  std::string prefix_, frame_id_, yaw_mode_;
  double rate_, lookahead_, traj_timeout_, pose_timeout_, max_vel_, max_acc_, max_yaw_rate_;
  double max_err_, yaw_min_speed_, path_speed_, z_offset_;
  bool send_vel_, send_acc_, send_yaw_, send_yaw_rate_, stop_at_end_, auto_enable_;
  std::vector<double> fence_min_, fence_max_;

  Trajectory traj_;
  bool enabled_{false};
  std::optional<RefState> hold_;
  std::string hold_status_;
  double last_cmd_yaw_{0.0};
  bool have_cmd_yaw_{false};
  double arm_z_{0.0};

  mavros_msgs::msg::State state_;
  PoseStamped pose_;
  uint8_t landed_state_{0};
  rclcpp::Time last_pose_{0, 0, RCL_ROS_TIME}, last_ext_{0, 0, RCL_ROS_TIME};
  rclcpp::Time last_traj_rx_{0, 0, RCL_ROS_TIME};

  rclcpp::Publisher<PositionTarget>::SharedPtr sp_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr ref_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<mavros_msgs::msg::ExtendedState>::SharedPtr ext_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub_;
  rclcpp::Subscription<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr en_srv_, dis_srv_;
  rclcpp::TimerBase::SharedPtr timer_, status_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TrajectoryTracker>());
  rclcpp::shutdown();
  return 0;
}
