// guided_waypoints.cpp
//
// Sends position waypoints to ArduCopter through MAVROS (setpoint_position/local),
// but ONLY while the FC is connected, armed and in GUIDED mode. It never changes
// mode or arms by itself: the pilot arms and switches to GUIDED on the RC, then
// calls ~/start.
//
// Waypoints are in the EKF local ENU frame (= /mavros/.../local_position/pose):
// x = mocap x (east), y = mocap y (north), z up, yaw 0 = facing +x, CCW positive.
// Use `mocap_z_offset` if EKF z and mocap z differ by a constant (EKF = mocap + offset).
//
// Inputs
//   param  waypoints            [x, y, z, yaw_deg, x, y, z, yaw_deg, ...]
//   topic  ~/path   (nav_msgs/Path)              replace the mission
//   topic  ~/goto   (geometry_msgs/PoseStamped)  replace the mission with one point
//   srv    ~/start  (std_srvs/Trigger)           start / resume (checks GUIDED + armed)
//   srv    ~/stop   (std_srvs/Trigger)           stop sending, go idle (FC holds in GUIDED)
// Output
//   <mavros_prefix>/setpoint_position/local, ~/status (std_msgs/String)
//
// Behaviour
//   * leaves GUIDED / disarms / loses pose  -> stops sending, mission paused
//   * optional takeoff (takeoff_alt > 0) via <prefix>/cmd/takeoff if still on the ground
//   * a waypoint counts as reached inside acceptance_radius and yaw_tolerance for hold_time
//   * after the last waypoint it keeps holding it (or loops if loop:=true)
//   * every waypoint is checked against a fence box before the mission is accepted

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <mavros_msgs/msg/extended_state.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <mavros_msgs/srv/command_tol.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>

using namespace std::chrono_literals;
using PoseStamped = geometry_msgs::msg::PoseStamped;

namespace
{
constexpr double kDeg = M_PI / 180.0;

double wrapPi(double a)
{
  while (a > M_PI) {a -= 2 * M_PI;}
  while (a < -M_PI) {a += 2 * M_PI;}
  return a;
}

double yawFromQuat(const geometry_msgs::msg::Quaternion & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}

struct Waypoint
{
  double x, y, z, yaw;  // ENU, yaw in rad
};
}  // namespace

class GuidedWaypoints : public rclcpp::Node
{
public:
  enum class Phase { IDLE, TAKEOFF, RUN, HOLD_LAST, PAUSED };

  GuidedWaypoints()
  : Node("guided_waypoints")
  {
    prefix_ = declare_parameter<std::string>("mavros_prefix", "/mavros/mavros");
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    rate_hz_ = declare_parameter<double>("setpoint_rate", 20.0);
    accept_radius_ = declare_parameter<double>("acceptance_radius", 0.15);
    yaw_tol_ = declare_parameter<double>("yaw_tolerance_deg", 10.0) * kDeg;
    hold_time_ = declare_parameter<double>("hold_time", 1.0);
    loop_ = declare_parameter<bool>("loop", false);
    auto_start_ = declare_parameter<bool>("auto_start", false);
    resume_on_guided_ = declare_parameter<bool>("resume_on_guided", false);
    pose_timeout_ = declare_parameter<double>("pose_timeout", 0.5);
    takeoff_alt_ = declare_parameter<double>("takeoff_alt", 0.0);   // m, 0 = no takeoff cmd
    takeoff_tol_ = declare_parameter<double>("takeoff_tolerance", 0.10);
    takeoff_timeout_ = declare_parameter<double>("takeoff_timeout", 20.0);
    z_offset_ = declare_parameter<double>("mocap_z_offset", 0.0);
    fence_min_ = declare_parameter<std::vector<double>>("fence_min", {-2.0, -2.0, 0.2});
    fence_max_ = declare_parameter<std::vector<double>>("fence_max", {2.0, 2.0, 1.8});
    const auto wp = declare_parameter<std::vector<double>>("waypoints", std::vector<double>{});

    if (fence_min_.size() != 3 || fence_max_.size() != 3) {
      throw std::runtime_error("fence_min / fence_max need 3 values");
    }
    if (!wp.empty()) {
      if (wp.size() % 4 != 0) {
        throw std::runtime_error("waypoints must be groups of 4: x y z yaw_deg");
      }
      std::vector<Waypoint> m;
      for (size_t i = 0; i < wp.size(); i += 4) {
        m.push_back({wp[i], wp[i + 1], wp[i + 2] + z_offset_, wp[i + 3] * kDeg});
      }
      setMission(m, "param");
    }

    sp_pub_ = create_publisher<PoseStamped>(prefix_ + "/setpoint_position/local", 10);
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
      [this](PoseStamped::ConstSharedPtr m) {
        pose_ = *m;
        last_pose_ = now();
      });
    path_sub_ = create_subscription<nav_msgs::msg::Path>(
      "~/path", 10, [this](nav_msgs::msg::Path::ConstSharedPtr m) {onPath(*m);});
    goto_sub_ = create_subscription<PoseStamped>(
      "~/goto", 10, [this](PoseStamped::ConstSharedPtr m) {
        nav_msgs::msg::Path p;
        p.poses.push_back(*m);
        onPath(p);
      });

    start_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/start", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        std::string why;
        res->success = start(why);
        res->message = why;
      });
    stop_srv_ = create_service<std_srvs::srv::Trigger>(
      "~/stop", [this](const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res) {
        setPhase(Phase::IDLE, "stopped by service");
        res->success = true;
        res->message = "idle; FC holds position in GUIDED";
      });
    takeoff_cli_ = create_client<mavros_msgs::srv::CommandTOL>(prefix_ + "/cmd/takeoff");

    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_hz_), std::bind(&GuidedWaypoints::tick, this));
    status_timer_ = create_wall_timer(1s, std::bind(&GuidedWaypoints::publishStatus, this));

    RCLCPP_INFO(
      get_logger(), "guided_waypoints ready: %zu waypoints, prefix %s. Arm, switch to GUIDED, "
      "then: ros2 service call /guided_waypoints/start std_srvs/srv/Trigger",
      mission_.size(), prefix_.c_str());
  }

private:
  // ------------------------------------------------------------------ gating
  bool poseFresh() const
  {
    return last_pose_.nanoseconds() > 0 && (now() - last_pose_).seconds() < pose_timeout_;
  }

  bool readyToFly(std::string & why) const
  {
    if (!state_.connected) {why = "FC not connected"; return false;}
    if (state_.mode != "GUIDED") {why = "FC not in GUIDED (mode " + state_.mode + ")"; return false;}
    if (!state_.armed) {why = "FC not armed"; return false;}
    if (!poseFresh()) {why = "no fresh local_position/pose from EKF"; return false;}
    return true;
  }

  bool onGround() const
  {
    // Prefer EXTENDED_SYS_STATE; fall back to height above where we armed
    if (last_ext_.nanoseconds() > 0 && (now() - last_ext_).seconds() < 2.0) {
      return landed_state_ == mavros_msgs::msg::ExtendedState::LANDED_STATE_ON_GROUND;
    }
    return pose_.pose.position.z - arm_z_ < 0.15;
  }

  // ---------------------------------------------------------------- mission
  bool inFence(const Waypoint & w) const
  {
    return w.x >= fence_min_[0] && w.x <= fence_max_[0] &&
           w.y >= fence_min_[1] && w.y <= fence_max_[1] &&
           w.z >= fence_min_[2] && w.z <= fence_max_[2];
  }

  bool setMission(const std::vector<Waypoint> & m, const std::string & src)
  {
    if (m.empty()) {
      RCLCPP_WARN(get_logger(), "empty mission from %s ignored", src.c_str());
      return false;
    }
    for (size_t i = 0; i < m.size(); ++i) {
      if (!inFence(m[i])) {
        RCLCPP_ERROR(
          get_logger(), "mission from %s REJECTED: waypoint %zu (%.2f, %.2f, %.2f) outside "
          "fence [%.1f..%.1f, %.1f..%.1f, %.1f..%.1f]", src.c_str(), i, m[i].x, m[i].y,
          m[i].z, fence_min_[0], fence_max_[0], fence_min_[1], fence_max_[1], fence_min_[2],
          fence_max_[2]);
        return false;
      }
    }
    mission_ = m;
    idx_ = 0;
    reached_since_.reset();
    RCLCPP_INFO(get_logger(), "mission from %s: %zu waypoints", src.c_str(), m.size());
    for (size_t i = 0; i < m.size(); ++i) {
      RCLCPP_INFO(
        get_logger(), "  [%zu] x=%.2f y=%.2f z=%.2f yaw=%.0f deg", i, m[i].x, m[i].y, m[i].z,
        m[i].yaw / kDeg);
    }
    return true;
  }

  void onPath(const nav_msgs::msg::Path & p)
  {
    std::vector<Waypoint> m;
    for (const auto & ps : p.poses) {
      if (!ps.header.frame_id.empty() && ps.header.frame_id != frame_id_) {
        RCLCPP_WARN(
          get_logger(), "pose frame '%s' != '%s'; using it as %s anyway",
          ps.header.frame_id.c_str(), frame_id_.c_str(), frame_id_.c_str());
      }
      m.push_back({ps.pose.position.x, ps.pose.position.y, ps.pose.position.z + z_offset_,
          yawFromQuat(ps.pose.orientation)});
    }
    const bool running = phase_ == Phase::RUN || phase_ == Phase::HOLD_LAST;
    if (setMission(m, "topic") && running) {
      phase_ = Phase::RUN;  // continue flying the new mission from waypoint 0
    }
  }

  bool start(std::string & why)
  {
    if (mission_.empty()) {why = "no waypoints loaded"; return false;}
    if (!readyToFly(why)) {
      RCLCPP_WARN(get_logger(), "start refused: %s", why.c_str());
      return false;
    }
    if (phase_ == Phase::RUN || phase_ == Phase::TAKEOFF) {
      why = "already running";
      return true;
    }
    if (takeoff_alt_ > 0.0 && onGround()) {
      if (!requestTakeoff(why)) {return false;}
      takeoff_target_z_ = arm_z_ + takeoff_alt_;
      takeoff_started_ = now();
      setPhase(Phase::TAKEOFF, "taking off");
      why = "takeoff requested";
      return true;
    }
    if (onGround()) {
      why = "vehicle is on the ground: take off first (or set takeoff_alt > 0)";
      RCLCPP_WARN(get_logger(), "start refused: %s", why.c_str());
      return false;
    }
    if (phase_ == Phase::HOLD_LAST || idx_ >= mission_.size()) {idx_ = 0;}
    reached_since_.reset();
    setPhase(Phase::RUN, "running from waypoint " + std::to_string(idx_));
    why = "running";
    return true;
  }

  bool requestTakeoff(std::string & why)
  {
    if (!takeoff_cli_->service_is_ready()) {
      why = "takeoff service " + prefix_ + "/cmd/takeoff not available";
      RCLCPP_WARN(get_logger(), "%s", why.c_str());
      return false;
    }
    auto req = std::make_shared<mavros_msgs::srv::CommandTOL::Request>();
    req->altitude = static_cast<float>(takeoff_alt_);
    req->yaw = std::numeric_limits<float>::quiet_NaN();
    takeoff_cli_->async_send_request(
      req, [this](rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedFuture f) {
        const auto r = f.get();
        if (!r->success) {
          RCLCPP_ERROR(get_logger(), "FC rejected takeoff (result %u)", r->result);
          setPhase(Phase::IDLE, "takeoff rejected");
        }
      });
    return true;
  }

  // ----------------------------------------------------------------- state
  void onState(const mavros_msgs::msg::State & s)
  {
    const bool was_guided = state_.mode == "GUIDED";
    const bool was_armed = state_.armed;
    state_ = s;
    if (s.armed && !was_armed) {
      arm_z_ = pose_.pose.position.z;  // ground reference for takeoff / landed fallback
    }
    const bool flying = phase_ == Phase::RUN || phase_ == Phase::TAKEOFF ||
      phase_ == Phase::HOLD_LAST;
    if (!s.armed && was_armed) {
      setPhase(Phase::IDLE, "disarmed");
      idx_ = 0;
      return;
    }
    if (flying && s.mode != "GUIDED") {
      setPhase(Phase::PAUSED, "left GUIDED (now " + s.mode + "); mission paused at waypoint " +
        std::to_string(idx_));
      return;
    }
    if (!was_guided && s.mode == "GUIDED") {
      RCLCPP_INFO(get_logger(), "FC entered GUIDED");
      if ((phase_ == Phase::PAUSED && resume_on_guided_) ||
        (phase_ == Phase::IDLE && auto_start_))
      {
        std::string why;
        start(why);
      }
    }
  }

  void setPhase(Phase p, const std::string & why)
  {
    if (p != phase_) {
      RCLCPP_INFO(get_logger(), "[%s -> %s] %s", name(phase_), name(p), why.c_str());
    }
    phase_ = p;
  }

  static const char * name(Phase p)
  {
    switch (p) {
      case Phase::IDLE: return "IDLE";
      case Phase::TAKEOFF: return "TAKEOFF";
      case Phase::RUN: return "RUN";
      case Phase::HOLD_LAST: return "HOLD_LAST";
      case Phase::PAUSED: return "PAUSED";
    }
    return "?";
  }

  // ------------------------------------------------------------------ loop
  void tick()
  {
    if (phase_ == Phase::IDLE || phase_ == Phase::PAUSED) {return;}

    std::string why;
    if (!readyToFly(why)) {
      setPhase(Phase::PAUSED, why);
      return;  // never send setpoints unless GUIDED + armed + fresh pose
    }

    if (phase_ == Phase::TAKEOFF) {
      if ((now() - takeoff_started_).seconds() > takeoff_timeout_) {
        setPhase(Phase::PAUSED, "takeoff did not reach target altitude in time");
        return;
      }
      if (std::abs(pose_.pose.position.z - takeoff_target_z_) < takeoff_tol_) {
        idx_ = 0;
        reached_since_.reset();
        setPhase(Phase::RUN, "takeoff complete");
      }
      return;  // FC is executing the takeoff command; don't fight it with setpoints
    }

    const Waypoint & w = mission_[std::min(idx_, mission_.size() - 1)];
    publishSetpoint(w);
    if (phase_ == Phase::HOLD_LAST) {return;}

    const auto & p = pose_.pose.position;
    const double d = std::sqrt(
      (p.x - w.x) * (p.x - w.x) + (p.y - w.y) * (p.y - w.y) + (p.z - w.z) * (p.z - w.z));
    const double dyaw = std::abs(wrapPi(yawFromQuat(pose_.pose.orientation) - w.yaw));
    if (d < accept_radius_ && dyaw < yaw_tol_) {
      if (!reached_since_) {
        reached_since_ = now();
        RCLCPP_INFO(get_logger(), "reached waypoint %zu (err %.2f m), holding %.1f s", idx_, d,
          hold_time_);
      } else if ((now() - *reached_since_).seconds() >= hold_time_) {
        reached_since_.reset();
        if (idx_ + 1 < mission_.size()) {
          ++idx_;
          RCLCPP_INFO(get_logger(), "-> waypoint %zu", idx_);
        } else if (loop_) {
          idx_ = 0;
          RCLCPP_INFO(get_logger(), "mission complete, looping");
        } else {
          setPhase(Phase::HOLD_LAST, "mission complete, holding last waypoint");
        }
      }
    } else {
      reached_since_.reset();
    }
  }

  void publishSetpoint(const Waypoint & w)
  {
    PoseStamped sp;
    sp.header.stamp = now();
    sp.header.frame_id = frame_id_;
    sp.pose.position.x = w.x;
    sp.pose.position.y = w.y;
    sp.pose.position.z = w.z;
    sp.pose.orientation.z = std::sin(w.yaw / 2.0);
    sp.pose.orientation.w = std::cos(w.yaw / 2.0);
    sp_pub_->publish(sp);
  }

  void publishStatus()
  {
    std_msgs::msg::String s;
    std::string why = "ok";
    readyToFly(why);
    s.data = std::string(name(phase_)) + " wp " + std::to_string(idx_) + "/" +
      std::to_string(mission_.size()) + " | mode " + state_.mode +
      (state_.armed ? " ARMED" : " disarmed") + " | " + why;
    status_pub_->publish(s);
  }

  // ---------------------------------------------------------------- members
  std::string prefix_, frame_id_;
  double rate_hz_, accept_radius_, yaw_tol_, hold_time_, pose_timeout_;
  double takeoff_alt_, takeoff_tol_, takeoff_timeout_, z_offset_;
  bool loop_, auto_start_, resume_on_guided_;
  std::vector<double> fence_min_, fence_max_;

  std::vector<Waypoint> mission_;
  size_t idx_{0};
  Phase phase_{Phase::IDLE};
  std::optional<rclcpp::Time> reached_since_;
  double arm_z_{0.0}, takeoff_target_z_{0.0};
  rclcpp::Time takeoff_started_{0, 0, RCL_ROS_TIME};

  mavros_msgs::msg::State state_;
  PoseStamped pose_;
  rclcpp::Time last_pose_{0, 0, RCL_ROS_TIME}, last_ext_{0, 0, RCL_ROS_TIME};
  uint8_t landed_state_{0};

  rclcpp::Publisher<PoseStamped>::SharedPtr sp_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Subscription<mavros_msgs::msg::ExtendedState>::SharedPtr ext_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub_, goto_sub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr path_sub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr start_srv_, stop_srv_;
  rclcpp::Client<mavros_msgs::srv::CommandTOL>::SharedPtr takeoff_cli_;
  rclcpp::TimerBase::SharedPtr timer_, status_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<GuidedWaypoints>());
  rclcpp::shutdown();
  return 0;
}
