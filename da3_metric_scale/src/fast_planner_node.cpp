// fast_planner_node - Fast-Planner (Zhou et al., RA-L 2019) on the nvblox ESDF, feeding
// trajectory_tracker (ArduCopter GUIDED via MAVROS).
//
// Built twice (CMakeLists.txt):
//   fast_planner    plain Fast-Planner
//   raptor_planner  (-DRAPTOR_NODE) + RAPTOR Sec. VI risk-aware trajectory refinement (Zhou et
//                   al., T-RO 2021): if the trajectory enters unknown space at p_f and p_f only
//                   becomes reliably visible too late to stop (v^2/2a_max > d - R_q), the curve
//                   is re-optimised so that p_f is viewed earlier with a safe reaction distance.
//
//   ESDF  /esdf_publisher/esdf_grid (da3_metric_scale/EsdfGrid, frame map)
//   odom  <mavros_prefix>/local_position/pose + velocity_local (EKF ENU, frame map)
//   FC    <mavros_prefix>/state  (plans/executes only while connected + armed + GUIDED)
//   goal  ~/goal (PoseStamped) or RViz "2D Goal Pose" (/goal_pose, z = default_goal_z)
//   mission  mission_goals (x, y, z, yaw_deg groups) flown in order; starts automatically when
//         the FC enters GUIDED (mission_auto_start) or via ~/start_mission. On GUIDED the planner
//         also enables trajectory_tracker itself (auto_enable_tracker -> /trajectory_tracker/enable)
//   out   /trajectory_tracker/trajectory (MultiDOFJointTrajectory: pos, vel, acc, yaw, yaw rate
//         every output_dt; republished at publish_rate so the tracker never goes stale)
//
// Pipeline per plan: kinodynamic A* (front end, ESDF clearance) -> uniform cubic B-spline fit ->
// NLopt optimisation (smoothness + ESDF collision + feasibility) -> non-uniform time adjustment ->
// dense collision check. Replanning: receding horizon every replan_period from the active
// trajectory's state at now + replan_lookahead (so pos/vel/acc stay continuous), plus an
// immediate replan when the remaining trajectory hits newly mapped obstacles, and an emergency
// hold when that hit is closer than emergency_time.
#include <chrono>
#include <cmath>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <mavros_msgs/msg/state.hpp>
#include <nav_msgs/msg/path.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <trajectory_msgs/msg/multi_dof_joint_trajectory.hpp>

#include "da3_metric_scale/fast_planner_core.hpp"
#include "da3_metric_scale/msg/esdf_grid.hpp"

using namespace std::chrono_literals;
using da3_metric_scale::EsdfMap;
using da3_metric_scale::FastPlannerCore;
using da3_metric_scale::NonUniformBspline;
using da3_metric_scale::PlannerParams;
using da3_metric_scale::PlanResult;
using V3 = Eigen::Vector3d;
using PoseStamped = geometry_msgs::msg::PoseStamped;
using Trigger = std_srvs::srv::Trigger;

namespace
{
template<typename Q>
double yawOf(const Q & q)
{
  return std::atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z));
}
double wrapPi(double a) {return std::atan2(std::sin(a), std::cos(a));}
builtin_interfaces::msg::Duration toDuration(double s)
{
  builtin_interfaces::msg::Duration d;
  d.sec = static_cast<int32_t>(std::floor(s));
  d.nanosec = static_cast<uint32_t>((s - std::floor(s)) * 1e9);
  return d;
}

struct Sample
{
  V3 p{V3::Zero()}, v{V3::Zero()}, a{V3::Zero()};
  double yaw{0}, yaw_rate{0};
};

// One executable trajectory (B-spline + yaw profile), optionally stitched after the previous
// one: before t0 it samples `prev`, so the published message is continuous across replans.
struct ActiveTraj
{
  double t0{0};                 // absolute time [s] of curve t = 0
  NonUniformBspline pos, vel, acc;
  double duration{0};
  bool to_goal{false};
  bool is_hold{false};
  std::vector<double> yaw;      // yaw samples every yaw_dt from t0
  double yaw_dt{0.05};
  std::shared_ptr<const ActiveTraj> prev;

  double tEndAbs() const {return t0 + duration;}

  Sample sample(double t_abs) const
  {
    if (t_abs < t0 && prev) {return prev->sample(t_abs);}
    Sample s;
    const double tau = std::min(std::max(t_abs - t0, 0.0), duration);
    s.p = pos.at(tau);
    if (!is_hold && t_abs - t0 < duration) {
      s.v = vel.at(tau);
      s.a = acc.at(tau);
    }
    if (!yaw.empty()) {
      const double f = tau / yaw_dt;
      const size_t i = std::min(static_cast<size_t>(f), yaw.size() - 1);
      const size_t j = std::min(i + 1, yaw.size() - 1);
      const double w = std::min(1.0, f - i);
      s.yaw = yaw[i] + w * (yaw[j] - yaw[i]);
      s.yaw_rate = (j > i && t_abs - t0 < duration) ? (yaw[j] - yaw[i]) / yaw_dt : 0.0;
    }
    return s;
  }
};

// stationary trajectory at p (cubic with all control points equal)
std::shared_ptr<ActiveTraj> makeHold(const V3 & p, double yaw, double t0)
{
  auto h = std::make_shared<ActiveTraj>();
  Eigen::MatrixXd c(4, 3);
  for (int i = 0; i < 4; ++i) {c.row(i) = p.transpose();}
  h->pos = NonUniformBspline(c, 3, 0.5);
  h->vel = h->pos.derivative();
  h->acc = h->vel.derivative();
  h->duration = h->pos.duration();
  h->t0 = t0;
  h->is_hold = true;
  h->yaw = {yaw, yaw};
  h->yaw_dt = h->duration;
  return h;
}
}  // namespace

#ifdef RAPTOR_NODE
static constexpr const char * kNodeName = "raptor_planner";
static constexpr bool kRiskDefault = true;
#else
static constexpr const char * kNodeName = "fast_planner";
static constexpr bool kRiskDefault = false;
#endif

class FastPlannerNode : public rclcpp::Node
{
  enum class State {WAIT_GOAL, WAIT_READY, GEN_NEW, EXEC};

public:
  FastPlannerNode()
  : Node(kNodeName)
  {
    prefix_ = declare_parameter<std::string>("mavros_prefix", "/mavros/mavros");
    frame_id_ = declare_parameter<std::string>("frame_id", "map");
    require_guided_ = declare_parameter<bool>("require_guided", true);
    exec_rate_ = declare_parameter<double>("exec_rate", 20.0);
    publish_rate_ = declare_parameter<double>("publish_rate", 10.0);
    replan_period_ = declare_parameter<double>("replan_period", 1.0);
    replan_lookahead_ = declare_parameter<double>("replan_lookahead", 0.15);
    no_replan_near_goal_ = declare_parameter<double>("no_replan_near_goal", 1.0);
    check_horizon_ = declare_parameter<double>("collision_check_horizon", 3.0);
    emergency_time_ = declare_parameter<double>("emergency_time", 0.6);
    max_deviation_ = declare_parameter<double>("max_deviation", 0.5);
    esdf_timeout_ = declare_parameter<double>("esdf_timeout", 1.0);
    pose_timeout_ = declare_parameter<double>("pose_timeout", 0.5);
    output_dt_ = declare_parameter<double>("output_dt", 0.05);
    output_horizon_ = declare_parameter<double>("output_horizon", 10.0);
    yaw_mode_ = declare_parameter<std::string>("yaw_mode", "velocity");  // velocity|goal|hold
    yaw_min_speed_ = declare_parameter<double>("yaw_min_speed", 0.2);
    max_yaw_rate_ = declare_parameter<double>("max_yaw_rate_deg", 45.0) * M_PI / 180.0;
    default_goal_z_ = declare_parameter<double>("default_goal_z", 1.0);
    hold_after_goal_ = declare_parameter<bool>("hold_after_goal", true);
    // replan start state: "auto" = continue the reference trajectory while the vehicle tracks it,
    // switch to the measured state (pose + velocity, predicted over the lookahead) when it doesn't;
    // "measured" = always the measured state; "reference" = always the reference (old behaviour)
    replan_start_ = declare_parameter<std::string>("replan_start", "auto");
    vel_mismatch_ = declare_parameter<double>("replan_vel_mismatch", 0.25);
    pos_mismatch_ = declare_parameter<double>("replan_pos_mismatch", 0.20);
    // GUIDED automation: enable the tracker and run the mission when the FC enters GUIDED
    auto_enable_tracker_ = declare_parameter<bool>("auto_enable_tracker", true);
    const std::string enable_srv =
      declare_parameter<std::string>("tracker_enable_service", "/trajectory_tracker/enable");
    const std::string tracker_status =
      declare_parameter<std::string>("tracker_status_topic", "/trajectory_tracker/status");
    const auto mg = declare_parameter<std::vector<double>>("mission_goals", std::vector<double>{});
    mission_auto_start_ = declare_parameter<bool>("mission_auto_start", true);
    mission_loop_ = declare_parameter<bool>("mission_loop", false);
    mission_hold_time_ = declare_parameter<double>("mission_hold_time", 1.0);
    if (mg.size() % 4 != 0) {
      throw std::invalid_argument("mission_goals needs groups of 4: x, y, z, yaw_deg");
    }
    for (size_t i = 0; i + 3 < mg.size(); i += 4) {
      mission_.push_back({V3(mg[i], mg[i + 1], mg[i + 2]), mg[i + 3] * M_PI / 180.0});
    }

    PlannerParams P;
    const double vmax = declare_parameter<double>("max_vel", 0.8);
    const double amax = declare_parameter<double>("max_acc", 1.0);
    P.astar.max_vel = P.opt.max_vel = vmax;
    P.astar.max_acc = P.opt.max_acc = amax;
    P.clearance = declare_parameter<double>("clearance", 0.3);
    P.min_clearance = declare_parameter<double>("min_clearance", 0.1);
    P.check_clearance = declare_parameter<double>("check_clearance", 0.15);
    P.check_dt = declare_parameter<double>("check_dt", 0.02);
    P.ctrl_pt_dist = declare_parameter<double>("ctrl_pt_dist", 0.3);
    P.time_alpha = declare_parameter<double>("time_alpha", 1.1);
    P.max_time_iterations = declare_parameter<int>("max_time_iterations", 30);
    P.feasibility_tolerance = declare_parameter<double>("feasibility_tolerance", 0.05);
    P.optimize = declare_parameter<bool>("optimize", true);
    P.astar.r = declare_parameter<int>("astar.r", 2);
    P.astar.tau = declare_parameter<double>("astar.tau", 0.6);
    P.astar.rho = declare_parameter<double>("astar.rho", 0.3);
    P.astar.resolution = declare_parameter<double>("astar.resolution", 0.1);
    P.astar.horizon = declare_parameter<double>("astar.horizon", 5.0);
    P.astar.goal_tol = declare_parameter<double>("astar.goal_tol", 0.15);
    P.astar.check_dt = declare_parameter<double>("astar.check_dt", 0.05);
    P.astar.sample_dt = declare_parameter<double>("astar.sample_dt", 0.1);
    P.astar.lambda_heu = declare_parameter<double>("astar.lambda_heu", 5.0);
    P.astar.max_nodes = declare_parameter<int>("astar.max_nodes", 10000);
    P.opt.lambda_smooth = declare_parameter<double>("opt.lambda_smooth", 10.0);
    P.opt.lambda_collision = declare_parameter<double>("opt.lambda_collision", 0.8);
    P.opt.lambda_feasibility = declare_parameter<double>("opt.lambda_feasibility", 0.01);
    P.opt.dist_threshold = declare_parameter<double>("opt.dist_threshold", 0.4);
    P.opt.max_iterations = declare_parameter<int>("opt.max_iterations", 200);
    P.opt.max_time = declare_parameter<double>("opt.max_time", 0.03);
    P.opt.algorithm = declare_parameter<std::string>("opt.algorithm", "LD_LBFGS");
    P.use_fence = declare_parameter<bool>("use_fence", true);
    const auto fmin = declare_parameter<std::vector<double>>("fence_min", {-1.8, -1.8, 0.5});
    const auto fmax = declare_parameter<std::vector<double>>("fence_max", {1.8, 1.8, 1.6});
    if (fmin.size() != 3 || fmax.size() != 3) {
      throw std::invalid_argument("fence_min / fence_max need 3 values");
    }
    // plan slightly inside the configured box: the optimiser treats the fence as a soft limit and
    // trajectory_tracker holds as soon as its reference leaves the (identical) tracker fence
    const double fence_margin = declare_parameter<double>("fence_margin", 0.1);
    P.fence_min = V3(fmin[0], fmin[1], fmin[2]) + V3::Constant(fence_margin);
    P.fence_max = V3(fmax[0], fmax[1], fmax[2]) - V3::Constant(fence_margin);
    core_ = std::make_unique<FastPlannerCore>(P);
    // RAPTOR Sec. VI risk-aware refinement
    P.risk.enable = declare_parameter<bool>("risk.enable", kRiskDefault);
    P.risk.psi_min = declare_parameter<double>("risk.psi_min", 0.25);
    P.risk.delta_v = declare_parameter<double>("risk.delta_v", 0.05);
    P.risk.R_q = declare_parameter<double>("risk.R_q", 0.3);
    P.risk.lambda_r = declare_parameter<double>("risk.lambda_r", 10.0);
    P.risk.w_r = declare_parameter<double>("risk.w_r", 1.0);
    P.risk.alpha = declare_parameter<double>("risk.alpha", 1.1);
    P.risk.max_iterations = declare_parameter<int>("risk.max_iterations", 5);
    P.risk.sample_dt = declare_parameter<double>("risk.sample_dt", 0.02);
    P.risk.min_frontier_time = declare_parameter<double>("risk.min_frontier_time", 0.3);
    core_->params().risk = P.risk;
    map_.unknown_distance = declare_parameter<double>("unknown_distance", 1.0);
    map_.unknown_is_occupied = declare_parameter<bool>("unknown_is_occupied", false);

    traj_pub_ = create_publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>(
      declare_parameter<std::string>("trajectory_topic", "/trajectory_tracker/trajectory"), 10);
    astar_pub_ = create_publisher<nav_msgs::msg::Path>("~/astar_path", 1);
    spline_pub_ = create_publisher<nav_msgs::msg::Path>("~/bspline_path", 1);
    ctrl_pub_ = create_publisher<nav_msgs::msg::Path>("~/control_points", 1);
    goal_pub_ = create_publisher<PoseStamped>("~/current_goal", rclcpp::QoS(1).transient_local());
    status_pub_ = create_publisher<std_msgs::msg::String>("~/status", 10);
    frontier_pub_ = create_publisher<PoseStamped>("~/frontier_point", 1);     // p_f
    critical_pub_ = create_publisher<PoseStamped>("~/critical_view_point", 1);  // p_c, along v_c
    viewpt_pub_ = create_publisher<PoseStamped>("~/view_point", 1);           // p_r(t_s)

    const auto sqos = rclcpp::SensorDataQoS();
    esdf_sub_ = create_subscription<da3_metric_scale::msg::EsdfGrid>(
      declare_parameter<std::string>("esdf_topic", "/esdf_publisher/esdf_grid"), rclcpp::QoS(1),
      [this](da3_metric_scale::msg::EsdfGrid::ConstSharedPtr m) {onEsdf(*m);});
    pose_sub_ = create_subscription<PoseStamped>(
      prefix_ + "/local_position/pose", sqos,
      [this](PoseStamped::ConstSharedPtr m) {
        const rclcpp::Time tn = now();
        const V3 p(m->pose.position.x, m->pose.position.y, m->pose.position.z);
        if (last_pose_.nanoseconds() > 0) {
          const double dt = (tn - last_pose_).seconds();
          if (dt > 0.005 && dt < 0.5) {
            const V3 v = (p - odomPos()) / dt;
            vel_fd_ = fd_valid_ ? V3(0.6 * vel_fd_ + 0.4 * v) : v;   // light low-pass
            fd_valid_ = true;
          } else if (dt >= 0.5) {
            fd_valid_ = false;
          }
        }
        pose_ = *m;
        last_pose_ = tn;
      });
    vel_sub_ = create_subscription<geometry_msgs::msg::TwistStamped>(
      prefix_ + "/local_position/velocity_local", sqos,
      [this](geometry_msgs::msg::TwistStamped::ConstSharedPtr m) {
        vel_ = V3(m->twist.linear.x, m->twist.linear.y, m->twist.linear.z);
        last_vel_ = now();
      });
    state_sub_ = create_subscription<mavros_msgs::msg::State>(
      prefix_ + "/state", sqos,
      [this](mavros_msgs::msg::State::ConstSharedPtr m) {fc_ = *m; last_fc_ = now();});
    goal_sub_ = create_subscription<PoseStamped>(
      "~/goal", 10, [this](PoseStamped::ConstSharedPtr m) {onGoal(*m, false);});
    rviz_goal_sub_ = create_subscription<PoseStamped>(
      declare_parameter<std::string>("rviz_goal_topic", "/goal_pose"), 10,
      [this](PoseStamped::ConstSharedPtr m) {onGoal(*m, true);});

    tracker_status_sub_ = create_subscription<std_msgs::msg::String>(
      tracker_status, 10, [this](std_msgs::msg::String::ConstSharedPtr m) {
        tracker_disabled_ = m->data.rfind("DISABLED", 0) == 0;
        last_tracker_status_ = now();
      });
    enable_client_ = create_client<Trigger>(enable_srv);
    start_mission_srv_ = create_service<Trigger>(
      "~/start_mission", [this](const std::shared_ptr<Trigger::Request>,
      std::shared_ptr<Trigger::Response> res) {
        res->success = startMission(res->message);
      });

    cancel_srv_ = create_service<Trigger>(
      "~/cancel", [this](const std::shared_ptr<Trigger::Request>,
      std::shared_ptr<Trigger::Response> res) {
        if (mission_active_) {RCLCPP_INFO(get_logger(), "mission stopped");}
        mission_active_ = false;
        has_goal_ = false;
        state_ = State::WAIT_GOAL;
        holdHere("cancelled");
        res->success = true;
        res->message = "goal cancelled, holding";
      });
    replan_srv_ = create_service<Trigger>(
      "~/replan", [this](const std::shared_ptr<Trigger::Request>,
      std::shared_ptr<Trigger::Response> res) {
        if (!has_goal_) {res->success = false; res->message = "no goal"; return;}
        force_replan_ = true;
        res->success = true;
        res->message = "replan requested";
      });

    exec_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / exec_rate_), std::bind(&FastPlannerNode::tick, this));
    pub_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    pub_timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / publish_rate_), std::bind(&FastPlannerNode::publish, this),
      pub_group_);
    status_timer_ = create_wall_timer(1s, std::bind(&FastPlannerNode::publishStatus, this));
    guided_timer_ = create_wall_timer(1s, std::bind(&FastPlannerNode::guidedAutomation, this));
    if (!mission_.empty()) {
      RCLCPP_INFO(get_logger(), "mission: %zu goals%s%s", mission_.size(),
        mission_auto_start_ ? ", starts when the FC enters GUIDED" : ", start with ~/start_mission",
        mission_loop_ ? ", looping" : "");
      for (size_t i = 0; i < mission_.size(); ++i) {
        RCLCPP_INFO(get_logger(), "  [%zu] %.2f %.2f %.2f yaw %.0f deg", i, mission_[i].p.x(),
          mission_[i].p.y(), mission_[i].p.z(), mission_[i].yaw * 180.0 / M_PI);
      }
    }

    RCLCPP_INFO(
      get_logger(), "%s ready: vmax %.2f amax %.2f clearance %.2f, fence [%.1f %.1f %.1f]"
      "..[%.1f %.1f %.1f], NLopt %s, risk-aware refinement %s. Send a goal: RViz '2D Goal Pose' "
      "or ~/goal", kNodeName, vmax, amax, P.clearance, fmin[0], fmin[1], fmin[2], fmax[0], fmax[1],
      fmax[2], P.opt.algorithm.c_str(), P.risk.enable ? "ON" : "off");
  }

private:
  // ---------------------------------------------------------------- inputs
  void onEsdf(const da3_metric_scale::msg::EsdfGrid & g)
  {
    const size_t n = static_cast<size_t>(g.nx) * g.ny * g.nz;
    if (n == 0 || g.distances.size() != n) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ESDF grid size mismatch, ignored");
      return;
    }
    if (!g.header.frame_id.empty() && g.header.frame_id != frame_id_) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "ESDF frame '%s' != frame_id '%s'",
        g.header.frame_id.c_str(), frame_id_.c_str());
    }
    map_.set(
      V3(g.origin.x, g.origin.y, g.origin.z), g.voxel_size_m, g.nx, g.ny, g.nz, g.distances,
      g.unobserved_value);
    last_esdf_ = now();
    have_map_ = true;
    new_map_ = true;
  }

  void onGoal(const PoseStamped & m, bool from_rviz)
  {
    if (!m.header.frame_id.empty() && m.header.frame_id != frame_id_) {
      RCLCPP_WARN(
        get_logger(), "goal frame '%s' != '%s' - use the %s frame (no TF lookup is done)",
        m.header.frame_id.c_str(), frame_id_.c_str(), frame_id_.c_str());
      return;
    }
    V3 g(m.pose.position.x, m.pose.position.y, m.pose.position.z);
    if (from_rviz && std::fabs(g.z()) < 1e-3) {g.z() = default_goal_z_;}
    if (mission_active_) {
      RCLCPP_WARN(get_logger(), "manual goal received: mission stopped");
      mission_active_ = false;
    }
    setGoal(g, yawOf(m.pose.orientation));
  }

  bool setGoal(V3 g, double yaw)
  {
    const auto & P = core_->params();
    if (P.use_fence) {
      const V3 c = g.cwiseMax(P.fence_min).cwiseMin(P.fence_max);
      if ((c - g).norm() > 1e-6) {
        RCLCPP_WARN(
          get_logger(), "goal (%.2f %.2f %.2f) outside the fence, clamped to (%.2f %.2f %.2f)",
          g.x(), g.y(), g.z(), c.x(), c.y(), c.z());
        g = c;
      }
    }
    if (have_map_) {
      const double d = map_.distance(g);
      if (d <= P.clearance) {
        RCLCPP_WARN(
          get_logger(), "goal (%.2f %.2f %.2f) rejected: ESDF %.2f m <= clearance %.2f m",
          g.x(), g.y(), g.z(), d, P.clearance);
        return false;
      }
    }
    goal_ = g;
    goal_yaw_ = yaw;
    has_goal_ = true;
    new_goal_ = true;
    fail_count_ = 0;
    PoseStamped gp;
    gp.header.frame_id = frame_id_;
    gp.header.stamp = now();
    gp.pose.position.x = g.x();
    gp.pose.position.y = g.y();
    gp.pose.position.z = g.z();
    gp.pose.orientation.z = std::sin(0.5 * yaw);
    gp.pose.orientation.w = std::cos(0.5 * yaw);
    goal_pub_->publish(gp);
    RCLCPP_INFO(get_logger(), "new goal (%.2f, %.2f, %.2f)", g.x(), g.y(), g.z());
    return true;
  }

  // ---------------------------------------------------------------- mission + GUIDED automation
  bool startMission(std::string & msg)
  {
    if (mission_.empty()) {msg = "no mission_goals configured"; return false;}
    mission_active_ = true;
    mission_done_ = false;
    mission_idx_ = 0;
    goal_reached_t_ = -1;
    if (!setGoal(mission_[0].p, mission_[0].yaw)) {
      mission_active_ = false;
      msg = "first mission goal rejected (see log)";
      return false;
    }
    msg = "mission started: " + std::to_string(mission_.size()) + " goals";
    RCLCPP_INFO(get_logger(), "%s", msg.c_str());
    return true;
  }

  // called when a goal is reached: hold, then move on to the next mission goal
  void missionStep(double t)
  {
    if (!mission_active_ || has_goal_) {return;}
    if (goal_reached_t_ < 0) {goal_reached_t_ = t;}
    if (t - goal_reached_t_ < mission_hold_time_) {return;}
    goal_reached_t_ = -1;
    ++mission_idx_;
    if (mission_idx_ >= mission_.size()) {
      if (!mission_loop_) {
        mission_active_ = false;
        mission_done_ = true;
        RCLCPP_INFO(get_logger(), "mission complete (%zu goals), holding", mission_.size());
        return;
      }
      mission_idx_ = 0;
    }
    RCLCPP_INFO(get_logger(), "mission goal %zu/%zu", mission_idx_ + 1, mission_.size());
    if (!setGoal(mission_[mission_idx_].p, mission_[mission_idx_].yaw)) {
      RCLCPP_WARN(get_logger(), "mission goal %zu rejected, skipping", mission_idx_ + 1);
    }
  }

  // 1 Hz: detect entering GUIDED, enable the tracker, start / resume the mission
  void guidedAutomation()
  {
    const bool guided = last_fc_.nanoseconds() > 0 && (now() - last_fc_).seconds() < 3.0 &&
      fc_.connected && fc_.armed && fc_.mode == "GUIDED";
    if (guided && !was_guided_) {
      RCLCPP_INFO(get_logger(), "FC entered GUIDED");
      if (mission_active_) {
        RCLCPP_INFO(get_logger(), "resuming mission at goal %zu/%zu", mission_idx_ + 1,
          mission_.size());
      } else if (mission_auto_start_ && !mission_.empty() && !mission_done_ && !has_goal_) {
        std::string msg;
        startMission(msg);
      }
    }
    if (!guided && was_guided_ && mission_active_) {
      RCLCPP_WARN(get_logger(), "left GUIDED: mission paused at goal %zu/%zu", mission_idx_ + 1,
        mission_.size());
    }
    was_guided_ = guided;

    // the tracker disables itself whenever the FC leaves GUIDED; re-enable it on return
    const bool status_fresh = last_tracker_status_.nanoseconds() > 0 &&
      (now() - last_tracker_status_).seconds() < 3.0;
    if (!auto_enable_tracker_ || !guided || !status_fresh || !tracker_disabled_ ||
      enable_pending_)
    {
      return;
    }
    if (!enable_client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "tracker enable service not available");
      return;
    }
    enable_pending_ = true;
    enable_client_->async_send_request(
      std::make_shared<Trigger::Request>(),
      [this](rclcpp::Client<Trigger>::SharedFuture f) {
        enable_pending_ = false;
        const auto r = f.get();
        if (r->success) {
          tracker_disabled_ = false;
          RCLCPP_INFO(get_logger(), "trajectory_tracker enabled: %s", r->message.c_str());
        } else {
          RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 3000,
            "trajectory_tracker enable refused (retrying): %s", r->message.c_str());
        }
      });
  }

  // ---------------------------------------------------------------- readiness
  bool poseFresh() const
  {
    return last_pose_.nanoseconds() > 0 && (now() - last_pose_).seconds() < pose_timeout_;
  }
  bool mapFresh() const
  {
    return have_map_ && (now() - last_esdf_).seconds() < esdf_timeout_;
  }
  bool fcReady(std::string & why) const
  {
    if (!require_guided_) {return true;}
    if (last_fc_.nanoseconds() == 0 || (now() - last_fc_).seconds() > 3.0) {
      why = "no FCU state";
      return false;
    }
    if (!fc_.connected) {why = "FCU not connected"; return false;}
    if (!fc_.armed) {why = "not armed"; return false;}
    if (fc_.mode != "GUIDED") {why = "mode " + fc_.mode + " (need GUIDED)"; return false;}
    return true;
  }
  V3 odomPos() const
  {
    return V3(pose_.pose.position.x, pose_.pose.position.y, pose_.pose.position.z);
  }
  // measured velocity: MAVROS velocity_local, else differentiated pose (never silently zero
  // while the vehicle is moving)
  V3 odomVel()
  {
    if (last_vel_.nanoseconds() > 0 && (now() - last_vel_).seconds() < pose_timeout_) {
      return vel_;
    }
    if (fd_valid_ && poseFresh()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 10000,
        "no velocity_local - using differentiated pose (set SR0_POSITION / stream rate)");
      return vel_fd_;
    }
    return V3::Zero();
  }

  // ---------------------------------------------------------------- FSM
  void tick()
  {
    const double t = now().seconds();
    if (!has_goal_) {
      state_ = State::WAIT_GOAL;
      missionStep(t);
      return;
    }
    if (!poseFresh()) {
      setReason("no fresh local_position/pose");
      return;
    }
    std::string why;
    if (!fcReady(why)) {
      if (state_ != State::WAIT_READY) {
        RCLCPP_WARN(get_logger(), "planner paused: %s (goal kept)", why.c_str());
        setActive(nullptr);        // stop publishing; replans from odom once GUIDED again
      }
      state_ = State::WAIT_READY;
      setReason(why);
      return;
    }
    if (!mapFresh()) {
      if (state_ == State::EXEC) {
        emergency("ESDF stale/missing");
      }
      state_ = State::GEN_NEW;
      setReason(have_map_ ? "ESDF stale" : "waiting for ESDF");
      return;
    }
    if (state_ == State::WAIT_READY || state_ == State::WAIT_GOAL) {state_ = State::GEN_NEW;}

    if (state_ == State::GEN_NEW) {
      if (t - last_attempt_ < replan_period_ && !new_goal_) {return;}
      new_goal_ = false;
      planFrom(t, odomPos(), odomVel(), V3::Zero(), yawOf(pose_.pose.orientation), nullptr,
        "from odom");
      return;
    }

    // ---- EXEC
    if (!active_) {state_ = State::GEN_NEW; return;}
    const Sample ref = active_->sample(t);
    if (active_->is_hold) {
      // holding (emergency / waiting): try again periodically
      if (t - last_attempt_ >= replan_period_ || new_goal_) {
        new_goal_ = false;
        planFrom(t, odomPos(), odomVel(), V3::Zero(), ref.yaw, nullptr, "from hold");
      }
      return;
    }
    // reached?
    if (active_->to_goal && t >= active_->tEndAbs() &&
      (odomPos() - goal_).norm() < 2.0 * core_->params().astar.goal_tol)
    {
      RCLCPP_INFO(get_logger(), "goal reached (%.2f, %.2f, %.2f)", goal_.x(), goal_.y(), goal_.z());
      has_goal_ = false;
      state_ = State::WAIT_GOAL;
      if (hold_after_goal_) {
        setActive(makeHold(active_->sample(active_->tEndAbs()).p, ref.yaw, t));
      } else {
        setActive(nullptr);
      }
      return;
    }
    // tracking deviation -> restart from the measured state
    const double dev = (odomPos() - ref.p).norm();
    if (dev > max_deviation_) {
      RCLCPP_WARN(get_logger(), "tracking deviation %.2f m > %.2f, replanning from odom", dev,
        max_deviation_);
      planFrom(t, odomPos(), odomVel(), V3::Zero(), ref.yaw, nullptr, "deviation");
      return;
    }
    // collision check of what is left against the newest map
    bool replan = force_replan_ || new_goal_;
    std::string reason = new_goal_ ? "new goal" : "requested";
    force_replan_ = false;
    if (new_map_) {
      new_map_ = false;
      const double tau = t - active_->t0;
      if (tau < active_->duration) {
        const double tc = core_->firstCollision(
          map_, active_->pos, std::max(0.0, tau), std::min(active_->duration, tau + check_horizon_),
          core_->params().check_clearance, nullptr, false);
        if (tc >= 0) {
          if (tc - tau < emergency_time_) {
            emergency("obstacle " + std::to_string(tc - tau).substr(0, 4) + " s ahead");
            return;
          }
          replan = true;
          reason = "collision ahead";
        }
      }
    }
    // receding horizon at a fixed interval (not when the goal is almost reached)
    const double remaining = active_->tEndAbs() - t;
    if (!replan && t - last_attempt_ >= replan_period_ &&
      (!active_->to_goal || remaining > no_replan_near_goal_))
    {
      replan = true;
      reason = "periodic";
    }
    if (!replan) {return;}
    new_goal_ = false;
    // start the new trajectory far enough ahead that it is ready before the vehicle gets there
    const double lookahead = std::min(1.0, std::max(replan_lookahead_, 1.5 * plan_time_ema_));
    const double ts = t + lookahead;
    Sample s = active_->sample(ts);
    // take the vehicle's actual motion into account: if it is not where / not as fast as the
    // reference says, start from the measured state predicted to ts
    std::string start = "ref";
    if (replan_start_ != "reference") {
      const V3 vm = odomVel();
      const V3 pm = odomPos() + vm * lookahead;
      if (replan_start_ == "measured" || (vm - s.v).norm() > vel_mismatch_ ||
        (pm - s.p).norm() > pos_mismatch_)
      {
        s.p = pm;
        s.v = vm;
        s.a = V3::Zero();
        start = "measured";
      }
    }
    const bool ok = planFrom(ts, s.p, s.v, s.a, s.yaw, active_, reason + ", start " + start);
    if (!ok && reason == "collision ahead") {
      // keep flying the old one only while the collision is far enough away
      const double tau = t - active_->t0;
      const double tc = core_->firstCollision(
        map_, active_->pos, std::max(0.0, tau), std::min(active_->duration, tau + check_horizon_),
        core_->params().check_clearance, nullptr, false);
      if (tc >= 0 && tc - tau < 2.0 * emergency_time_) {emergency("replan failed, obstacle ahead");}
    }
  }

  // plan from a start state; on success the new trajectory becomes active at t_start
  bool planFrom(
    double t_start, const V3 & p, const V3 & v, const V3 & a, double yaw0,
    std::shared_ptr<const ActiveTraj> prev, const std::string & reason)
  {
    last_attempt_ = now().seconds();
    const auto w0 = std::chrono::steady_clock::now();
    PlanResult r = core_->plan(map_, p, v, a, goal_);
    const double dt_plan =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count();
    plan_time_ema_ = std::max(dt_plan, 0.8 * plan_time_ema_ + 0.2 * dt_plan);
    last_result_ = r.message;
    if (r.at_goal) {
      has_goal_ = false;
      state_ = State::WAIT_GOAL;
      setActive(makeHold(p, yaw0, t_start));
      RCLCPP_INFO(get_logger(), "already at goal, holding");
      return true;
    }
    if (!r.ok) {
      ++fail_count_;
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 1000, "plan (%s) failed [%d]: %s", reason.c_str(), fail_count_,
        r.message.c_str());
      if (!active_ || reason == "deviation" || reason == "from odom") {
        // no usable trajectory to fall back on: hold where the vehicle is, retry periodically
        setActive(makeHold(odomPos(), yaw0, now().seconds()));
      }
      state_ = State::EXEC;
      return false;
    }
    fail_count_ = 0;
    reason_.clear();
    auto tr = std::make_shared<ActiveTraj>();
    tr->t0 = t_start;
    tr->pos = r.pos;
    tr->vel = r.vel;
    tr->acc = r.acc;
    tr->duration = r.duration;
    tr->to_goal = r.status == kino::Status::REACH_GOAL;
    buildYaw(*tr, yaw0);
    if (prev) {
      // drop the old one's own prefix so the chain never grows
      auto p2 = std::make_shared<ActiveTraj>(*prev);
      p2->prev.reset();
      tr->prev = p2;
    }
    setActive(tr);
    state_ = State::EXEC;
    ++n_plans_;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "plan #%d (%s) %s: %.2f s, |v|max %.2f |a|max %.2f, min ESDF %.2f m | A* %.1f ms (%d nodes) "
      "NLopt %.1f ms (%d evals, %.3f -> %.3f) time %.1f ms",
      n_plans_, reason.c_str(), r.message.c_str(), r.duration, r.max_vel, r.max_acc, r.min_dist,
      r.t_astar_ms, r.expanded, r.t_opt_ms, r.opt.evaluations, r.opt.cost_before,
      r.opt.cost_after, r.t_time_ms);
    publishViz(r);
    if (core_->params().risk.enable) {
      last_refine_ = r.refine.message;
      if (r.refine.refined || !r.refine.safe) {
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, "risk-aware: %s%s (%.1f ms)",
          r.refine.message.c_str(), r.used_refined ? "" : " [refined curve not used]",
          r.t_refine_ms);
      }
    }
    return true;
  }

  void buildYaw(ActiveTraj & tr, double yaw0) const
  {
    tr.yaw_dt = output_dt_;
    const int n = std::max(2, static_cast<int>(std::ceil(tr.duration / tr.yaw_dt)) + 1);
    tr.yaw.resize(n);
    double y = yaw0;
    const double step = max_yaw_rate_ * tr.yaw_dt;
    for (int i = 0; i < n; ++i) {
      const double tau = std::min(i * tr.yaw_dt, tr.duration);
      double target = y;
      if (yaw_mode_ == "velocity") {
        const V3 v = tr.vel.at(tau);
        if (v.head<2>().norm() > yaw_min_speed_) {target = std::atan2(v.y(), v.x());}
      } else if (yaw_mode_ == "goal") {
        target = goal_yaw_;
      }
      if (i > 0) {
        const double e = wrapPi(target - y);
        y += std::max(-step, std::min(step, e));
      }
      tr.yaw[i] = y;          // continuous (unwrapped) yaw
    }
  }

  void emergency(const std::string & why)
  {
    RCLCPP_ERROR(get_logger(), "EMERGENCY HOLD: %s", why.c_str());
    holdHere(why);
    state_ = State::EXEC;
    publish();                     // do not wait for the next publish tick
  }

  void holdHere(const std::string & why)
  {
    const double t = now().seconds();
    if (active_) {
      const Sample s = active_->sample(t);
      setActive(makeHold(s.p, s.yaw, t));
    } else if (poseFresh()) {
      setActive(makeHold(odomPos(), yawOf(pose_.pose.orientation), t));
    }
    setReason(why);
  }

  // ---------------------------------------------------------------- outputs
  void setActive(std::shared_ptr<ActiveTraj> a)
  {
    std::lock_guard<std::mutex> lk(active_mtx_);
    active_ = std::move(a);
  }

  void publish()
  {
    std::shared_ptr<const ActiveTraj> act;
    {
      std::lock_guard<std::mutex> lk(active_mtx_);
      act = active_;
    }
    if (!act) {return;}
    const double t = now().seconds();
    trajectory_msgs::msg::MultiDOFJointTrajectory m;
    m.header.stamp = now();
    m.header.frame_id = frame_id_;
    m.joint_names = {"base_link"};
    const double end = std::min(act->tEndAbs(), t + output_horizon_);
    const int n = std::max(1, static_cast<int>(std::ceil((end - t) / output_dt_)));
    m.points.reserve(n + 1);
    for (int k = 0; k <= n; ++k) {
      const double tk = std::min(t + k * output_dt_, std::max(end, t + output_dt_));
      const Sample s = act->sample(tk);
      trajectory_msgs::msg::MultiDOFJointTrajectoryPoint pt;
      geometry_msgs::msg::Transform tf;
      tf.translation.x = s.p.x();
      tf.translation.y = s.p.y();
      tf.translation.z = s.p.z();
      tf.rotation.z = std::sin(0.5 * s.yaw);
      tf.rotation.w = std::cos(0.5 * s.yaw);
      geometry_msgs::msg::Twist vel, acc;
      vel.linear.x = s.v.x();
      vel.linear.y = s.v.y();
      vel.linear.z = s.v.z();
      vel.angular.z = s.yaw_rate;
      acc.linear.x = s.a.x();
      acc.linear.y = s.a.y();
      acc.linear.z = s.a.z();
      pt.transforms.push_back(tf);
      pt.velocities.push_back(vel);
      pt.accelerations.push_back(acc);
      pt.time_from_start = toDuration(tk - t);
      m.points.push_back(pt);
    }
    traj_pub_->publish(m);
  }

  void publishViz(const PlanResult & r)
  {
    nav_msgs::msg::Path a, b, c;
    a.header.stamp = b.header.stamp = c.header.stamp = now();
    a.header.frame_id = b.header.frame_id = c.header.frame_id = frame_id_;
    auto ps = [this](const V3 & p) {
        PoseStamped q;
        q.header.frame_id = frame_id_;
        q.pose.position.x = p.x();
        q.pose.position.y = p.y();
        q.pose.position.z = p.z();
        q.pose.orientation.w = 1.0;
        return q;
      };
    for (const auto & s : r.astar) {a.poses.push_back(ps(s.pos));}
    for (double t = 0; t <= r.duration + 1e-9; t += 0.05) {b.poses.push_back(ps(r.pos.at(t)));}
    for (int i = 0; i < r.pos.ctrl().rows(); ++i) {
      c.poses.push_back(ps(r.pos.ctrl().row(i).transpose()));
    }
    astar_pub_->publish(a);
    spline_pub_->publish(b);
    ctrl_pub_->publish(c);
    const auto & V = r.refine.vis;
    if (core_->params().risk.enable && V.has_frontier) {
      PoseStamped f = ps(V.p_f);
      f.header.stamp = a.header.stamp;
      frontier_pub_->publish(f);
      if (!V.visible) {
        PoseStamped pc = ps(V.p_c);
        pc.header.stamp = a.header.stamp;
        // orientation: x axis along v_c (from p_f toward p_c), for an RViz arrow
        const double yaw = std::atan2(V.v_c.y(), V.v_c.x());
        const double pitch = -std::asin(std::max(-1.0, std::min(1.0, V.v_c.z())));
        const Eigen::Quaterniond q = Eigen::AngleAxisd(yaw, V3::UnitZ()) *
          Eigen::AngleAxisd(pitch, V3::UnitY());
        pc.pose.orientation.x = q.x();
        pc.pose.orientation.y = q.y();
        pc.pose.orientation.z = q.z();
        pc.pose.orientation.w = q.w();
        critical_pub_->publish(pc);
      }
      if (r.used_refined) {
        PoseStamped v = ps(r.refine.p_s);
        v.header.stamp = a.header.stamp;
        viewpt_pub_->publish(v);
      }
    }
  }

  void setReason(const std::string & s) {reason_ = s;}

  void publishStatus()
  {
    static const char * names[] = {"WAIT_GOAL", "WAIT_READY", "GEN_NEW", "EXEC"};
    std::string s = std::string(names[static_cast<int>(state_)]);
    if (state_ == State::EXEC && active_) {
      s += active_->is_hold ? " (holding)" : (active_->to_goal ? " (to goal)" : " (to horizon)");
    }
    if (mission_active_) {
      s += " | mission " + std::to_string(mission_idx_ + 1) + "/" + std::to_string(mission_.size());
    } else if (mission_done_) {
      s += " | mission done";
    }
    if (has_goal_) {
      char buf[96];
      std::snprintf(buf, sizeof(buf), " | goal %.2f %.2f %.2f", goal_.x(), goal_.y(), goal_.z());
      s += buf;
    }
    s += " | map " + std::string(mapFresh() ? "ok" : (have_map_ ? "stale" : "none"));
    s += " | plans " + std::to_string(n_plans_) + " | last: " + last_result_;
    if (core_->params().risk.enable) {s += " | risk: " + last_refine_;}
    if (!reason_.empty() && state_ != State::EXEC) {s += " | " + reason_;}
    std_msgs::msg::String m;
    m.data = s;
    status_pub_->publish(m);
  }

  // params
  std::string prefix_, frame_id_, yaw_mode_;
  bool require_guided_{true}, hold_after_goal_{true};
  double exec_rate_, publish_rate_, replan_period_, replan_lookahead_, no_replan_near_goal_;
  double check_horizon_, emergency_time_, max_deviation_, esdf_timeout_, pose_timeout_;
  double output_dt_, output_horizon_, yaw_min_speed_, max_yaw_rate_, default_goal_z_;

  std::unique_ptr<FastPlannerCore> core_;
  EsdfMap map_;
  bool have_map_{false}, new_map_{false};
  rclcpp::Time last_esdf_{0, 0, RCL_ROS_TIME};

  PoseStamped pose_;
  V3 vel_{V3::Zero()};
  rclcpp::Time last_pose_{0, 0, RCL_ROS_TIME}, last_vel_{0, 0, RCL_ROS_TIME};
  V3 vel_fd_{V3::Zero()};
  bool fd_valid_{false};
  std::string replan_start_{"auto"};
  double vel_mismatch_{0.25}, pos_mismatch_{0.20};
  mavros_msgs::msg::State fc_;
  rclcpp::Time last_fc_{0, 0, RCL_ROS_TIME};

  State state_{State::WAIT_GOAL};
  bool has_goal_{false}, new_goal_{false}, force_replan_{false};
  V3 goal_{V3::Zero()};
  double goal_yaw_{0};
  std::shared_ptr<ActiveTraj> active_;   // written by the FSM thread only, read by publish()
  std::mutex active_mtx_;
  double plan_time_ema_{0.05};
  rclcpp::CallbackGroup::SharedPtr pub_group_;
  double last_attempt_{0};
  int fail_count_{0}, n_plans_{0};
  std::string last_result_{"-"}, reason_, last_refine_{"-"};

  rclcpp::Publisher<trajectory_msgs::msg::MultiDOFJointTrajectory>::SharedPtr traj_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr astar_pub_, spline_pub_, ctrl_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr goal_pub_, frontier_pub_, critical_pub_, viewpt_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Subscription<da3_metric_scale::msg::EsdfGrid>::SharedPtr esdf_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr pose_sub_, goal_sub_, rviz_goal_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr vel_sub_;
  rclcpp::Subscription<mavros_msgs::msg::State>::SharedPtr state_sub_;
  rclcpp::Service<Trigger>::SharedPtr cancel_srv_, replan_srv_, start_mission_srv_;
  rclcpp::Client<Trigger>::SharedPtr enable_client_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr tracker_status_sub_;
  rclcpp::TimerBase::SharedPtr guided_timer_;
  struct MissionGoal {V3 p; double yaw;};
  std::vector<MissionGoal> mission_;
  size_t mission_idx_{0};
  bool mission_active_{false}, mission_done_{false}, mission_auto_start_{true}, mission_loop_{false};
  double mission_hold_time_{1.0}, goal_reached_t_{-1};
  bool auto_enable_tracker_{true}, was_guided_{false}, tracker_disabled_{true}, enable_pending_{false};
  rclcpp::Time last_tracker_status_{0, 0, RCL_ROS_TIME};
  rclcpp::TimerBase::SharedPtr exec_timer_, pub_timer_, status_timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<FastPlannerNode>();
  // 2 threads: FSM/planning + trajectory publishing (planning can take longer than 1/publish_rate)
  rclcpp::executors::MultiThreadedExecutor exec(rclcpp::ExecutorOptions(), 2);
  exec.add_node(node);
  exec.spin();
  rclcpp::shutdown();
  return 0;
}
