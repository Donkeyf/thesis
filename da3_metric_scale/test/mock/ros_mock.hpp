// Minimal stand-in for the rclcpp / message APIs used by fast_planner_node.cpp, so the node can be
// compiled and run closed-loop in a plain C++ test (no ROS install). Test-only.
#pragma once
#include <any>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace builtin_interfaces::msg {
struct Time {int32_t sec{0}; uint32_t nanosec{0};};
struct Duration {int32_t sec{0}; uint32_t nanosec{0};};
}
#define MOCK_MSG(T) using SharedPtr = std::shared_ptr<T>; using ConstSharedPtr = std::shared_ptr<const T>;
namespace std_msgs::msg {
struct Header {builtin_interfaces::msg::Time stamp; std::string frame_id; MOCK_MSG(Header)};
struct String {std::string data; MOCK_MSG(String)};
}
namespace geometry_msgs::msg {
struct Point {double x{0}, y{0}, z{0};};
struct Vector3 {double x{0}, y{0}, z{0};};
struct Quaternion {double x{0}, y{0}, z{0}, w{1};};
struct Pose {Point position; Quaternion orientation;};
struct PoseStamped {std_msgs::msg::Header header; Pose pose; MOCK_MSG(PoseStamped)};
struct Twist {Vector3 linear, angular;};
struct TwistStamped {std_msgs::msg::Header header; Twist twist; MOCK_MSG(TwistStamped)};
struct Transform {Vector3 translation; Quaternion rotation;};
}
namespace mavros_msgs::msg {
struct State {std_msgs::msg::Header header; bool connected{false}, armed{false}; std::string mode; MOCK_MSG(State)};
}
namespace nav_msgs::msg {
struct Path {std_msgs::msg::Header header; std::vector<geometry_msgs::msg::PoseStamped> poses; MOCK_MSG(Path)};
}
namespace std_srvs::srv {
struct Trigger {
  struct Request {};
  struct Response {bool success{false}; std::string message;};
};
}
namespace trajectory_msgs::msg {
struct MultiDOFJointTrajectoryPoint {
  std::vector<geometry_msgs::msg::Transform> transforms;
  std::vector<geometry_msgs::msg::Twist> velocities, accelerations;
  builtin_interfaces::msg::Duration time_from_start;
};
struct MultiDOFJointTrajectory {
  std_msgs::msg::Header header; std::vector<std::string> joint_names;
  std::vector<MultiDOFJointTrajectoryPoint> points; MOCK_MSG(MultiDOFJointTrajectory)
};
}
namespace da3_metric_scale::msg {
struct EsdfGrid {
  std_msgs::msg::Header header; geometry_msgs::msg::Point origin; float voxel_size_m{0};
  uint32_t nx{0}, ny{0}, nz{0}; float unobserved_value{-1000}; std::vector<float> distances;
  MOCK_MSG(EsdfGrid)
};
}

#define RCL_ROS_TIME 0
namespace rclcpp {
inline double g_now = 1000.0;               // simulated clock [s]
inline bool g_quiet = false;
inline std::map<std::string, std::any> g_params;
struct Duration {double s; double seconds() const {return s;}};
struct Time {
  int64_t ns{0};
  Time() = default;
  Time(int32_t sec, uint32_t nsec, int = 0) : ns(int64_t(sec) * 1000000000LL + nsec) {}
  static Time fromSec(double s) {Time t; t.ns = int64_t(std::llround(s * 1e9)); return t;}
  int64_t nanoseconds() const {return ns;}
  double seconds() const {return ns * 1e-9;}
  Duration operator-(const Time & o) const {return {(ns - o.ns) * 1e-9};}
  operator builtin_interfaces::msg::Time() const {
    builtin_interfaces::msg::Time t; t.sec = int32_t(ns / 1000000000LL); t.nanosec = uint32_t(ns % 1000000000LL); return t;}
};
enum class CallbackGroupType {MutuallyExclusive, Reentrant};
struct CallbackGroup {using SharedPtr = std::shared_ptr<CallbackGroup>;};
struct ExecutorOptions {};
struct QoS {explicit QoS(int) {} QoS & transient_local() {return *this;}};
struct SensorDataQoS : QoS {SensorDataQoS() : QoS(5) {}};
struct TimerBase {
  using SharedPtr = std::shared_ptr<TimerBase>;
  double period, next; std::function<void()> cb;
};
template<class T> struct Publisher {
  using SharedPtr = std::shared_ptr<Publisher<T>>;
  std::string topic; std::vector<T> sent;
  void publish(const T & m) {sent.push_back(m); if (sent.size() > 50) sent.erase(sent.begin());}
};
template<class T> struct Subscription {
  using SharedPtr = std::shared_ptr<Subscription<T>>;
  std::string topic; std::function<void(std::shared_ptr<const T>)> cb;
  void inject(const T & m) {cb(std::make_shared<const T>(m));}
};
// service clients: requests go to a handler registered by the test (g_service_handlers)
inline std::map<std::string, std::function<std::shared_ptr<void>(std::shared_ptr<void>)>> g_service_handlers;
template<class S> struct Client {
  using SharedPtr = std::shared_ptr<Client<S>>;
  struct SharedFuture {
    std::shared_ptr<typename S::Response> r;
    std::shared_ptr<typename S::Response> get() const {return r;}
  };
  std::string name;
  int calls{0};
  bool service_is_ready() const {return g_service_handlers.count(name) > 0;}
  template<class F> void async_send_request(std::shared_ptr<typename S::Request> req, F && cb) {
    ++calls;
    auto it = g_service_handlers.find(name);
    if (it == g_service_handlers.end()) {return;}
    SharedFuture f{std::static_pointer_cast<typename S::Response>(it->second(req))};
    cb(f);
  }
};
template<class S> struct Service {
  using SharedPtr = std::shared_ptr<Service<S>>;
  std::function<void(std::shared_ptr<typename S::Request>, std::shared_ptr<typename S::Response>)> cb;
};
class Node {
public:
  explicit Node(const std::string & name) : name_(name) {}
  virtual ~Node() = default;
  template<class T> T declare_parameter(const std::string & n, const T & def) {
    auto it = g_params.find(n);
    return it == g_params.end() ? def : std::any_cast<T>(it->second);
  }
  template<class T> typename Publisher<T>::SharedPtr create_publisher(const std::string & t, const QoS &) {
    auto p = std::make_shared<Publisher<T>>(); p->topic = t; pubs[t] = p; return p;}
  template<class T> typename Publisher<T>::SharedPtr create_publisher(const std::string & t, int d) {
    return create_publisher<T>(t, QoS(d));}
  template<class T, class F> typename Subscription<T>::SharedPtr create_subscription(
    const std::string & t, const QoS &, F && f) {
    auto s = std::make_shared<Subscription<T>>(); s->topic = t; s->cb = f; subs[t] = s; return s;}
  template<class T, class F> typename Subscription<T>::SharedPtr create_subscription(
    const std::string & t, int d, F && f) {return create_subscription<T>(t, QoS(d), std::forward<F>(f));}
  template<class S> typename Client<S>::SharedPtr create_client(const std::string & n) {
    auto c = std::make_shared<Client<S>>(); c->name = n; clients[n] = c; return c;}
  template<class S, class F> typename Service<S>::SharedPtr create_service(const std::string & n, F && f) {
    auto s = std::make_shared<Service<S>>(); s->cb = f; srvs[n] = s; return s;}
  CallbackGroup::SharedPtr create_callback_group(CallbackGroupType) {return std::make_shared<CallbackGroup>();}
  template<class D, class F> TimerBase::SharedPtr create_wall_timer(D period, F && f, CallbackGroup::SharedPtr) {
    return create_wall_timer(period, std::forward<F>(f));}
  template<class D, class F> TimerBase::SharedPtr create_wall_timer(D period, F && f) {
    auto t = std::make_shared<TimerBase>();
    t->period = std::chrono::duration<double>(period).count(); t->next = g_now + t->period; t->cb = f;
    timers.push_back(t); return t;}
  Time now() const {return Time::fromSec(g_now);}
  int get_logger() const {return 0;}
  std::shared_ptr<int> get_clock() const {return std::make_shared<int>(0);}
  void spinUntil(double t_end) {         // run due timers in time order
    while (true) {
      TimerBase * best = nullptr;
      for (auto & t : timers) if (!best || t->next < best->next) best = t.get();
      if (!best || best->next > t_end) break;
      g_now = best->next; best->next += best->period; best->cb();
    }
    g_now = t_end;
  }
  std::map<std::string, std::shared_ptr<void>> pubs, subs, srvs, clients;
  std::vector<TimerBase::SharedPtr> timers;
  std::string name_;
};
namespace executors {
struct MultiThreadedExecutor {
  MultiThreadedExecutor(const ExecutorOptions &, size_t) {}
  template<class N> void add_node(N) {}
  void spin() {}
};
}
inline void init(int, char **) {}
inline void shutdown() {}
template<class N> void spin(N) {}
}  // namespace rclcpp

#define MOCK_LOG(lvl, fmt, ...) do { if (!rclcpp::g_quiet) { std::printf("[%8.3f %s] " fmt "\n", rclcpp::g_now, lvl, ##__VA_ARGS__);} } while (0)
#define RCLCPP_INFO(l, fmt, ...) MOCK_LOG("INFO", fmt, ##__VA_ARGS__)
#define RCLCPP_WARN(l, fmt, ...) MOCK_LOG("WARN", fmt, ##__VA_ARGS__)
#define RCLCPP_ERROR(l, fmt, ...) MOCK_LOG("ERROR", fmt, ##__VA_ARGS__)
#define RCLCPP_INFO_THROTTLE(l, c, ms, fmt, ...) do { (void)(c); MOCK_LOG("INFO", fmt, ##__VA_ARGS__);} while (0)
#define RCLCPP_WARN_THROTTLE(l, c, ms, fmt, ...) do { (void)(c); MOCK_LOG("WARN", fmt, ##__VA_ARGS__);} while (0)
#define RCLCPP_INFO_ONCE(l, fmt, ...) MOCK_LOG("INFO", fmt, ##__VA_ARGS__)
