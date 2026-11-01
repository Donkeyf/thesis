// tf2_ros stand-in for tests: one global transform target <- source
#pragma once
#include <stdexcept>
#include "../ros_mock.hpp"
namespace tf2_ros {
inline geometry_msgs::msg::TransformStamped g_tf;
inline bool g_tf_ok = true;
struct Buffer {
  template<class C> explicit Buffer(C) {}
  geometry_msgs::msg::TransformStamped lookupTransform(
    const std::string &, const std::string &, const rclcpp::Time &, const rclcpp::Duration &) const
  {
    if (!g_tf_ok) {throw std::runtime_error("no transform");}
    return g_tf;
  }
};
}
