cmake_minimum_required(VERSION 3.16)
project(da3_metric_scale LANGUAGES CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -O2)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc calib3d video)
find_package(Eigen3 REQUIRED)
find_package(mavros_msgs REQUIRED)
find_package(geographic_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(std_srvs REQUIRED)

add_executable(da3_metric_scaler src/da3_metric_scaler.cpp)
target_link_libraries(da3_metric_scaler
  rclcpp::rclcpp
  ${sensor_msgs_TARGETS}
  ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${builtin_interfaces_TARGETS}
  cv_bridge::cv_bridge
  message_filters::message_filters
  tf2_ros::tf2_ros
  ${OpenCV_LIBS}
  Eigen3::Eigen)
target_include_directories(da3_metric_scaler PRIVATE ${OpenCV_INCLUDE_DIRS})

add_executable(mocap_to_mavros src/mocap_to_mavros.cpp)
target_link_libraries(mocap_to_mavros
  rclcpp::rclcpp
  ${geometry_msgs_TARGETS}
  ${builtin_interfaces_TARGETS}
  ${mavros_msgs_TARGETS}
  ${geographic_msgs_TARGETS}
  Eigen3::Eigen)

add_executable(guided_waypoints src/guided_waypoints.cpp)
target_link_libraries(guided_waypoints
  rclcpp::rclcpp
  ${geometry_msgs_TARGETS}
  ${std_msgs_TARGETS}
  ${nav_msgs_TARGETS}
  ${std_srvs_TARGETS}
  ${mavros_msgs_TARGETS})

install(TARGETS da3_metric_scaler mocap_to_mavros guided_waypoints DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
