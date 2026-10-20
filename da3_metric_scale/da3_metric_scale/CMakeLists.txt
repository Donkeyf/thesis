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

add_executable(da3_metric_scaler src/da3_metric_scaler.cpp)
target_include_directories(da3_metric_scaler PRIVATE ${EIGEN3_INCLUDE_DIR})
target_link_libraries(da3_metric_scaler ${OpenCV_LIBS})
ament_target_dependencies(da3_metric_scaler
  rclcpp sensor_msgs geometry_msgs std_msgs builtin_interfaces
  cv_bridge message_filters tf2_ros)

add_executable(mocap_to_mavros src/mocap_to_mavros.cpp)
target_include_directories(mocap_to_mavros PRIVATE ${EIGEN3_INCLUDE_DIR})
ament_target_dependencies(mocap_to_mavros
  rclcpp geometry_msgs builtin_interfaces mavros_msgs geographic_msgs)

install(TARGETS da3_metric_scaler mocap_to_mavros DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})

ament_package()
