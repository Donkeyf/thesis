// esdf_publisher.cpp
//
// Publishes nvblox's 3D ESDF as topics. In esdf_mode "3d" nvblox only streams 2D
// slices; the full 3D grid is available through its
//   /nvblox_node/get_esdf_and_gradient   (nvblox_msgs/srv/EsdfAndGradients)
// service. This node calls it at `rate` Hz for an axis-aligned box around the
// vehicle (or the whole map) and publishes:
//   ~/esdf_grid   da3_metric_scale/msg/EsdfGrid      dense grid for planners
//   ~/esdf_cloud  sensor_msgs/PointCloud2 (x,y,z,distance)  observed voxels, for RViz /
//                 planners that prefer points (optionally only near obstacles)

#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <nvblox_msgs/srv/esdf_and_gradients.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#if __has_include(<tf2/exceptions.hpp>)
#include <tf2/exceptions.hpp>
#else
#include <tf2/exceptions.h>
#endif
#if __has_include(<tf2_ros/buffer.hpp>)
#include <tf2_ros/buffer.hpp>
#include <tf2_ros/transform_listener.hpp>
#else
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#endif

#include "da3_metric_scale/msg/esdf_grid.hpp"

using EsdfSrv = nvblox_msgs::srv::EsdfAndGradients;
using EsdfGrid = da3_metric_scale::msg::EsdfGrid;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::PointField;

class EsdfPublisher : public rclcpp::Node
{
public:
  EsdfPublisher()
  : Node("esdf_publisher")
  {
    service_ = declare_parameter<std::string>("service", "/nvblox_node/get_esdf_and_gradient");
    global_frame_ = declare_parameter<std::string>("global_frame", "map");
    center_frame_ = declare_parameter<std::string>("center_frame", "camera_optical");
    rate_ = declare_parameter<double>("rate", 2.0);
    aabb_size_ = declare_parameter<std::vector<double>>("aabb_size", {4.0, 4.0, 2.0});
    z_limits_ = declare_parameter<std::vector<double>>("z_limits", {-0.2, 2.5});
    full_map_ = declare_parameter<bool>("full_map", false);
    update_esdf_ = declare_parameter<bool>("update_esdf", true);
    visualize_ = declare_parameter<bool>("nvblox_visualize", false);
    unobserved_ = static_cast<float>(declare_parameter<double>("unobserved_value", -1000.0));
    cloud_max_dist_ = declare_parameter<double>("cloud_max_distance", 1.0);  // <=0: all
    publish_cloud_ = declare_parameter<bool>("publish_cloud", true);
    if (aabb_size_.size() != 3 || z_limits_.size() != 2) {
      throw std::runtime_error("aabb_size needs 3 values, z_limits 2");
    }

    grid_pub_ = create_publisher<EsdfGrid>("~/esdf_grid", rclcpp::QoS(1));
    cloud_pub_ = create_publisher<PointCloud2>("~/esdf_cloud", rclcpp::QoS(1));
    client_ = create_client<EsdfSrv>(service_);
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
    timer_ = create_wall_timer(
      std::chrono::duration<double>(1.0 / rate_), std::bind(&EsdfPublisher::tick, this));
    if (full_map_) {
      RCLCPP_INFO(get_logger(), "esdf_publisher: %s @ %.1f Hz, full map", service_.c_str(), rate_);
    } else {
      RCLCPP_INFO(
        get_logger(), "esdf_publisher: %s @ %.1f Hz, %.1fx%.1fx%.1f m box around '%s'",
        service_.c_str(), rate_, aabb_size_[0], aabb_size_[1], aabb_size_[2],
        center_frame_.c_str());
    }
  }

private:
  void tick()
  {
    if (in_flight_) {
      if ((now() - sent_at_).seconds() < 5.0) {return;}
      RCLCPP_WARN(get_logger(), "ESDF request timed out, retrying");
      in_flight_ = false;
    }
    if (!client_->service_is_ready()) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "waiting for %s (nvblox running in esdf_mode 3d?)",
        service_.c_str());
      return;
    }
    auto req = std::make_shared<EsdfSrv::Request>();
    req->update_esdf = update_esdf_;
    req->visualize_esdf = visualize_;
    req->frame_id = global_frame_;
    req->use_aabb = !full_map_;
    if (!full_map_) {
      double cx = 0, cy = 0, cz = 0;
      if (!center_frame_.empty()) {
        try {
          const auto tf = tf_buffer_->lookupTransform(
            global_frame_, center_frame_, tf2::TimePointZero);
          cx = tf.transform.translation.x;
          cy = tf.transform.translation.y;
          cz = tf.transform.translation.z;
        } catch (const tf2::TransformException & e) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 5000, "no TF %s -> %s yet: %s", global_frame_.c_str(),
            center_frame_.c_str(), e.what());
          return;
        }
      }
      double zmin = std::max(cz - aabb_size_[2] / 2, z_limits_[0]);
      double zmax = std::min(cz + aabb_size_[2] / 2, z_limits_[1]);
      if (zmax <= zmin) {zmax = zmin + 0.1;}
      req->aabb_min_m.x = cx - aabb_size_[0] / 2;
      req->aabb_min_m.y = cy - aabb_size_[1] / 2;
      req->aabb_min_m.z = zmin;
      req->aabb_size_m.x = aabb_size_[0];
      req->aabb_size_m.y = aabb_size_[1];
      req->aabb_size_m.z = zmax - zmin;
    }
    in_flight_ = true;
    sent_at_ = now();
    client_->async_send_request(
      req, [this](rclcpp::Client<EsdfSrv>::SharedFuture f) {
        in_flight_ = false;
        onResponse(*f.get());
      });
  }

  void onResponse(const EsdfSrv::Response & r)
  {
    if (!r.success) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "nvblox returned no ESDF grid");
      return;
    }
    const auto & arr = r.esdf_and_gradients;
    if (arr.layout.dim.size() < 3) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "unexpected ESDF array layout");
      return;
    }
    const uint32_t nx = arr.layout.dim[0].size, ny = arr.layout.dim[1].size,
      nz = arr.layout.dim[2].size;
    if (static_cast<size_t>(nx) * ny * nz != arr.data.size()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "ESDF array size mismatch");
      return;
    }

    EsdfGrid g;
    g.header = r.header;
    if (g.header.frame_id.empty()) {g.header.frame_id = global_frame_;}
    g.origin = r.origin_m;
    g.voxel_size_m = r.voxel_size_m;
    g.nx = nx;
    g.ny = ny;
    g.nz = nz;
    g.unobserved_value = unobserved_;
    g.distances = arr.data;
    grid_pub_->publish(g);

    if (publish_cloud_ && cloud_pub_->get_subscription_count() > 0) {
      publishCloud(g);
    }
  }

  void publishCloud(const EsdfGrid & g)
  {
    PointCloud2 pc;
    pc.header = g.header;
    pc.height = 1;
    pc.is_dense = true;
    pc.is_bigendian = false;
    const char * names[4] = {"x", "y", "z", "distance"};
    for (uint32_t i = 0; i < 4; ++i) {
      PointField f;
      f.name = names[i];
      f.offset = 4 * i;
      f.datatype = PointField::FLOAT32;
      f.count = 1;
      pc.fields.push_back(f);
    }
    pc.point_step = 16;
    std::vector<float> pts;
    pts.reserve(g.distances.size() / 4);
    const float vs = g.voxel_size_m;
    for (uint32_t ix = 0; ix < g.nx; ++ix) {
      for (uint32_t iy = 0; iy < g.ny; ++iy) {
        for (uint32_t iz = 0; iz < g.nz; ++iz) {
          const float d = g.distances[(static_cast<size_t>(ix) * g.ny + iy) * g.nz + iz];
          if (std::fabs(d - g.unobserved_value) < 1e-2f) {continue;}
          if (cloud_max_dist_ > 0 && d > cloud_max_dist_) {continue;}
          pts.push_back(static_cast<float>(g.origin.x + (ix + 0.5f) * vs));
          pts.push_back(static_cast<float>(g.origin.y + (iy + 0.5f) * vs));
          pts.push_back(static_cast<float>(g.origin.z + (iz + 0.5f) * vs));
          pts.push_back(d);
        }
      }
    }
    pc.width = static_cast<uint32_t>(pts.size() / 4);
    pc.row_step = pc.point_step * pc.width;
    pc.data.resize(pts.size() * sizeof(float));
    std::memcpy(pc.data.data(), pts.data(), pc.data.size());
    cloud_pub_->publish(pc);
  }

  std::string service_, global_frame_, center_frame_;
  double rate_, cloud_max_dist_;
  std::vector<double> aabb_size_, z_limits_;
  bool full_map_, update_esdf_, visualize_, publish_cloud_;
  float unobserved_;
  bool in_flight_{false};
  rclcpp::Time sent_at_{0, 0, RCL_ROS_TIME};

  rclcpp::Client<EsdfSrv>::SharedPtr client_;
  rclcpp::Publisher<EsdfGrid>::SharedPtr grid_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cloud_pub_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EsdfPublisher>());
  rclcpp::shutdown();
  return 0;
}
