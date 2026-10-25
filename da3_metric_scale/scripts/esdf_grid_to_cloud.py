#!/usr/bin/env python3
"""Republish da3_metric_scale/msg/EsdfGrid as sensor_msgs/PointCloud2 for RViz.

  ros2 bag play esdf_run1 --clock --loop
  python3 esdf_grid_to_cloud.py --ros-args -p use_sim_time:=true
  rviz2 --ros-args -p use_sim_time:=true
     Fixed Frame: map | Add -> By topic -> /esdf_grid_cloud (PointCloud2)
     Color Transformer: Intensity, Channel Name: distance
     Decay Time: e.g. 1000 (accumulates every box -> whole map)

Params:
  input   (/esdf_publisher/esdf_grid)  output (/esdf_grid_cloud)
  max_distance  (0.10)  only voxels with ESDF <= this [m]; 0.10 ~ obstacle surfaces.
                        <= -1 publishes every observed voxel (big!)
  min_distance  (-1.0)  lower bound (negative = inside obstacles)
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2, PointField

from da3_metric_scale.msg import EsdfGrid


class EsdfGridToCloud(Node):
    def __init__(self):
        super().__init__('esdf_grid_to_cloud')
        self.max_d = self.declare_parameter('max_distance', 0.10).value
        self.min_d = self.declare_parameter('min_distance', -1.0).value
        inp = self.declare_parameter('input', '/esdf_publisher/esdf_grid').value
        out = self.declare_parameter('output', '/esdf_grid_cloud').value
        self.pub = self.create_publisher(PointCloud2, out, QoSProfile(depth=5))
        self.create_subscription(EsdfGrid, inp, self.cb, QoSProfile(depth=5))
        self.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                       for i, n in enumerate(('x', 'y', 'z', 'distance'))]
        self.get_logger().info(f'{inp} -> {out} (voxels with {self.min_d} <= ESDF <= '
                               f'{self.max_d if self.max_d > -1 else "inf"} m)')

    def cb(self, g):
        nx, ny, nz = g.nx, g.ny, g.nz
        d = np.asarray(g.distances, dtype=np.float32)
        if d.size != nx * ny * nz or d.size == 0:
            return
        d = d.reshape(nx, ny, nz)                       # index (ix*ny + iy)*nz + iz
        keep = np.abs(d - g.unobserved_value) > 1e-2
        if self.max_d > -1:
            keep &= (d <= self.max_d) & (d >= self.min_d)
        ix, iy, iz = np.nonzero(keep)
        vs = g.voxel_size_m
        pts = np.empty((ix.size, 4), dtype=np.float32)
        pts[:, 0] = g.origin.x + (ix + 0.5) * vs
        pts[:, 1] = g.origin.y + (iy + 0.5) * vs
        pts[:, 2] = g.origin.z + (iz + 0.5) * vs
        pts[:, 3] = d[ix, iy, iz]

        pc = PointCloud2()
        pc.header = g.header
        pc.height = 1
        pc.width = int(pts.shape[0])
        pc.fields = self.fields
        pc.is_bigendian = False
        pc.point_step = 16
        pc.row_step = 16 * pc.width
        pc.is_dense = True
        pc.data = pts.tobytes()
        self.pub.publish(pc)


def main():
    rclpy.init()
    rclpy.spin(EsdfGridToCloud())


if __name__ == '__main__':
    main()
