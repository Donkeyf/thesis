#!/usr/bin/env python3
"""Test publisher for trajectory_tracker: re-plans a horizontal circle at 10 Hz,
mimicking a spline planner (MultiDOFJointTrajectory with pos/vel/acc/yaw/yaw-rate).

  python3 test_circle_trajectory.py --center 0 0 1.0 --radius 0.5 --period 12

Frame: EKF local ENU (map). The circle starts at the point nearest the vehicle's
current angle around the centre, so it doesn't jump. Ctrl-C stops publishing; the
tracker then holds position after traj_timeout.
"""
import argparse
import math

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped, Quaternion, Transform, Twist, Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint


def quat_yaw(y):
    return Quaternion(x=0.0, y=0.0, z=math.sin(y / 2), w=math.cos(y / 2))


class Circle(Node):
    def __init__(self, a):
        super().__init__('test_circle_trajectory')
        self.a = a
        self.w = 2 * math.pi / a.period          # rad/s
        self.phase = None                         # angle at t0
        self.t0 = None
        self.pub = self.create_publisher(MultiDOFJointTrajectory, a.topic, 10)
        self.create_subscription(PoseStamped, a.pose_topic, self.on_pose, qos_profile_sensor_data)
        self.pose = None
        self.create_timer(1.0 / a.replan_hz, self.replan)
        speed = self.w * a.radius
        self.get_logger().info(
            f'circle r={a.radius} m, period {a.period} s -> {speed:.2f} m/s, '
            f'{self.w * speed:.2f} m/s^2 centripetal; publishing {a.topic} at {a.replan_hz} Hz')

    def on_pose(self, m):
        self.pose = m

    def replan(self):
        now = self.get_clock().now()
        tn = now.nanoseconds * 1e-9
        if self.t0 is None:
            if self.pose is None:
                self.get_logger().info('waiting for local_position/pose...', once=True)
                return
            p = self.pose.pose.position
            self.phase = math.atan2(p.y - self.a.center[1], p.x - self.a.center[0])
            self.t0 = tn
        msg = MultiDOFJointTrajectory()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = 'map'
        cx, cy, cz = self.a.center
        r, w = self.a.radius, self.w
        n = int(self.a.horizon / self.a.dt) + 1
        for k in range(n):
            tau = k * self.a.dt
            th = self.phase + w * (tn - self.t0 + tau)
            pt = MultiDOFJointTrajectoryPoint()
            pt.time_from_start = Duration(sec=int(tau), nanosec=int((tau % 1) * 1e9))
            yaw = th + math.pi / 2 if self.a.face_tangent else 0.0
            pt.transforms.append(Transform(
                translation=Vector3(x=cx + r * math.cos(th), y=cy + r * math.sin(th), z=cz),
                rotation=quat_yaw(yaw)))
            pt.velocities.append(Twist(
                linear=Vector3(x=-r * w * math.sin(th), y=r * w * math.cos(th), z=0.0),
                angular=Vector3(z=w if self.a.face_tangent else 0.0)))
            pt.accelerations.append(Twist(
                linear=Vector3(x=-r * w * w * math.cos(th), y=-r * w * w * math.sin(th), z=0.0)))
            msg.points.append(pt)
        self.pub.publish(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--center', type=float, nargs=3, default=[0.0, 0.0, 1.0])
    ap.add_argument('--radius', type=float, default=0.5)
    ap.add_argument('--period', type=float, default=12.0, help='s per lap')
    ap.add_argument('--replan-hz', type=float, default=10.0)
    ap.add_argument('--horizon', type=float, default=2.0, help='s of trajectory per message')
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--face-tangent', action='store_true', help='yaw along direction of travel')
    ap.add_argument('--topic', default='/trajectory_tracker/trajectory')
    ap.add_argument('--pose-topic', default='/mavros/mavros/local_position/pose')
    a = ap.parse_args()
    rclpy.init()
    node = Circle(a)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    rclpy.shutdown()


if __name__ == '__main__':
    main()
