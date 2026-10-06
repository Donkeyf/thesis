import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
from mavros_msgs.srv import MessageInterval
from rclpy.node import Node


class MavlinkMiddleware(Node):

    def __init(self):
        super().__init__('task_control')

        # Service clients
        self.declare_parameter("message_id", 27)       # RAW_IMU
        self.declare_parameter("rate_hz", 200.0)
        self.msg_interval_client = self.create_client(MessageInterval, "/mavros/set_message_interval")
        self.done = False
        self.timer = self.create_timer(2.0, self.try_set)  # retry every 2 s


        
    # setter function for service
    def try_set(self):
        if self.done or not self.cli.service_is_ready():
            return
        req = MessageInterval.Request()
        req.message_id = self.get_parameter("message_id").value
        req.message_rate = float(self.get_parameter("rate_hz").value)
        self.cli.call_async(req).add_done_callback(self.on_result)


    # check if parameter set
    def on_result(self, future):
        res = future.result()
        if res is not None and res.success:
            self.get_logger().info("Message interval set")
            self.done = True
        else:
            self.get_logger().warn("FC rejected the request, will retry")

