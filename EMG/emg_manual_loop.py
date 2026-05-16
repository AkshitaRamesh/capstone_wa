#!/usr/bin/env python3
"""
Manual EMG command loop.

This node runs on the robot and continuously publishes fake EMG commands and a
True deadman signal to exercise the motion pipeline without live EMG. It is
meant to be used together with emg_velo3.py. The loop cycles Deltoid, Bicep,
and Forearm with short rest gaps between them.
"""

import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray, Bool

SEQUENCE = [
    ('Deltoid', [0, 0, 1, 0], 0.8),
    ('Rest', [0, 0, 0, 0], 0.25),
    ('Bicep', [0, 1, 0, 0], 0.8),
    ('Rest', [0, 0, 0, 0], 0.25),
    ('Forearm', [1, 0, 0, 0], 0.8),
    ('Rest', [0, 0, 0, 0], 0.25),
]
PUBLISH_RATE_HZ = 20.0


class ManualLoop(Node):
    def __init__(self):
        super().__init__('emg_manual_loop')
        self.cmd_pub = self.create_publisher(Int32MultiArray, '/emg_commands', 10)
        self.deadman_pub = self.create_publisher(Bool, '/deadman', 10)
        self.get_logger().info('Manual EMG loop ready')
        self.get_logger().info('Fast infinite sequence: Deltoid -> Bicep -> Forearm')

    def publish_deadman(self, state: bool):
        msg = Bool()
        msg.data = state
        self.deadman_pub.publish(msg)

    def publish_command(self, data):
        msg = Int32MultiArray()
        msg.data = data
        self.cmd_pub.publish(msg)

    def hold(self, data, duration):
        period = 1.0 / PUBLISH_RATE_HZ
        end_time = time.time() + duration
        while time.time() < end_time and rclpy.ok():
            self.publish_deadman(True)
            self.publish_command(data)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(period)

    def run(self):
        self.get_logger().info('Starting in 1 second...')
        self.hold([0, 0, 0, 0], 1.0)
        while rclpy.ok():
            for name, cmd, duration in SEQUENCE:
                self.get_logger().info(f'{name}: {cmd} for {duration:.2f}s')
                self.hold(cmd, duration)

    def stop_outputs(self):
        self.publish_command([0, 0, 0, 0])
        self.publish_deadman(False)


def main():
    rclpy.init()
    node = ManualLoop()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_outputs()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
