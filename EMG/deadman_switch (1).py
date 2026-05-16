#!/usr/bin/env python3
"""
Manual deadman publisher.

This node runs on the Raspberry Pi and publishes a Bool on /deadman. Pressing
Enter toggles the signal between True and False so the robot motion node can be
armed or disarmed for testing.
"""

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class DeadmanSwitch(Node):
    def __init__(self):
        super().__init__('deadman_switch')
        self.pub = self.create_publisher(Bool, '/deadman', 10)
        self.state = False
        self.timer = self.create_timer(0.1, self.publish_state)
        self.get_logger().info('Press Enter to toggle deadman ON/OFF')
        self.thread = threading.Thread(target=self.input_loop, daemon=True)
        self.thread.start()

    def publish_state(self):
        msg = Bool()
        msg.data = self.state
        self.pub.publish(msg)

    def input_loop(self):
        while rclpy.ok():
            try:
                input()
            except EOFError:
                break
            self.state = not self.state
            self.get_logger().info(f'Deadman set to {self.state}')


def main():
    rclpy.init()
    node = DeadmanSwitch()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
