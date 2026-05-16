#!/usr/bin/env python3
"""
Direct gripper action test utility.

This node runs on the robot and sends a single open or close goal to the
Robotiq gripper action server. It is useful for checking whether the gripper
hardware path works independently of EMG or the main robot motion node.
"""

import sys

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import GripperCommand


class GripperTester(Node):
    def __init__(self):
        super().__init__('gripper_tester')
        self.client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

    def send(self, position: float, max_effort: float):
        self.get_logger().info('Waiting for gripper action server...')
        if not self.client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Gripper action server not available')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = max_effort

        self.get_logger().info(f'Sending goal: position={position}, max_effort={max_effort}')
        future = self.client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, future)

        goal_handle = future.result()
        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error('Goal was rejected')
            return False

        self.get_logger().info('Goal accepted, waiting for result...')
        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)
        result = result_future.result()
        self.get_logger().info(f'Result: {result}')
        return True


def main():
    rclpy.init()
    node = GripperTester()
    try:
        mode = sys.argv[1] if len(sys.argv) > 1 else 'close'
        if mode == 'close':
            node.send(1.0, 100.0)
        elif mode == 'open':
            node.send(0.0, 100.0)
        else:
            print('Usage: python3 /home/robot/test_gripper.py [open|close]')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
