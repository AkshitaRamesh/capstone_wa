#!/usr/bin/env python3
"""
Direct bounded joint demo loop.

This node runs on the robot and moves the mapped arm joints directly without
using live EMG or emg_velo3.py. It captures the startup home pose, then sweeps
Deltoid, Bicep, and Forearm joints around home in a small bounded pattern and
repeats forever.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
FOREARM_JOINT = 4
BICEP_JOINT = 5
DELTOID_JOINT = 6
SWEEP = 0.12
MOVE_TIME_SEC = 0
MOVE_TIME_NSEC = 400000000
PAUSE_BETWEEN = 0.4
SEQUENCE = [
    ('Deltoid', DELTOID_JOINT),
    ('Bicep', BICEP_JOINT),
    ('Forearm', FOREARM_JOINT),
]


class JointDemo(Node):
    def __init__(self):
        super().__init__('joint_demo_loop')
        self.current_positions = None
        self.home_positions = None
        self.pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)
        self.create_subscription(JointState, '/joint_states', self.joint_callback, qos_profile_sensor_data)
        self.create_subscription(JointState, '/a200_0652/platform/joint_states', self.joint_callback, qos_profile_sensor_data)
        self.get_logger().info('Waiting for joint states...')

    def joint_callback(self, msg):
        if len(msg.position) >= 7:
            self.current_positions = list(msg.position[:7])
            if self.home_positions is None:
                self.home_positions = list(msg.position[:7])
                self.get_logger().info(f'Captured home positions: {self.home_positions}')

    def send_positions(self, positions):
        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = MOVE_TIME_SEC
        point.time_from_start.nanosec = MOVE_TIME_NSEC
        traj.points = [point]
        self.pub.publish(traj)

    def sweep_joint(self, name, joint_index):
        base = list(self.current_positions if self.current_positions else self.home_positions)
        home = self.home_positions[joint_index]

        pos1 = list(base)
        pos1[joint_index] = home + SWEEP
        self.get_logger().info(f'{name}: +sweep')
        self.send_positions(pos1)
        time.sleep(0.8)

        pos2 = list(base)
        pos2[joint_index] = home - SWEEP
        self.get_logger().info(f'{name}: -sweep')
        self.send_positions(pos2)
        time.sleep(0.8)

        pos3 = list(base)
        pos3[joint_index] = home
        self.get_logger().info(f'{name}: home')
        self.send_positions(pos3)
        time.sleep(PAUSE_BETWEEN)

    def run(self):
        while rclpy.ok() and self.home_positions is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        self.get_logger().info('Starting bounded demo loop. Ctrl+C to stop.')
        while rclpy.ok():
            for name, joint_index in SEQUENCE:
                for _ in range(5):
                    rclpy.spin_once(self, timeout_sec=0.05)
                self.sweep_joint(name, joint_index)


def main():
    rclpy.init()
    node = JointDemo()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
