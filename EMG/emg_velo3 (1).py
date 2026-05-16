#!/usr/bin/env python3
"""
EMG-to-robot motion node.

This node runs on the robot computer. It listens to one-hot EMG commands on
/emg_commands and a deadman signal on /deadman, reads live joint states,
captures a startup home pose, and sends bounded joint trajectory commands to
move the arm. It maps Forearm, Bicep, and Deltoid to joints 5, 6, and 7.
Thumb can toggle the gripper if that controller path is working.
"""

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray, Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import GripperCommand

JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6', 'joint_7']
FOREARM_JOINT = 4
BICEP_JOINT = 5
DELTOID_JOINT = 6
JOINT_DIRECTIONS = [1, 1, 1, 1, -1, 1, 1]
STEP = 0.10
SOFT_WINDOW = 0.20
MOVE_TIME_SEC = 0
MOVE_TIME_NSEC = 250000000
GRIPPER_COOLDOWN = 2.0
GRIPPER_CLOSED_POS = 1.0
GRIPPER_OPEN_POS = 0.0
GRIPPER_MAX_EFFORT = 100.0
PRINT_DEBUG = True


class EMGVelocity(Node):
    def __init__(self):
        super().__init__('emg_velocity')
        self.deadman_active = False
        self.current_positions = None
        self.home_positions = None
        self.emg_state = [0, 0, 0, 0]
        self.prev_emg_state = [0, 0, 0, 0]
        self.joint_directions = list(JOINT_DIRECTIONS)
        self.gripper_closed = False
        self.last_gripper_toggle = 0.0

        self.publisher = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 1)
        self.gripper_client = ActionClient(self, GripperCommand, '/robotiq_gripper_controller/gripper_cmd')

        self.create_subscription(JointState, '/joint_states', self.joint_callback, qos_profile_sensor_data)
        self.create_subscription(JointState, '/a200_0652/platform/joint_states', self.joint_callback, qos_profile_sensor_data)
        self.create_subscription(Int32MultiArray, '/emg_commands', self.emg_callback, 10)
        self.create_subscription(Bool, '/deadman', self.deadman_callback, 10)

        self.log('EMG robot controller ready')
        self.log('Waiting for joint states, EMG, and deadman')
        self.log(f'STEP={STEP} SOFT_WINDOW={SOFT_WINDOW} MOVE_TIME={MOVE_TIME_SEC}.{MOVE_TIME_NSEC:09d}')

    def log(self, msg):
        if PRINT_DEBUG:
            self.get_logger().info(msg)

    def joint_callback(self, msg: JointState):
        if len(msg.position) >= 7:
            self.current_positions = list(msg.position[:7])
            if self.home_positions is None:
                self.home_positions = list(msg.position[:7])
                self.log(f'Captured home positions: {self.home_positions}')

    def deadman_callback(self, msg: Bool):
        self.deadman_active = bool(msg.data)
        self.log(f'Deadman: {self.deadman_active}')

    def emg_callback(self, msg: Int32MultiArray):
        vals = list(msg.data[:4])
        while len(vals) < 4:
            vals.append(0)
        self.prev_emg_state = self.emg_state
        self.emg_state = vals
        self.log(f'EMG state: {self.emg_state}')

        if not self.deadman_active:
            return
        if self.current_positions is None or self.home_positions is None:
            self.log('Skipping motion: no joint states/home yet')
            return

        rising = [self.prev_emg_state[i] == 0 and self.emg_state[i] == 1 for i in range(4)]

        if rising[0]:
            self.move_joint(FOREARM_JOINT)
        elif rising[1]:
            self.move_joint(BICEP_JOINT)
        elif rising[2]:
            self.move_joint(DELTOID_JOINT)
        elif rising[3]:
            self.toggle_gripper()

    def move_joint(self, joint_index: int):
        current = self.current_positions[joint_index]
        home = self.home_positions[joint_index]
        direction = self.joint_directions[joint_index]
        min_limit = home - SOFT_WINDOW
        max_limit = home + SOFT_WINDOW
        new_pos = current + STEP * direction

        if new_pos >= max_limit:
            new_pos = max_limit
            self.joint_directions[joint_index] = -1
            self.log(f'Joint {joint_index + 1} hit soft max {max_limit:.3f}, reversing')
        elif new_pos <= min_limit:
            new_pos = min_limit
            self.joint_directions[joint_index] = 1
            self.log(f'Joint {joint_index + 1} hit soft min {min_limit:.3f}, reversing')

        positions = list(self.current_positions)
        positions[joint_index] = new_pos

        traj = JointTrajectory()
        traj.joint_names = JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = MOVE_TIME_SEC
        point.time_from_start.nanosec = MOVE_TIME_NSEC
        traj.points = [point]

        self.log(f'Publishing joint {joint_index + 1} -> {new_pos:.4f}')
        self.publisher.publish(traj)

    def toggle_gripper(self):
        now = time.time()
        if now - self.last_gripper_toggle < GRIPPER_COOLDOWN:
            self.log('Skipping gripper toggle: cooldown')
            return

        goal = GripperCommand.Goal()
        if self.gripper_closed:
            goal.command.position = GRIPPER_OPEN_POS
            self.log('Opening gripper')
        else:
            goal.command.position = GRIPPER_CLOSED_POS
            self.log('Closing gripper')
        goal.command.max_effort = GRIPPER_MAX_EFFORT

        self.gripper_client.wait_for_server(timeout_sec=1.0)
        self.gripper_client.send_goal_async(goal)
        self.gripper_closed = not self.gripper_closed
        self.last_gripper_toggle = now


def main():
    rclpy.init()
    node = EMGVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
