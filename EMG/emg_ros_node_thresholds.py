#!/usr/bin/env python3
"""
EMG ROS threshold node.

This node runs on the Raspberry Pi connected to the OpenBCI Cyton. It loads
calibrated thresholds from emg_thresholds.json, processes incoming EMG samples,
and publishes winner-take-most one-hot commands on /emg_commands. This version
is configured to ignore the thumb channel and only select the best signal among
Forearm, Bicep, and Deltoid.
"""

import json
import time
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from emg_processor_brainflow import EMGProcessor

JSON_PATH = Path('/home/capstone/emg_thresholds.json')
SERIAL_PORT = '/dev/ttyUSB0'
MUSCLES = ["Forearm", "Bicep", "Deltoid", "Thumb"]


class EMGNodeThresholds(Node):
    def __init__(self):
        super().__init__('emg_node_thresholds')
        self.publisher = self.create_publisher(Int32MultiArray, '/emg_commands', 10)

        data = json.loads(JSON_PATH.read_text())
        self.baseline = [float(x) for x in data['baseline']]
        self.threshold_on = [float(x) for x in data['threshold_on']]
        self.threshold_off = [float(x) for x in data['threshold_off']]
        self.active_levels = [float(x) for x in data.get('active_median', data['threshold_on'])]

        self.get_logger().info(f'Loaded thresholds from {JSON_PATH}')
        self.get_logger().info('Thumb disabled: winner-take-most uses Forearm/Bicep/Deltoid only')
        for i, name in enumerate(MUSCLES):
            self.get_logger().info(
                f'{name}: baseline={self.baseline[i]:.2f} '
                f'on={self.threshold_on[i]:.2f} off={self.threshold_off[i]:.2f} '
                f'active={self.active_levels[i]:.2f}'
            )

        params = BrainFlowInputParams()
        params.serial_port = SERIAL_PORT
        self.board = BoardShim(BoardIds.CYTON_BOARD.value, params)

        self.processor = EMGProcessor(sampling_rate=250, num_channels=4)
        self.processor.set_calibration(self.baseline, self.active_levels)
        self.processor.set_normalized_thresholds(on=0.25, off=0.15)
        self.processor.set_winner_take_most(enabled=False)

        self.current_winner = None
        self.switch_margin = 0.20

        self.board.prepare_session()
        self.board.start_stream()
        time.sleep(1.5)
        self.board.get_board_data()

        self.get_logger().info(f'OpenBCI Cyton streaming on {SERIAL_PORT}')
        self.timer = self.create_timer(0.02, self.read_emg)

    def choose_winner(self, smoothed, norm):
        candidates = [i for i in range(3) if smoothed[i] >= self.threshold_on[i]]

        proposed = None
        if candidates:
            ranked = sorted(candidates, key=lambda i: norm[i], reverse=True)
            top = ranked[0]
            top_val = norm[top]
            second_val = norm[ranked[1]] if len(ranked) > 1 else -1.0
            if len(ranked) == 1 or (top_val - second_val) >= self.switch_margin:
                proposed = top

        if self.current_winner is not None:
            if self.current_winner > 2 or smoothed[self.current_winner] < self.threshold_off[self.current_winner]:
                self.current_winner = None

        if self.current_winner is None and proposed is not None:
            self.current_winner = proposed
        elif self.current_winner is not None and proposed is not None and proposed != self.current_winner:
            if norm[proposed] > norm[self.current_winner] + self.switch_margin:
                self.current_winner = proposed

        out = [0, 0, 0, 0]
        if self.current_winner is not None:
            out[self.current_winner] = 1
        out[3] = 0
        return out

    def read_emg(self):
        data = self.board.get_board_data()
        if data.shape[1] == 0:
            return

        eeg_channels = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
        latest_smoothed = None
        latest_norm = None

        for sample_idx in range(data.shape[1]):
            raw_emg = [float(data[eeg_channels[i], sample_idx]) for i in range(4)]
            _, smoothed, _ = self.processor.process_sample(raw_emg)
            latest_smoothed = smoothed
            latest_norm = self.processor.compute_normalized(smoothed)

        if latest_smoothed is None:
            return

        winner_state = self.choose_winner(latest_smoothed, latest_norm)
        msg = Int32MultiArray()
        msg.data = winner_state
        self.publisher.publish(msg)

    def destroy_node(self):
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception as e:
            self.get_logger().warn(f'BrainFlow cleanup error: {e}')
        super().destroy_node()


def main():
    rclpy.init()
    node = EMGNodeThresholds()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
