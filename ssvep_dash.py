"""

SSVEP Dashboard - Custom Built PsychoPy dashboard for SSVEP stimulu
modes: 
  1. full session: calibration -> training -> live (default)
  2. calibration only: just calibration phase (for collect-only runs on pi)
  3. live only: skip calibration, go straight to live (use with pi --skip-calib)
  4. record mode: one-frequency-at-a-time flashing for offline openbci gui recording

  python3 ssvep_dashboard.py              # full session (calib + train + live)
  python3 ssvep_dashboard.py --calib-only # calibration only, no live
  python3 ssvep_dashboard.py --live-only  # skip calibration, go to live
  python3 ssvep_dashboard.py --record     # manual record mode (use with openbci gui)

controls:
  W = 12 Hz (FORWARD)
  A = 6 Hz  (LEFT)
  S = 10 Hz (BACKWARD)
  D = 7.5 Hz (RIGHT)
  SPACE (hold) = deadman / drive enable
  Q = quit
"""

from psychopy import visual, core, event
import numpy as np
import socket
import json
import time
import argparse
from threading import Thread, Lock

# network config
PI_IP = "10.0.0.113"
MARKER_PORT = 5557
FEEDBACK_PORT = 5555
STATUS_PORT = 5558
DEADMAN_PORT = 5559

# stimulus config
FREQUENCIES = [6.0, 7.5, 10.0, 12.0]
MONITOR_REFRESH = 60

# calibration config
N_TRIALS_PER_FREQ = 8
TRIAL_DURATION = 5.0
INTER_TRIAL_REST = 2.0
INTER_BLOCK_REST = 5.0
PRE_TRIAL_CUE = 1.5

# record mode config
RECORD_FLICKER_SIZE = 0.5

# live mode config
CONFIDENCE_THRESHOLD = 0.50
LIVE_FLICKER_SIZE = 0.7  # big single square for wasd-controlled live mode

# layout: 12=FORWARD (top), 10=BACKWARD (bottom), 6=LEFT, 7.5=RIGHT
POSITIONS = {
    12.0: (0, 0.4),
    6.0: (-0.6, 0),
    7.5: (0.6, 0),
    10.0: (0, -0.4),
}

LABELS = {
    12.0: "12 Hz\nFORWARD",
    6.0: "6 Hz\nLEFT",
    7.5: "7.5 Hz\nRIGHT",
    10.0: "10 Hz\nBACKWARD",
}

# wasd for freq mapping for live mode
WASD_TO_FREQ = {
    "w": 12.0,
    "a": 6.0,
    "s": 10.0,
    "d": 7.5,
}

WASD_DIRECTION = {
    "w": "FORWARD",
    "a": "LEFT",
    "s": "BACKWARD",
    "d": "RIGHT",
}


class SSVEPStimulus:
    def __init__(self, freq, position, size=0.25):
        self.freq = freq
        self.position = position
        self.size = size
        frames_per_cycle = MONITOR_REFRESH / freq
        self.n_frames = int(round(frames_per_cycle * 2))
        self.pattern = []
        for frame in range(self.n_frames):
            phase = (frame / frames_per_cycle) * 2 * np.pi
            on = np.sin(phase) > 0
            self.pattern.append(on)
        self.frame_counter = 0

    def get_color(self):
        return 1.0 if self.pattern[self.frame_counter % self.n_frames] else -1.0

    def advance_frame(self):
        self.frame_counter += 1

    def reset(self):
        self.frame_counter = 0


class MarkerSender:
    def __init__(self, pi_ip, port):
        self.addr = (pi_ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, event_type, **kwargs):
        msg = {"event": event_type, "timestamp": time.time(), **kwargs}
        try:
            self.sock.sendto(json.dumps(msg).encode(), self.addr)
        except Exception as e:
            print(f"marker send error: {e}")


class FeedbackListener(Thread):
    def __init__(self, port=FEEDBACK_PORT):
        super().__init__(daemon=True)
        self.port = port
        self.prediction = None
        self.confidence = 0.0
        self.last_update = 0
        self.running = True
        self.lock = Lock()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                with self.lock:
                    self.prediction = msg.get("frequency")
                    self.confidence = msg.get("confidence", 0.0)
                    self.last_update = time.time()
            except socket.timeout:
                continue
            except Exception as e:
                print(f"feedback listener error: {e}")
        sock.close()

    def get(self):
        with self.lock:
            if time.time() - self.last_update > 2.0:
                return None, 0.0
            return self.prediction, self.confidence


class StatusListener(Thread):
    def __init__(self, port=STATUS_PORT):
        super().__init__(daemon=True)
        self.port = port
        self.status = None
        self.message = ""
        self.running = True
        self.lock = Lock()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                with self.lock:
                    self.status = msg.get("status")
                    self.message = msg.get("message", "")
                print(f"status update: {self.status} - {self.message}")
            except socket.timeout:
                continue
            except Exception as e:
                print(f"status listener error: {e}")
        sock.close()

    def get(self):
        with self.lock:
            return self.status, self.message


class Dashboard:
    def __init__(self):
        self.win = visual.Window(
            size=[1920, 1080],
            fullscr=False,
            monitor="testMonitor",
            units="norm",
            color=[-0.5, -0.5, -0.5],
        )

        self.stimuli = []
        self.squares = []
        self.labels = []
        self.highlights = []
        for freq in FREQUENCIES:
            stim = SSVEPStimulus(freq=freq, position=POSITIONS[freq])
            self.stimuli.append(stim)
            self.squares.append(
                visual.Rect(
                    self.win,
                    width=stim.size,
                    height=stim.size,
                    pos=stim.position,
                    fillColor=[1, 1, 1],
                    lineColor=None,
                )
            )
            self.labels.append(
                visual.TextStim(
                    self.win,
                    text=LABELS[freq],
                    pos=(stim.position[0], stim.position[1] - 0.20),
                    height=0.05,
                    color=[1, 1, 1],
                )
            )
            self.highlights.append(
                visual.Circle(
                    self.win,
                    radius=stim.size * 0.65,
                    pos=stim.position,
                    lineColor=[0, 1, 0],
                    lineWidth=8,
                    fillColor=None,
                )
            )

        self.record_square = visual.Rect(
            self.win,
            width=RECORD_FLICKER_SIZE,
            height=RECORD_FLICKER_SIZE,
            pos=(0, 0),
            fillColor=[1, 1, 1],
            lineColor=None,
        )

        # big single square for wasd-controlled live mode
        self.live_square = visual.Rect(
            self.win,
            width=LIVE_FLICKER_SIZE,
            height=LIVE_FLICKER_SIZE,
            pos=(0, 0),
            fillColor=[1, 1, 1],
            lineColor=None,
        )

        self.instruction = visual.TextStim(
            self.win, text="", pos=(0, 0.85), height=0.08, color=[1, 1, 1], bold=True
        )
        self.feedback = visual.TextStim(
            self.win, text="", pos=(0, -0.85), height=0.06, color=[0.7, 0.7, 0.7]
        )

        self.marker = MarkerSender(PI_IP, MARKER_PORT)
        self.feedback_listener = FeedbackListener()
        self.status_listener = StatusListener()
        self.feedback_listener.start()
        self.status_listener.start()

        # key state, populated by _setup_key_tracking
        self._space_down = False
        self._active_wasd = None  # one of "w","a","s","d" or None

    def check_quit(self):
        if "q" in event.getKeys():
            self.shutdown()
            return True
        return False

    def shutdown(self):
        self.feedback_listener.running = False
        self.status_listener.running = False
        self.win.close()
        core.quit()

    # ---------------- record mode ----------------

    def run_record_mode(self):
        """manual recording to match this protocol:
          block 00: rest (eyes open, blank screen), 60s
          blocks 01-04: 7 Hz,  15s each
          blocks 05-08: 10 Hz, 15s each
          blocks 09-12: 12 Hz, 15s each
          blocks 13-16: 13 Hz, 15s each
          blocks 17-20: 17 Hz, 15s each
        openbci gui saves each recording as a separate file.
        rename each to BrainFlow-RAW_<subject>-<blocknum>_0.csv afterward.
        """
        print("\nrecord mode")
        print("protocol (match these block numbers when renaming openbci files):")
        print("  00:    rest (eyes open), 60s")
        print("  01-04: 7 Hz,  15s each")
        print("  05-08: 10 Hz, 15s each")
        print("  09-12: 12 Hz, 15s each")
        print("  13-16: 13 Hz, 15s each")
        print("  17-20: 17 Hz, 15s each")
        print()
        print("controls:")
        print("  0:   rest mode (blank screen)")
        print("  1-5: jump to freq (7, 10, 12, 13, 17 Hz)")
        print("  N: next freq     P: previous freq")
        print("  SPACE: pause/resume flicker (use while starting/stopping openbci)")
        print("  Q: quit")
        print()

        freq_idx = 0
        rest_mode = False
        paused = False
        single_stim = SSVEPStimulus(FREQUENCIES[freq_idx], (0, 0), RECORD_FLICKER_SIZE)

        # first block number for each mode (there are 4 blocks per freq)
        suggested_blocks = {
            "rest": 0,
            7: 1,
            10: 5,
            12: 9,
            13: 13,
            17: 17,
        }

        clock = core.Clock()
        flicker_start = clock.getTime()

        while True:
            keys = event.getKeys()
            if "q" in keys:
                break
            if "space" in keys:
                paused = not paused
                if not paused:
                    flicker_start = clock.getTime()
                    single_stim.reset()
            if "0" in keys:
                rest_mode = True
                flicker_start = clock.getTime()
            if "n" in keys:
                rest_mode = False
                freq_idx = (freq_idx + 1) % len(FREQUENCIES)
                single_stim = SSVEPStimulus(
                    FREQUENCIES[freq_idx], (0, 0), RECORD_FLICKER_SIZE
                )
                flicker_start = clock.getTime()
            if "p" in keys:
                rest_mode = False
                freq_idx = (freq_idx - 1) % len(FREQUENCIES)
                single_stim = SSVEPStimulus(
                    FREQUENCIES[freq_idx], (0, 0), RECORD_FLICKER_SIZE
                )
                flicker_start = clock.getTime()
            for i, digit in enumerate(["1", "2", "3", "4", "5"]):
                if digit in keys:
                    rest_mode = False
                    freq_idx = i
                    single_stim = SSVEPStimulus(
                        FREQUENCIES[freq_idx], (0, 0), RECORD_FLICKER_SIZE
                    )
                    flicker_start = clock.getTime()

            elapsed = clock.getTime() - flicker_start

            if rest_mode:
                self.instruction.text = "REST  -  eyes open, look at center"
                self.instruction.color = [0.7, 0.7, 0.9]
                self.record_square.fillColor = [-0.4, -0.4, -0.4]
                self.record_square.width = 0.03
                self.record_square.height = 0.03
                self.record_square.draw()
                block_hint = suggested_blocks["rest"]
                target_duration = 60
                status_label = "rest (60s)"
            else:
                current_freq = FREQUENCIES[freq_idx]
                self.record_square.width = RECORD_FLICKER_SIZE
                self.record_square.height = RECORD_FLICKER_SIZE

                if paused:
                    self.record_square.fillColor = [-0.5, -0.5, -0.5]
                    self.instruction.text = f"PAUSED  -  {current_freq} Hz"
                    self.instruction.color = [1, 1, 0]
                else:
                    color = single_stim.get_color()
                    self.record_square.fillColor = [color, color, color]
                    single_stim.advance_frame()
                    self.instruction.text = f"recording  -  {current_freq} Hz"
                    self.instruction.color = [0, 1, 0]
                self.record_square.draw()
                block_hint = suggested_blocks[current_freq]
                target_duration = 15
                status_label = f"{current_freq} Hz (15s per block)"

            self.feedback.text = (
                f"{status_label}   first block: {block_hint:02d}   "
                f"elapsed: {elapsed:.1f}s / target ~{target_duration}s   "
                f"[0] rest  [1-5] jump  [N]ext  [P]rev  [SPACE] pause  [Q]uit"
            )
            self.feedback.color = [0.8, 0.8, 0.8]
            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()

        print("\nrecord mode done")

    # ---------------- calibration mode ----------------

    def draw_static_labels_only(self):
        for label in self.labels:
            label.draw()

    def flicker_frame(self, active_freqs=None):
        for i, stim in enumerate(self.stimuli):
            if active_freqs is None or stim.freq in active_freqs:
                color = stim.get_color()
                self.squares[i].fillColor = [color, color, color]
                stim.advance_frame()
                self.squares[i].draw()
            else:
                self.squares[i].fillColor = [-0.3, -0.3, -0.3]
                self.squares[i].draw()
            self.labels[i].draw()

    def wait_with_countdown(self, duration, message, freq_to_highlight=None):
        """rest / cue period.

        if freq_to_highlight is given, shows a dim square in the center
        (same position the flicker will appear) so the subject's gaze is
        already anchored where the trial will happen. text conveys which
        direction/frequency is coming.

        if None, draws no square at all (used for inter-block rest).
        """
        start = time.time()
        while time.time() - start < duration:
            if self.check_quit():
                return False
            remaining = duration - (time.time() - start)
            self.instruction.text = message
            self.instruction.color = [1, 1, 0]
            self.feedback.text = f"{remaining:.1f}s"
            self.feedback.color = [0.7, 0.7, 0.7]

            if freq_to_highlight is not None:
                # dim center square in the same spot the flicker will appear
                self.live_square.fillColor = [-0.3, -0.3, -0.3]
                self.live_square.draw()

            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()
        return True

    def run_trial(self, target_freq, trial_num, total_trials):
        """flicker the big center square at target_freq for TRIAL_DURATION."""
        # use a fresh stim object so phase always starts clean
        center_stim = SSVEPStimulus(target_freq, (0, 0), LIVE_FLICKER_SIZE)

        self.marker.send(
            "trial_start",
            frequency=target_freq,
            trial_num=trial_num,
            duration=TRIAL_DURATION,
        )

        direction = LABELS[target_freq].replace("\n", " - ")

        start = time.time()
        while time.time() - start < TRIAL_DURATION:
            if self.check_quit():
                return False
            remaining = TRIAL_DURATION - (time.time() - start)

            self.instruction.text = f"focus on {direction}"
            self.instruction.color = [0, 1, 0]
            self.feedback.text = (
                f"trial {trial_num}/{total_trials}  -  {remaining:.1f}s"
            )
            self.feedback.color = [1, 1, 1]

            # one big center square flickering at target freq
            color = center_stim.get_color()
            self.live_square.fillColor = [color, color, color]
            self.live_square.draw()
            center_stim.advance_frame()

            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()

        self.marker.send("trial_end", frequency=target_freq, trial_num=trial_num)
        return True

    def run_calibration(self):
        print("\ncalibration starting")
        print(
            f"{len(FREQUENCIES)} frequencies x {N_TRIALS_PER_FREQ} trials x "
            f"{TRIAL_DURATION}s each"
        )

        self.instruction.text = "calibration"
        self.feedback.text = (
            "a cue will appear, then a big square will flicker in the center\n"
            "focus on the center square\n"
            "starting in 5s..."
        )
        self.feedback.color = [1, 1, 1]
        start = time.time()
        while time.time() - start < 5.0:
            if self.check_quit():
                return False
            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()

        self.marker.send(
            "calibration_start",
            frequencies=FREQUENCIES,
            n_trials=N_TRIALS_PER_FREQ,
            trial_duration=TRIAL_DURATION,
        )

        block_order = FREQUENCIES.copy()
        np.random.shuffle(block_order)

        for block_idx, freq in enumerate(block_order):
            print(f"\nblock {block_idx + 1}/{len(block_order)}: {freq} Hz")
            self.marker.send("block_start", frequency=freq, block_idx=block_idx)

            if not self.wait_with_countdown(
                PRE_TRIAL_CUE, f"next: {freq} Hz - get ready", freq_to_highlight=freq
            ):
                return False

            for trial in range(N_TRIALS_PER_FREQ):
                if not self.run_trial(freq, trial + 1, N_TRIALS_PER_FREQ):
                    return False

                if trial < N_TRIALS_PER_FREQ - 1:
                    if not self.wait_with_countdown(
                        INTER_TRIAL_REST, "rest", freq_to_highlight=freq
                    ):
                        return False

            self.marker.send("block_end", frequency=freq)

            if block_idx < len(block_order) - 1:
                if not self.wait_with_countdown(
                    INTER_BLOCK_REST, "block complete - rest", freq_to_highlight=None
                ):
                    return False

        self.marker.send("calibration_done")
        print("\ncalibration complete")
        return True

    def wait_for_training(self):
        start = time.time()
        while True:
            if self.check_quit():
                return False
            status, message = self.status_listener.get()

            self.instruction.text = "training model..."
            self.instruction.color = [1, 1, 0]
            elapsed = time.time() - start
            self.feedback.text = message if message else f"elapsed: {elapsed:.0f}s"
            self.feedback.color = [0.7, 0.7, 0.7]

            self.draw_static_labels_only()
            for i, stim in enumerate(self.stimuli):
                self.squares[i].fillColor = [-0.3, -0.3, -0.3]
                self.squares[i].draw()

            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()

            if status == "ready":
                return True
            if status == "error":
                self.instruction.text = f"training failed: {message}"
                self.instruction.color = [1, 0, 0]
                self.instruction.draw()
                self.win.flip()
                core.wait(5.0)
                return False

    def _setup_key_tracking(self):
        # register pyglet key event handlers for deadman and wasd.
        self._space_down = False
        self._active_wasd = None
        try:
            import pyglet.window.key as pkey

            win = self.win.backend.winHandle

            # map pyglet symbol to our wasd letter
            wasd_symbols = {
                pkey.W: "w",
                pkey.A: "a",
                pkey.S: "s",
                pkey.D: "d",
            }

            @win.event
            def on_key_press(symbol, modifiers):
                if symbol == pkey.SPACE:
                    self._space_down = True
                elif symbol in wasd_symbols:
                    # latch: each press sets the active direction
                    self._active_wasd = wasd_symbols[symbol]

            @win.event
            def on_key_release(symbol, modifiers):
                if symbol == pkey.SPACE:
                    self._space_down = False
                # wasd is latched: releasing the key does not clear it.
                # press a different wasd key to switch, or no flicker until you do.

            print("[keys] pyglet key tracking enabled (space + wasd)")
        except Exception as e:
            print(f"[keys] pyglet key tracking failed: {e}")
            print(f"[keys] falling back to event.getKeys (less reliable)")

    def send_deadman(self, enabled):
        """send deadman state to pi over udp."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.sendto(
                json.dumps({"enabled": bool(enabled)}).encode(), (PI_IP, DEADMAN_PORT)
            )
            sock.close()
        except Exception as e:
            print(f"deadman send error: {e}")

    def run_live(self):
        """live mode: single big square flickering at one freq, chosen by wasd.

        controls:
          w/a/s/d  - pick which freq to flicker (12 / 6 / 10 / 7.5 Hz)
          space    - hold to enable robot motion (deadman button)
          q        - quit
        """
        print("\n live mode")
        print("press W/A/S/D to pick a frequency to flicker:")
        print("  W = 12 Hz (FORWARD)")
        print("  A =  6 Hz (LEFT)")
        print("  S = 10 Hz (BACKWARD)")
        print("  D = 7.5 Hz (RIGHT)")
        print("HOLD SPACEBAR to enable robot motion. RELEASE to stop.")
        print("Q to quit.")

        self.marker.send("live_start")
        self._setup_key_tracking()

        self.send_deadman(False)
        last_deadman_state = False
        last_deadman_send_time = 0.0
        DEADMAN_HEARTBEAT_SEC = 0.2

        # per-frequency stim objects so each freq keeps its own phase counter.
        # building them up-front avoids resetting phase every time you switch.
        live_stims = {
            letter: SSVEPStimulus(freq, (0, 0), LIVE_FLICKER_SIZE)
            for letter, freq in WASD_TO_FREQ.items()
        }

        last_active = None
        last_marker_freq = None

        while True:
            if self.check_quit():
                self.send_deadman(False)
                return

            # ---- deadman ----
            space_down = getattr(self, "_space_down", False)
            now = time.time()
            if (
                space_down != last_deadman_state
                or now - last_deadman_send_time > DEADMAN_HEARTBEAT_SEC
            ):
                self.send_deadman(space_down)
                last_deadman_state = space_down
                last_deadman_send_time = now

            # ---- wasd selection ----
            active = getattr(self, "_active_wasd", None)

            # on first selection or change, reset that freq's phase and send marker
            if active is not None and active != last_active:
                live_stims[active].reset()
                freq = WASD_TO_FREQ[active]
                if freq != last_marker_freq:
                    self.marker.send("live_freq_select", frequency=freq, key=active)
                    last_marker_freq = freq
            last_active = active

            # ---- draw ----
            if active is None:
                # nothing selected yet: dim placeholder square
                self.live_square.fillColor = [-0.4, -0.4, -0.4]
                self.live_square.draw()
                self.instruction.text = "press W / A / S / D to start flickering"
                self.instruction.color = [1, 1, 0]
            else:
                stim = live_stims[active]
                color = stim.get_color()
                self.live_square.fillColor = [color, color, color]
                stim.advance_frame()
                self.live_square.draw()

                freq = WASD_TO_FREQ[active]
                direction = WASD_DIRECTION[active]
                drive_str = "[DRIVE ENABLED]" if space_down else "[HOLD SPACE TO DRIVE]"
                self.instruction.text = f"{freq} Hz  -  {direction}   {drive_str}"
                self.instruction.color = [0, 1, 0] if space_down else [1, 1, 1]

            # ---- classifier feedback (optional, still useful for debugging) ----
            pred, conf = self.feedback_listener.get()
            if pred is not None and conf >= CONFIDENCE_THRESHOLD:
                self.feedback.text = f"classifier: {pred} Hz  -  confidence: {conf:.2f}"
                self.feedback.color = [0, 1, 0]
            elif pred is not None:
                self.feedback.text = f"classifier weak: {pred} Hz  -  {conf:.2f}"
                self.feedback.color = [1, 1, 0]
            else:
                self.feedback.text = "waiting for classifier signal..."
                self.feedback.color = [0.5, 0.5, 0.5]

            self.instruction.draw()
            self.feedback.draw()
            self.win.flip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--calib-only", action="store_true", help="run calibration only (no live phase)"
    )
    parser.add_argument(
        "--live-only",
        action="store_true",
        help="skip calibration, go straight to live mode "
        "(use with pi's --skip-calib)",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="manual record mode for offline openbci gui recording",
    )
    args = parser.parse_args()

    dashboard = Dashboard()
    try:
        if args.record:
            dashboard.run_record_mode()
            return

        if args.live_only:
            # skip calibration entirely, go to live
            dashboard.run_live()
            return

        if not dashboard.run_calibration():
            return

        if args.calib_only:
            dashboard.instruction.text = "calibration done - press Q to quit"
            dashboard.feedback.text = ""
            dashboard.instruction.draw()
            dashboard.win.flip()
            while not dashboard.check_quit():
                core.wait(0.1)
            return

        if not dashboard.wait_for_training():
            return
        dashboard.run_live()
    finally:
        dashboard.shutdown()


if __name__ == "__main__":
    main()
