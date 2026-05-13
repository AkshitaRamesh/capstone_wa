"""
full_bci_pipeline.py
4-class SSVEP {6, 7.5, 10, 12 Hz} with Husky ROS2 control.

husky mapping:
  12 Hz  -> FORWARD
  10 Hz  -> BACKWARD
   6 Hz  -> LEFT
  15 Hz  -> RIGHT

deadman: spacebar held in dashboard + BCI pred within 500ms

cmds:
  python3 full_bci_pipeline.py                 - full session
  python3 full_bci_pipeline.py --no-ros        - no husky publishing
  python3 full_bci_pipeline.py --collect-only  - record session, exit
  python3 full_bci_pipeline.py --skip-calib    - skip calibration, use existing model
"""

import argparse
import socket
import json
import time
import threading
import queue
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from collections import deque

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, NoiseTypes
from sklearn.svm import SVC
from sklearn.cross_decomposition import CCA
from scipy.signal import welch

DASHBOARD_IP = "10.0.0.30"
MARKER_PORT = 5557
FEEDBACK_PORT = 5555
STATUS_PORT = 5558
DEADMAN_PORT = 5559

HUSKY_TOPIC = "/a200_0652/cmd_vel"
PUBLISH_RATE_HZ = 20
DEADMAN_TIMEOUT_SEC = 0.5
CMD_FWD_SPEED = 0.15
CMD_TURN_SPEED = 0.3

FREQ_TO_TWIST = {
    12.0: (CMD_FWD_SPEED, 0.0),  # FORWARD
    10.0: (-CMD_FWD_SPEED, 0.0),  # BACKWARD
    6.0: (0.0, CMD_TURN_SPEED),  # LEFT
    7.5: (0.0, -CMD_TURN_SPEED),  # RIGHT
}

STIMULUS_FREQUENCIES = [6.0, 7.5, 10.0, 12.0]
FBCCA_SUBBANDS = [4.0, 8.0, 14.0, 20.0, 28.0]
N_HARMONICS = 2
SAMPLE_RATE = 125
WINDOW_SEC = 2.0
WINDOW_SAMPLES = int(WINDOW_SEC * SAMPLE_RATE)

LIVE_WINDOW_SEC = 3.0
LIVE_WINDOW_SAMPLES = int(LIVE_WINDOW_SEC * SAMPLE_RATE)

STEP_SEC = 0.25

# per-frequency confidence thresholds. higher threshold = fewer false positives
PER_FREQ_THRESHOLD = {
    6.0: 0.70,
    7.5: 0.60,
    10.0: 0.60,
    12.0: 0.50,
}
DEFAULT_THRESHOLD = 0.60


TRIAL_SKIP_SEC = 0.5
TRIAL_SKIP_SAMPLES = int(TRIAL_SKIP_SEC * SAMPLE_RATE)

BANDPASS_LOW = 4.0
BANDPASS_HIGH = 40.0
NOTCH_FREQ = 60.0

N_CAR_CHANNELS = 14
ALIGNED_14 = [
    "Fp1",
    "Fp2",
    "C3",
    "C4",
    "P7",
    "P8",
    "O1",
    "O2",
    "F7",
    "F8",
    "F3",
    "F4",
    "P3",
    "P4",
]
N_FEATURE_CHANNELS = 8
SSVEP_INDICES = [2, 3, 4, 5, 6, 7, 12, 13]

DRY_COL_TO_ALIGNED = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9,
    10: 10,
    11: 11,
    14: 12,
    15: 13,
}

WET_PRETRAINED_PATH = Path("wet_pretrained_model.pkl")
FINETUNED_PATH = Path("dry_finetuned_model.pkl")
SESSIONS_DIR = Path("dry_sessions")


# ROS publisher


class HuskyPublisher:
    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import TwistStamped

        self.rclpy = rclpy
        self.TwistStamped = TwistStamped

        rclpy.init(args=None)
        self.node = Node("bci_husky_publisher")
        self.pub = self.node.create_publisher(TwistStamped, HUSKY_TOPIC, 10)

        self._lock = threading.Lock()
        self._target_linear = 0.0
        self._target_angular = 0.0
        self._last_bci_time = 0.0
        self._deadman_enabled = False
        self._running = True

        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        print(f"[ROS] publishing to {HUSKY_TOPIC} at {PUBLISH_RATE_HZ} Hz")
        print(f"[ROS] deadman DISABLED by default (hold space in dashboard to enable)")

    def update_command(self, linear, angular):
        with self._lock:
            self._target_linear = linear
            self._target_angular = angular
            self._last_bci_time = time.time()

    def set_deadman(self, enabled):
        with self._lock:
            if enabled != self._deadman_enabled:
                print(f"[DEADMAN] {'ENABLED' if enabled else 'DISABLED'}")
            self._deadman_enabled = enabled

    def _publish_loop(self):
        period = 1.0 / PUBLISH_RATE_HZ
        while self._running:
            with self._lock:
                deadman = self._deadman_enabled
                last = self._last_bci_time
                lin = self._target_linear
                ang = self._target_angular

            age = time.time() - last
            ok = deadman and age < DEADMAN_TIMEOUT_SEC
            gated_lin = lin if ok else 0.0
            gated_ang = ang if ok else 0.0

            msg = self.TwistStamped()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.header.frame_id = "base_link"
            msg.twist.linear.x = gated_lin
            msg.twist.angular.z = gated_ang
            self.pub.publish(msg)
            time.sleep(period)

    def shutdown(self):
        self._running = False
        try:
            self.node.destroy_node()
            self.rclpy.shutdown()
        except Exception:
            pass


class DeadmanListener(threading.Thread):
    def __init__(self, publisher):
        super().__init__(daemon=True)
        self.publisher = publisher
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", DEADMAN_PORT))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                self.publisher.set_deadman(bool(msg.get("enabled", False)))
            except socket.timeout:
                continue
            except Exception as e:
                print(f"deadman listener error: {e}")
        sock.close()


#  UDP helpers


def send_status(status, message=""):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(
            json.dumps({"status": status, "message": message}).encode(),
            (DASHBOARD_IP, STATUS_PORT),
        )
    except Exception as e:
        print(f"status send error: {e}")
    sock.close()


def send_prediction(freq, confidence):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # send as float since 7.5 isn't int
        sock.sendto(
            json.dumps(
                {"frequency": float(freq), "confidence": float(confidence)}
            ).encode(),
            (DASHBOARD_IP, FEEDBACK_PORT),
        )
    except Exception as e:
        print(f"prediction send error: {e}")
    sock.close()


def remap_to_14(raw_16):
    aligned = np.zeros((N_CAR_CHANNELS, raw_16.shape[1]))
    for src_col, aligned_idx in DRY_COL_TO_ALIGNED.items():
        aligned[aligned_idx] = raw_16[src_col]
    return aligned


#  features


def preprocess_8ch(data_8):
    data_8 = data_8.copy().astype(np.float64)
    for ch in range(data_8.shape[0]):
        DataFilter.remove_environmental_noise(
            data_8[ch], SAMPLE_RATE, NoiseTypes.SIXTY.value
        )
        DataFilter.perform_bandpass(
            data_8[ch],
            SAMPLE_RATE,
            BANDPASS_LOW,
            BANDPASS_HIGH,
            4,
            FilterTypes.BUTTERWORTH.value,
            0,
        )
    for ch in range(data_8.shape[0]):
        mu, sigma = data_8[ch].mean(), data_8[ch].std()
        if sigma > 0:
            data_8[ch] = np.clip(data_8[ch], mu - 6 * sigma, mu + 6 * sigma)
    return data_8 - data_8.mean(axis=0, keepdims=True)


def subband_filter(data_8_raw, low, high):
    out = data_8_raw.copy().astype(np.float64)
    for ch in range(out.shape[0]):
        DataFilter.perform_bandpass(
            out[ch], SAMPLE_RATE, low, high, 4, FilterTypes.BUTTERWORTH.value, 0
        )
    return out - out.mean(axis=0, keepdims=True)


def build_cca_refs(n_samples):
    t = np.arange(n_samples) / SAMPLE_RATE
    refs = {}
    for f in STIMULUS_FREQUENCIES:
        components = []
        for h in range(1, N_HARMONICS + 1):
            components.append(np.sin(2 * np.pi * h * f * t))
            components.append(np.cos(2 * np.pi * h * f * t))
        refs[f] = np.array(components).T
    return refs


def cca_correlations(data_8, refs):
    X = data_8.T
    out = []
    for f in STIMULUS_FREQUENCIES:
        Y = refs[f]
        try:
            cca = CCA(n_components=1, max_iter=1000)
            cca.fit(X, Y)
            xc, yc = cca.transform(X, Y)
            out.append(abs(np.corrcoef(xc.T, yc.T)[0, 1]))
        except Exception:
            out.append(0.0)
    return np.array(out)


def fbcca_correlations(data_8_raw, refs):
    out = []
    for sb_low in FBCCA_SUBBANDS:
        try:
            filtered = subband_filter(data_8_raw, sb_low, BANDPASS_HIGH)
            corrs = cca_correlations(filtered, refs)
            out.extend(corrs.tolist())
        except Exception:
            out.extend([0.0] * len(STIMULUS_FREQUENCIES))
    return np.array(out)


def fft_features(data_8, nperseg):
    feats = []
    for ch in range(data_8.shape[0]):
        f, psd = welch(
            data_8[ch], fs=SAMPLE_RATE, nperseg=min(nperseg, len(data_8[ch]))
        )
        for target in STIMULUS_FREQUENCIES:
            for h in range(1, N_HARMONICS + 1):
                idx = np.argmin(np.abs(f - h * target))
                feats.append(psd[idx])
    return np.log1p(np.array(feats))


def extract_features(window_14, refs, nperseg):
    raw_8 = window_14[SSVEP_INDICES, :]
    pp_8 = preprocess_8ch(raw_8)
    fft = fft_features(pp_8, nperseg)
    cca = cca_correlations(pp_8, refs)
    fbcca = fbcca_correlations(raw_8, refs)
    return np.concatenate([fft, cca, fbcca])


#  calibration


class MarkerListener(threading.Thread):
    def __init__(self, port, event_queue):
        super().__init__(daemon=True)
        self.port = port
        self.queue = event_queue
        self.running = True

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", self.port))
        sock.settimeout(0.5)
        while self.running:
            try:
                data, _ = sock.recvfrom(2048)
                msg = json.loads(data.decode())
                self.queue.put(msg)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"marker listener error: {e}")
        sock.close()


def run_calibration(board, subject_name):
    marker_q = queue.Queue()
    listener = MarkerListener(MARKER_PORT, marker_q)
    listener.start()

    print(f"waiting for calibration markers on port {MARKER_PORT}...")
    trials = []
    calibration_active = False
    pending_trial = None

    while True:
        try:
            msg = marker_q.get(timeout=1.0)
        except queue.Empty:
            continue

        event_type = msg.get("event")
        print(f"event: {event_type}  {msg}")

        if event_type == "calibration_start":
            calibration_active = True
            board.get_board_data()

        elif event_type == "trial_start":
            freq = float(msg["frequency"])
            if freq not in STIMULUS_FREQUENCIES:
                pending_trial = None
                continue
            pending_trial = {
                "freq": freq,
                "start_samples": board.get_board_data_count(),
                "duration": msg.get("duration", 5.0),
            }

        elif event_type == "trial_end" and pending_trial is not None:
            time.sleep(0.1)
            end_samples = board.get_board_data_count()
            n_new = end_samples - pending_trial["start_samples"]

            all_data = board.get_current_board_data(n_new + WINDOW_SAMPLES)
            eeg_rows = BoardShim.get_eeg_channels(board.get_board_id())
            raw16 = all_data[eeg_rows, -n_new:]
            aligned14 = remap_to_14(raw16)

            if aligned14.shape[1] < WINDOW_SAMPLES + TRIAL_SKIP_SAMPLES:
                print(
                    f"  warn: trial too short ({aligned14.shape[1]} samples), skipping"
                )
                pending_trial = None
                continue

            aligned14 = aligned14[:, TRIAL_SKIP_SAMPLES:]
            hop = WINDOW_SAMPLES // 2
            n_windows = max(1, (aligned14.shape[1] - WINDOW_SAMPLES) // hop + 1)
            for w in range(n_windows):
                start = w * hop
                win = aligned14[:, start : start + WINDOW_SAMPLES]
                if win.shape[1] == WINDOW_SAMPLES:
                    trials.append((pending_trial["freq"], win))
            print(
                f"  captured {n_windows} windows from {pending_trial['freq']} Hz trial"
            )
            pending_trial = None

        elif event_type == "calibration_done":
            break

    listener.running = False

    if len(trials) == 0:
        raise RuntimeError("no trials captured")

    print(f"\ntotal windows captured: {len(trials)}")
    X_raw = np.array([t[1] for t in trials])
    y = np.array([t[0] for t in trials])
    unique, counts = np.unique(y, return_counts=True)
    print(f"per-class counts: {dict(zip(unique.tolist(), counts.tolist()))}")

    SESSIONS_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    session_path = SESSIONS_DIR / f"dry_{subject_name}_{timestamp}.npz"
    np.savez(
        session_path,
        X=X_raw,
        y=y,
        fs=SAMPLE_RATE,
        subject=subject_name,
        timestamp=timestamp,
        aligned_positions=ALIGNED_14,
    )
    print(f"saved session to {session_path}")
    return X_raw, y


#  training


def load_pretrained():
    if not WET_PRETRAINED_PATH.exists():
        raise FileNotFoundError(f"no pre-trained model at {WET_PRETRAINED_PATH}")
    with open(WET_PRETRAINED_PATH, "rb") as f:
        return pickle.load(f)


def finetune_model(X_raw, y):
    pretrained = load_pretrained()
    if pretrained.get("feature_version") != "fft_cca_fbcca_5class_nakanishi_v1":
        raise ValueError(
            f"pretrained has feature_version {pretrained.get('feature_version')}, "
            f"expected fft_cca_fbcca_5class_nakanishi_v1."
        )

    refs = build_cca_refs(WINDOW_SAMPLES)

    # data augmentation: each real window becomes N_AUG augmented copies
    # cheap variety via amplitude scaling + small gaussian noise
    N_AUG = 3  # each window -> 3 copies (original + 2 augmented)
    AMP_JITTER = 0.10  # +/- 10% amplitude
    NOISE_STD_FRAC = 0.02  # gaussian noise at 2% of signal std

    print(f"extracting features from {len(X_raw)} windows, N_AUG={N_AUG}...")
    feats = []
    labels = []
    rng = np.random.default_rng(42)

    for i, window14 in enumerate(X_raw):
        # always include original unmodified
        feats.append(extract_features(window14, refs, WINDOW_SAMPLES))
        labels.append(y[i])

        # augmented versions
        for _ in range(N_AUG - 1):
            aug = window14.copy().astype(np.float64)
            # per-channel amplitude scaling
            scales = rng.uniform(
                1.0 - AMP_JITTER, 1.0 + AMP_JITTER, size=(aug.shape[0], 1)
            )
            aug = aug * scales
            # gaussian noise proportional to each channel's std
            per_ch_std = aug.std(axis=1, keepdims=True)
            noise = rng.normal(0, NOISE_STD_FRAC, size=aug.shape) * per_ch_std
            aug = aug + noise
            feats.append(extract_features(aug, refs, WINDOW_SAMPLES))
            labels.append(y[i])

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(X_raw)}")

    X_feat = np.array(feats)
    y_aug = np.array(labels)
    print(f"after augmentation: {len(X_feat)} training samples")

    # cast labels to int (x10) to match pretrain convention
    y_int = np.array([int(round(v * 10)) for v in y_aug])

    scaler = pretrained["scaler"]
    X_scaled = scaler.transform(X_feat)
    svm = SVC(
        kernel="rbf", C=1.0, gamma="scale", probability=True, class_weight="balanced"
    )
    svm.fit(X_scaled, y_int)
    print(f"train accuracy: {svm.score(X_scaled, y_int):.3f}")
    print(f"classes (x10): {svm.classes_}")

    model = {
        "scaler": scaler,
        "svm": svm,
        "frequencies": STIMULUS_FREQUENCIES,
        "window_samples": WINDOW_SAMPLES,
        "sample_rate": SAMPLE_RATE,
        "n_feature_channels": N_FEATURE_CHANNELS,
        "feature_version": "fft_cca_fbcca_5class_nakanishi_v1",
    }
    with open(FINETUNED_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"saved fine-tuned model to {FINETUNED_PATH}")
    return model


#  live


def run_live(board, model, husky_pub):
    print("\nlive classification running. ctrl-c to stop.")
    if husky_pub is not None:
        print("Hold space in the dashboard to drive husky.")
    eeg_rows = BoardShim.get_eeg_channels(board.get_board_id())
    scaler = model["scaler"]
    svm = model["svm"]
    refs_live = build_cca_refs(LIVE_WINDOW_SAMPLES)
    classes = svm.classes_

    recent_preds = deque(maxlen=6)
    recent_confs = deque(maxlen=6)

    board.get_board_data()
    last_classify = time.time()

    try:
        while True:
            now = time.time()
            if now - last_classify < STEP_SEC:
                time.sleep(0.01)
                continue

            if board.get_board_data_count() < LIVE_WINDOW_SAMPLES:
                time.sleep(0.05)
                continue

            data = board.get_current_board_data(LIVE_WINDOW_SAMPLES)
            raw16 = data[eeg_rows, :]
            if raw16.shape[1] < LIVE_WINDOW_SAMPLES:
                continue

            aligned14 = remap_to_14(raw16)
            feat = extract_features(aligned14, refs_live, LIVE_WINDOW_SAMPLES).reshape(
                1, -1
            )
            feat_scaled = scaler.transform(feat)
            probs = svm.predict_proba(feat_scaled)[0]
            pred_idx = np.argmax(probs)
            # svm classes are ints (freq * 10) — convert back to float Hz
            pred_freq = float(classes[pred_idx]) / 10.0
            conf = probs[pred_idx]

            if conf < PER_FREQ_THRESHOLD.get(pred_freq, DEFAULT_THRESHOLD):
                last_classify = now
                continue

            recent_preds.append(pred_freq)
            recent_confs.append(conf)

            if len(recent_preds) >= 4:
                counts = {f: 0 for f in STIMULUS_FREQUENCIES}
                for p in recent_preds:
                    counts[p] = counts.get(p, 0) + 1
                smoothed = max(counts, key=counts.get)
                agreeing = [
                    c for p, c in zip(recent_preds, recent_confs) if p == smoothed
                ]
                smoothed_conf = float(np.mean(agreeing))
            else:
                smoothed, smoothed_conf = pred_freq, conf

            send_prediction(smoothed, smoothed_conf)

            if husky_pub is not None and smoothed in FREQ_TO_TWIST:
                lin, ang = FREQ_TO_TWIST[smoothed]
                husky_pub.update_command(lin, ang)

            last_classify = now

    except KeyboardInterrupt:
        print("\nstopping live mode")


def setup_board():
    params = BrainFlowInputParams()
    params.serial_port = "/dev/ttyUSB0"
    board_id = BoardIds.CYTON_DAISY_BOARD.value
    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    time.sleep(2.0)
    print("board streaming")
    return board, board_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="run calibration only, save session, exit",
    )
    parser.add_argument(
        "--skip-calib",
        action="store_true",
        help="skip calibration, load existing finetuned model, go straight to live",
    )
    parser.add_argument("--subject", type=str, default=None)
    parser.add_argument("--no-ros", action="store_true")
    args = parser.parse_args()

    if args.collect_only and args.skip_calib:
        print("error: --collect-only and --skip-calib are mutually exclusive")
        return

    subject = args.subject
    if subject is None and not args.skip_calib:
        subject = input("subject name (no spaces): ").strip().lower() or "unknown"

    husky_pub = None
    deadman_listener = None
    if not args.no_ros and not args.collect_only:
        try:
            husky_pub = HuskyPublisher()
            deadman_listener = DeadmanListener(husky_pub)
            deadman_listener.start()
        except Exception as e:
            print(f"[ROS] failed to init: {e}")
            print(f"[ROS] did you source /opt/ros/jazzy/setup.bash?")
            return

    board, board_id = setup_board()
    print(f"4-class {STIMULUS_FREQUENCIES}, fbcca, subject={subject}")
    if husky_pub is not None:
        print(f"[ROS] mapping: 12=FWD, 10=BACK, 6=LEFT, 7.5=RIGHT")

    try:
        if args.skip_calib:
            # load existing finetuned model, skip calibration
            if not FINETUNED_PATH.exists():
                raise FileNotFoundError(
                    f"--skip-calib needs {FINETUNED_PATH} but it doesn't exist. "
                    f"run calibration first."
                )
            print(f"\n[skip-calib] loading existing model from {FINETUNED_PATH}")
            with open(FINETUNED_PATH, "rb") as f:
                model = pickle.load(f)
            if model.get("feature_version") != "fft_cca_fbcca_5class_nakanishi_v1":
                raise ValueError(
                    f"finetuned model has feature_version "
                    f"{model.get('feature_version')}, expected "
                    f"fft_cca_fbcca_5class_nakanishi_v1. rerun calibration."
                )
            print(f"[skip-calib] classes (x10): {model['svm'].classes_}")
            send_status("ready", "skipping calibration, entering live mode")
            time.sleep(1.0)
            run_live(board, model, husky_pub)
            return

        X_raw, y = run_calibration(board, subject)

        if args.collect_only:
            print("\ncollect-only: session saved, exiting.")
            return

        send_status("training", "fine-tuning")
        model = finetune_model(X_raw, y)
        send_status("ready", "entering live mode")
        time.sleep(1.0)
        run_live(board, model, husky_pub)

    finally:
        board.stop_stream()
        board.release_session()
        print("board released")
        if deadman_listener is not None:
            deadman_listener.running = False
        if husky_pub is not None:
            husky_pub.shutdown()


if __name__ == "__main__":
    main()
