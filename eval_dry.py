"""
Hybrid FFT + CCA + FBCCA SSVEP classifier using RBF SVM.

Features:
- FFT band power (per channel, harmonics)
- Standard CCA correlation per frequency
- FBCCA subband correlations

Evaluated with stratified k-fold cross-validation.
"""

import sys
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import full_bci_pipeline as bci
from full_bci_pipeline import (
    SSVEP_INDICES,
    SSVEP_CHANNEL_NAMES,
    CAL_WINDOW_SAMPLES,
    SAMPLE_RATE,
    BANDPASS_LOW,
    BANDPASS_HIGH,
    FBCCA_SUBBANDS,
    preprocess_occipital,
    subband_filter,
    build_cca_refs,
    cca_corr,
)
from eval_openbci_gui import (
    load_dataset,
    BLOCK_TO_FREQ,
    FREQ_ORDER_4CLASS,
    FREQ_ORDER_5CLASS,
)

N_HARMONICS_FFT = 2


def fft_features(data_8, nperseg, freqs):
    #FFT power at stimulus frequencies and harmonics (log-scaled).
    from scipy.signal import welch

    feats = []
    for ch in range(data_8.shape[0]):
        f, psd = welch(
            data_8[ch], fs=SAMPLE_RATE, nperseg=min(nperseg, len(data_8[ch]))
        )
        for target in freqs:
            for h in range(1, N_HARMONICS_FFT + 1):
                idx = np.argmin(np.abs(f - h * target))
                feats.append(psd[idx])
    return np.log1p(np.array(feats))


def standard_cca_correlations(data_pp, freqs, refs):
    # CCA correlation between EEG and sinusoidal reference signals
    out = []
    X = data_pp.T
    for f in freqs:
        out.append(cca_corr(X, refs[f]))
    return np.array(out)


def fbcca_correlations(data_raw, freqs, n_samples):
    """
    FBCCA correlations across subbands.
    """
    refs = build_cca_refs(n_samples)
    out = []
    for sb_low, sb_high in FBCCA_SUBBANDS:
        filtered = subband_filter(data_raw, sb_low, min(sb_high, SAMPLE_RATE / 2 - 1))
        for f in freqs:
            out.append(cca_corr(filtered.T, refs[f]))
    return np.array(out)


def extract_features(window_14, n_samples, freqs):
    # Construct full feature vector for a single EEG window.
    raw_ssvep = window_14[SSVEP_INDICES, :n_samples]
    pp_ssvep = preprocess_occipital(raw_ssvep)

    fft = fft_features(pp_ssvep, n_samples, freqs)
    cca = standard_cca_correlations(pp_ssvep, freqs, build_cca_refs(n_samples))
    fbcca = fbcca_correlations(raw_ssvep, freqs, n_samples)

    return np.concatenate([fft, cca, fbcca])


def build_feature_matrix(X_windows, freqs, verbose=True):
    # Extract features for all windows.
    feats = []
    for i, w in enumerate(X_windows):
        feats.append(extract_features(w, CAL_WINDOW_SAMPLES, freqs))
        if verbose and (i + 1) % 50 == 0:
            print(f"    features {i+1}/{len(X_windows)}")
    return np.array(feats)


# evaluation


def stratified_kfold_eval(X_feat, y, n_splits=5, freqs=None, verbose=True):
    """
    Stratified CV evaluation using RBF SVM.

    Labels are scaled to integers because sklearn SVC does not handle floats
    cleanly for class identity tasks.
    """
    y_int = np.round(np.asarray(y) * 10).astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    all_pred = np.empty_like(y_int)
    all_true = np.empty_like(y_int)
    fill = 0

    fold_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_feat, y_int)):
        X_train, X_test = X_feat[train_idx], X_feat[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        svm = SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )
        svm.fit(X_train_s, y_train)
        y_pred = svm.predict(X_test_s)

        fold_accuracies.append((y_pred == y_test).mean())

        n_test = len(y_test)
        all_pred[fill : fill + n_test] = y_pred
        all_true[fill : fill + n_test] = y_test
        fill += n_test

    all_pred_f = all_pred / 10.0
    all_true_f = all_true / 10.0

    accuracy = (all_pred_f == all_true_f).mean()

    if freqs is None:
        freqs = sorted(set(all_true_f.tolist()))

    idx = {f: i for i, f in enumerate(freqs)}
    mat = np.zeros((len(freqs), len(freqs)), dtype=int)

    for t, p in zip(all_true_f, all_pred_f):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1

    per_class = {}
    for i, f in enumerate(freqs):
        true_n = mat[i].sum()
        pred_n = mat[:, i].sum()
        correct = mat[i, i]

        per_class[f] = {
            "recall": correct / true_n if true_n else 0,
            "precision": correct / pred_n if pred_n else 0,
            "true_n": int(true_n),
            "predicted_n": int(pred_n),
        }

    return accuracy, mat, per_class, fold_accuracies
