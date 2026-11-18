"""
caisr_arousal.py
----------------
Arousal-event detection module for CAISR (Complete AI Sleep Report).

Workflow
--------
1. Discover .h5 PSG files in the input directory.
2. Pre-process each recording (notch filter → bandpass → resample → temporal scaling).
3. Run the trained arousal model and apply post-processing.
4. Write per-subject arousal CSV files to the output directory.

Processing can run sequentially or in parallel depending on the
``multiprocess`` flag in the run-parameter CSV (see ``extract_run_parameters``).

Usage
-----
    python caisr_arousal.py \
        --input_data_dir  /path/to/prepared_h5_files/ \
        --output_csv_dir  /path/to/output_csvs/ \
        --param_dir       /path/to/run_params/
"""

import os
import sys
import time
import logging
import warnings
import argparse
import multiprocessing
import tempfile
import shutil
from math import gcd

import h5py as h5
import mne
import numpy as np
import pandas as pd
from glob import glob
from typing import List, Tuple

from scipy.stats import iqr
from scipy.signal import resample, resample_poly
from sklearn.preprocessing import RobustScaler

# EDF loader (only required when an .edf input is encountered).
try:
    from edf_loader import load_edf_for_caisr as _load_edf_for_caisr
except ImportError:
    _load_edf_for_caisr = None

# TS-Zarr loader (BDSP primary stores); needs `pip install bdsp`.
from zarr_loader import is_zarr_input, load_raw_for_caisr


def _load_arousal_edf_as_sig_group(file: str, target_fs: int = 200) -> dict:
    """Read an .edf file and return a {channel_name: 1D ndarray} dict at *target_fs*.

    Drop-in for the dict produced by the H5 loader paths in
    ``pre_process_temporal_scaling`` — same shape, same units (Volts).
    Each EDF channel is independently resampled to *target_fs* via polyphase
    filtering. Channel renaming / bipolar derivations are handled by
    ``edf_loader.load_edf_for_caisr`` using ``channel_table.csv``.
    """
    if _load_edf_for_caisr is None and not is_zarr_input(file):
        raise ImportError(
            "EDF input detected but `edfio` is not installed. "
            "Run `pip install edfio` inside the caisr_arousal environment."
        )
    channel_data, channel_fs = load_raw_for_caisr(file, _load_edf_for_caisr)
    sig_group: dict = {}
    for ch, sig in channel_data.items():
        sig = np.asarray(sig, dtype=np.float64)
        src_fs = float(channel_fs[ch])
        if src_fs != float(target_fs):
            g = gcd(int(target_fs), int(src_fs))
            sig = resample_poly(sig, int(target_fs) // g, int(src_fs) // g)
        sig_group[ch] = sig
    return sig_group

# Suppress noisy framework logs before any TF/MNE imports
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", category=UserWarning)

import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
mne.set_log_level(verbose="CRITICAL")

from mne.preprocessing import create_ecg_epochs, EOGRegression
from os.path import join as opj

# Project-internal utilities
from arousal.utils.models.model_init import *
from arousal.utils.load_write.ids2label import *
from arousal.utils.pre_processing.scaling import *
from arousal.utils.post_processing.label_class import *
from arousal.utils.hyperparameters.hyperparameters import *
from arousal.utils.pre_processing.quality_control_funcs import *
from arousal.utils.post_processing.smoothing_arousal import movav
from arousal.utils.post_processing.rem_post_processing import rem_post_processing
from arousal.utils.post_processing.smoothing_arousal import post_process_after_smoothing

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------
# Root directory of the arousal sub-package (adjust if your layout differs).
AROUSAL_PKG_DIR = "arousal"

# YAML file with model hyper-parameters.
HPARAMS_PATH = os.path.join(AROUSAL_PKG_DIR, "utils", "models", "FINAL_MODELS", "hparams.yaml")

# CSV that maps subject IDs to the cross-validation model fold they belong to.
MODEL_SPLIT_CSV = os.path.join(AROUSAL_PKG_DIR, "utils", "splits", "model_path_for_split.csv")

# Default (general-population) trained model weights directory.
DEFAULT_MODEL_DIR = os.path.join(AROUSAL_PKG_DIR, "utils", "models", "FINAL_MODELS", "Model_CAISR_AROUSAL")

# Number of parallel worker processes used when multiprocessing is enabled.
MAX_WORKERS = 8


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def timer(tag: str) -> None:
    """Print *tag* with a brief animated progress indicator."""
    print(tag)
    for i in range(1, len(tag) + 1):
        print("." * i + "     ", end="\r")
        time.sleep(1.5 / len(tag))
    print()


def extract_run_parameters(param_csv: str) -> Tuple[bool, bool]:
    """
    Read run-time flags from a parameter CSV file.

    Expected columns
    ----------------
    overwrite : bool-like
        If True, re-process files that already have output CSVs.
    multiprocess : bool-like  (optional, defaults to False)
        If True, process files in parallel using ``multiprocessing.Pool``.

    Returns
    -------
    overwrite : bool
    multiprocess : bool
    """
    assert os.path.exists(param_csv), f"Run parameter file not found: {param_csv}"
    params = pd.read_csv(param_csv)
    overwrite = bool(params["overwrite"].values[0])
    multiprocess_val = params.get("multiprocess", pd.Series([False])).values[0]
    multiprocess = str(multiprocess_val).strip().lower() == "true"
    return overwrite, multiprocess


# ---------------------------------------------------------------------------
# I/O path management
# ---------------------------------------------------------------------------

def set_output_paths(
    input_paths: List[str], csv_folder: str, overwrite: bool
) -> Tuple[List[str], List[str]]:
    """
    Derive output CSV paths from input .h5 paths and optionally skip
    files that have already been processed.

    Parameters
    ----------
    input_paths : list of str
    csv_folder  : str  Root folder for CSV outputs; arousal CSVs go in
                       ``<csv_folder>/arousal/``.
    overwrite   : bool  If False, already-existing CSVs are skipped.

    Returns
    -------
    filtered input paths, filtered csv paths
    """
    ids = [p.split("/")[-1].split(".")[0] for p in input_paths]
    csv_paths = [os.path.join(csv_folder, "arousal", f"{id_}_arousal.csv") for id_ in ids]
    os.makedirs(os.path.join(csv_folder, "arousal"), exist_ok=True)
    assert len(input_paths) == len(csv_paths), (
        "SETUP ERROR: mismatch between input file count and CSV path count."
    )
    return filter_already_processed_files(input_paths, csv_paths, overwrite)


def filter_already_processed_files(
    input_paths: List[str], csv_paths: List[str], overwrite: bool
) -> Tuple[List[str], List[str]]:
    """Remove files whose output CSVs already exist (unless *overwrite* is True)."""
    total = len(input_paths)
    if not overwrite:
        todo_indices = [i for i, p in enumerate(csv_paths) if not os.path.exists(p)]
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        processed_count = total - len(todo_indices)
    else:
        processed_count = 0

    tag = "(overwrite)" if overwrite else ""
    print(
        f">> {processed_count}/{total} files already processed\n"
        f">> {len(input_paths)} to go.. {tag}\n"
    )
    return input_paths, csv_paths


def filter_todo_files(
    input_paths: List[str],
    csv_paths: List[str],
    keep_indices: List[int],
) -> Tuple[List[str], List[str]]:
    """
    Subset input/CSV path lists by *keep_indices*.

    Parameters
    ----------
    input_paths, csv_paths : list of str
    keep_indices : list of int  Indices of files to retain.

    Returns
    -------
    Filtered input paths, filtered csv paths.
    """
    input_paths = np.array(input_paths)[keep_indices].tolist()
    csv_paths = np.array(csv_paths)[keep_indices].tolist()
    return input_paths, csv_paths


# ---------------------------------------------------------------------------
# Signal clipping helpers
# ---------------------------------------------------------------------------

def clip_chin(image: np.ndarray, filtered: bool = False) -> np.ndarray:
    """
    Remove high-amplitude artefacts from a chin-EMG channel.

    Steps
    -----
    1. (If *filtered* is False) subtract a 50-sample rolling-mean baseline.
    2. Clip samples that exceed ±1.5× the local absolute-difference envelope.
    3. Clip samples that exceed ±3× the long-range absolute-amplitude envelope.

    Parameters
    ----------
    image    : 1-D array  Raw chin signal (200 Hz assumed).
    filtered : bool       Set True if the signal has already been baseline-corrected.

    Returns
    -------
    Clipped signal (same length as *image*).
    """
    rolling_abs_diff = np.zeros(len(image))
    rolling_abs_diff_ref = np.zeros(len(image))

    if not filtered:
        drift = np.squeeze(
            pd.DataFrame({"x": image}).rolling(50, center=True, min_periods=0).mean()
        )
        image = image - drift

    # Local and long-range smoothed absolute differences
    rolling_abs_diff[:-1] = np.squeeze(
        pd.DataFrame({"x": np.abs(np.diff(image))})
        .rolling(50, center=True, min_periods=0)
        .mean()
    )
    rolling_abs_diff_ref[:-1] = np.squeeze(
        pd.DataFrame({"x": np.abs(np.diff(image))})
        .rolling(1000, center=True, min_periods=0)
        .mean()
    )

    thres_max = rolling_abs_diff_ref + rolling_abs_diff * 1.5
    thres_min = -rolling_abs_diff_ref - rolling_abs_diff * 1.5
    image = np.clip(image, thres_min, thres_max)

    if not filtered:
        image = image + drift

    # Secondary clip: long-range amplitude envelope
    rolling_long = (
        np.squeeze(
            pd.DataFrame({"x": np.abs(image)})
            .rolling(200_000, center=True, min_periods=0)
            .mean()
        )
        * 3
    )
    rolling_std = np.squeeze(
        pd.DataFrame({"x": np.abs(image)})
        .rolling(10_000, center=True, min_periods=0)
        .std()
    )
    image[rolling_std > rolling_long] = np.minimum(
        rolling_std[rolling_std > rolling_long],
        image[rolling_std > rolling_long],
    )
    image[-rolling_std < -rolling_long] = np.maximum(
        -rolling_std[-rolling_std < -rolling_long],
        image[-rolling_std < -rolling_long],
    )
    return image


# ---------------------------------------------------------------------------
# Chin-EMG ECG artefact removal
# ---------------------------------------------------------------------------

def chin_unscaled(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove ECG artefacts from a chin-EMG signal via MNE EOG regression.

    Parameters
    ----------
    data : ndarray, shape (2, n_samples)
        Row 0 — raw chin EMG (200 Hz).
        Row 1 — ECG (200 Hz), used as the regression reference.

    Returns
    -------
    emg_filtered      : 1-D array  Bandpass-filtered (10–45 Hz) chin signal,
                                   threshold-clipped; intended for REM post-processing.
    emg_model_input   : 1-D array  Broadband cleaned chin signal for model input.
    """
    sfreq = 128  # Signals are expected at 128 Hz at this stage

    # Pad a zero reference channel so MNE's EEG reference machinery works
    data_extra = np.zeros((data.shape[0] + 1, data.shape[1]))
    data_extra[: data.shape[0], :] = data.copy()

    info = mne.create_info(
        ch_names=["M2", "ECG", "M1"],
        sfreq=sfreq,
        ch_types=["eeg", "ecg", "eeg"],
    )
    raw = mne.io.RawArray(data_extra, info, first_samp=0, verbose=None)
    raw.set_eeg_reference(ref_channels=["M1"])
    raw.set_montage(mne.channels.make_standard_montage("standard_1020"))

    # Regress out the average ECG evoked response from EEG channels
    ecg_evoked = create_ecg_epochs(raw).average("all")
    model_evoked = EOGRegression(picks="eeg", picks_artifact="ecg").fit(ecg_evoked)
    raw_clean = model_evoked.apply(raw)
    data_clean = raw_clean.get_data()

    # --- Broadband chin (for model input) ---
    emg_model_input = data_clean[-3, :].copy()
    emg_mov_mean = (
        pd.DataFrame({"x": emg_model_input})
        .rolling(sfreq, min_periods=1, center=True)
        .mean()
        .values
    )
    emg_mov_std = (
        pd.DataFrame({"x": emg_model_input})
        .rolling(sfreq, min_periods=1, center=True)
        .std()
        .values
        * 2.5
    )
    emg_mov_mean = np.squeeze(np.nan_to_num(emg_mov_mean, nan=1.0))
    emg_mov_std = np.squeeze(np.nan_to_num(emg_mov_std, nan=1.0))

    upper = emg_mov_mean + emg_mov_std
    lower = emg_mov_mean - emg_mov_std
    emg_model_input = np.clip(emg_model_input, lower, upper)
    global_std = pd.DataFrame({"x": emg_model_input}).std().values * 20
    emg_model_input = np.clip(emg_model_input, -global_std, global_std)

    # --- Bandpass chin (for REM post-processing) ---
    raw_clean_for_filter = mne.io.RawArray(data_clean, info, first_samp=0, verbose=None)
    raw_clean_for_filter.set_eeg_reference(ref_channels=["M1"])
    raw_clean_for_filter.filter(l_freq=10, h_freq=45)
    emg_filtered = raw_clean_for_filter.get_data()[-3, :]

    emg_mov = np.squeeze(
        np.nan_to_num(
            pd.DataFrame({"x": emg_filtered})
            .rolling(sfreq, min_periods=1, center=True)
            .std()
            .values
            * 2.5,
            nan=1.0,
        )
    )
    emg_filtered = np.clip(emg_filtered, -emg_mov, emg_mov)
    global_std_filtered = pd.DataFrame({"x": emg_filtered}).std().values * 20
    emg_filtered = np.clip(emg_filtered, -global_std_filtered, global_std_filtered)

    return emg_filtered, emg_model_input


# ---------------------------------------------------------------------------
# Pre-processing pipeline
# ---------------------------------------------------------------------------

def do_initial_preprocessing(
    signals: pd.DataFrame, new_Fs: int, chin_filter: bool = False
) -> pd.DataFrame:
    """
    Apply notch filter → bandpass filter → resample to all PSG channels.

    The pipeline assumes the signals arrive at 200 Hz.  Resampling to
    *new_Fs* is applied where appropriate; for channels not listed, a
    simple factor-of-2 decimation is used.

    Parameters
    ----------
    signals     : DataFrame  One column per channel (lower-case names).
    new_Fs      : int        Target sample rate (Hz).
    chin_filter : bool       If True, chin data are treated as already
                             baseline-corrected before clipping.

    Returns
    -------
    DataFrame with the same columns, filtered and resampled.
    """
    from mne.filter import filter_data, notch_filter
    from scipy.signal import resample_poly

    NOTCH_US = 60.0          # North-American power-line frequency (Hz)
    BP_EEG = [0.2, 35]      # EEG / EOG / chin bandpass (Hz)
    BP_RESP = [0.0, 10]     # Respiratory channel bandpass (Hz)
    BP_ECG = [0.2, 35]      # ECG bandpass (Hz)
    ORIG_FS = 200            # Expected input sample rate (Hz)

    EEG_CHANS = {"f3-m2", "f4-m1", "c3-m2", "c4-m1", "cz-oz", "o1-m2", "o2-m1", "e1-m2"}
    EOG_CHANS = {"e1-m2"}
    CHIN_CHANS = {"chin1-chin2", "chin"}
    RESP_CHANS = {"abd", "chest", "airflow", "ptaf", "cflow"}
    ALL_NOTCH = EEG_CHANS | CHIN_CHANS | RESP_CHANS | {"ecg"}
    ALL_RESAMPLE = EEG_CHANS | CHIN_CHANS | RESP_CHANS | {"ecg"}

    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        image = signals[sig].values

        # 1. Notch filter (60 Hz)
        if sig in ALL_NOTCH:
            image = notch_filter(image, ORIG_FS, NOTCH_US, verbose=False)

        # 2. Bandpass filter
        if sig in EEG_CHANS:
            image = filter_data(image, ORIG_FS, *BP_EEG, verbose=False)
        elif sig in CHIN_CHANS:
            image = clip_chin(image, chin_filter)
            image = filter_data(np.asarray(image), ORIG_FS, *BP_EEG, verbose=False)
        elif sig in RESP_CHANS:
            image = filter_data(image, ORIG_FS, *BP_RESP, verbose=False)
        elif sig == "ecg":
            image = filter_data(image, ORIG_FS, *BP_ECG, verbose=False)

        # 3. Resample
        if new_Fs != ORIG_FS:
            if sig in ALL_RESAMPLE:
                image = resample_poly(image, new_Fs, ORIG_FS)
            else:
                image = image[::2]  # Simple decimation for other channels

        new_df.loc[:, sig] = image

    del signals
    return new_df


# ---------------------------------------------------------------------------
# Temporal amplitude scaling
# ---------------------------------------------------------------------------

def apply_pre_process(
    data: np.ndarray, channel_type: str, time_min: int = 10,
    use_robust_scaler: bool = False,
) -> np.ndarray:
    """
    Normalise a single-channel signal.

    Two modes are supported:

    *use_robust_scaler=False* (default, raw EDF/HSP data in Volts):
        Rolling median / IQR scaling with channel-specific physiological
        IQR bounds.  The signal is divided into non-overlapping *time_min*-
        minute windows; per-window IQR and median are smoothed and used to
        z-score the signal.

    *use_robust_scaler=True* (pre-normalised qli/JMLR data, IQR ≈ 1):
        Scale-agnostic normalisation: clip outliers at ±20 × global IQR,
        then apply sklearn RobustScaler (centre by median, scale by IQR).
        Matches the preprocessing used by the staging pipeline.

    Parameters
    ----------
    data              : 1-D array at 128 Hz.
    channel_type      : str   One of 'EEG', 'ECG', 'CHIN', 'EYE'.
    time_min          : int   Window length in minutes (default 10).
    use_robust_scaler : bool  True for pre-normalised input (default False).

    Returns
    -------
    Scaled 1-D array of the same length as *data*.
    """
    # Clip outliers before any normalisation path
    eeg_2row = np.vstack([data, data]).T
    clipped, _ = clip_noisy_values(eeg_2row, 128, len(eeg_2row) / 128, min_max_times_global_iqr=20)
    signal = clipped[:, 0].copy()

    if use_robust_scaler:
        sig = np.atleast_2d(signal).T
        return np.squeeze(RobustScaler().fit_transform(sig))

    # --- Rolling IQR/median scaling (raw V-scale data) ---
    # IQR physiological bounds (empirically determined per channel type)
    IQR_BOUNDS = {
        "EEG":  (8.30e-6, 3.50e-5),
        "ECG":  (2.40e-5, 2.00e-4),
        "CHIN": (7.20e-7, 2.00e-5),
        "EYE":  (8.40e-6, 3.70e-5),
    }

    win = time_min * 60 * 128  # window length in samples
    signal_cut = signal[: len(signal) // win * win].reshape(-1, win)

    median_vec = np.median(signal_cut, axis=1)
    iqr_vec = iqr(signal_cut, axis=1)

    # Clamp IQR to physiological bounds
    if channel_type in IQR_BOUNDS:
        lo, hi = IQR_BOUNDS[channel_type]
        iqr_vec = np.clip(iqr_vec, lo, hi)

    # Expand window-level statistics to sample-level
    iqr_trace = np.full(len(signal), iqr(signal[len(signal_cut.ravel()) :]))
    iqr_trace[: len(iqr_vec) * win] = iqr_vec.repeat(win)

    med_trace = np.full(len(signal), np.median(signal))
    med_trace[: len(median_vec) * win] = median_vec.repeat(win)

    # Smooth statistics over 4-window span
    iqr_smooth = movav(iqr_trace, win * 4)
    med_smooth = movav(med_trace, win * 4)

    return (signal - med_smooth) / iqr_smooth


# ---------------------------------------------------------------------------
# H5 I/O helpers
# ---------------------------------------------------------------------------

def load_h5_signals(path: str) -> dict:
    """
    Load all datasets from an HDF5 file into a plain dictionary.

    The ``channel_names`` dataset (if present) is decoded to a Python
    list of strings.

    Parameters
    ----------
    path : str  Path to the .h5 file.

    Returns
    -------
    dict  {key: numpy array}, with channel_names as list[str] if present.
    """
    import h5py

    signals = {}
    with h5py.File(path, "r") as f:
        for key in f.keys():
            signals[key] = f[key][:]
    if "channel_names" in signals:
        signals["channel_names"] = list(signals["channel_names"].astype(str))
    return signals


# ---------------------------------------------------------------------------
# Full pre-processing pipeline (one subject)
# ---------------------------------------------------------------------------

def pre_process_temporal_scaling(
    file: str,
    path_write: str,
    channels: List[str],
    channel_type: List[str],
    temporal_resolution: int = 5,
    chin_filter: bool = False,
) -> str:
    """
    Pre-process a raw PSG HDF5 file and write a scaled intermediate HDF5.

    Steps
    -----
    1. Load raw signals from *file*.
    2. If a CHIN_RAW channel is present, remove ECG artefacts via
       ``chin_unscaled`` and replace the CHIN channel with the cleaned signal.
    3. Notch-filter, bandpass-filter, and resample to 128 Hz.
    4. Resample to 128 Hz with ``scipy.signal.resample``.
    5. Apply temporal IQR/median scaling per channel.
    6. Write the result to *path_write* (gzip-compressed HDF5).

    Parameters
    ----------
    file               : str   Path to raw input .h5 file.
    path_write         : str   Destination path for the processed .h5 file.
    channels           : list  Channel names matching H5 dataset keys.
    channel_type       : list  Semantic type for each channel (same order).
    temporal_resolution: int   Window size (minutes) for IQR scaling.
    chin_filter        : bool  See ``clip_chin``.

    Returns
    -------
    path_write : str  Path to the written file (for chaining).
    """
    # Fallback chain for each EEG channel: try these in order when the primary
    # channel and its contralateral are both absent (e.g. MESA has cz-oz only).
    _EEG_FALLBACK = {
        'c3-m2': ['c4-m1', 'cz-oz', 'f3-m2', 'f4-m1', 'o1-m2', 'o2-m1'],
        'c4-m1': ['c3-m2', 'cz-oz', 'f4-m1', 'f3-m2', 'o2-m1', 'o1-m2'],
    }
    # Fallback for non-EEG channels (e.g. SUTS has "chin" instead of "chin1-chin2")
    _NON_EEG_FALLBACK = {
        'chin1-chin2': ['chin'],
    }

    # BITS layout: single 2D top-level `Xy` dataset; rows match BITS_XY_COLS.
    _BITS_XY_COLS = [
        'f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1',
        'e1-m2', 'e2-m1', 'chin1-chin2',
        'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'spo2', 'ecg',
        'lat', 'rat', 'cpres', 'cpap_on',
        'stage_majority', 'arousal_majority', 'resp_majority', 'limb_majority',
        'stage_0', 'arousal_0', 'resp_0', 'limb_0',
        'stage_1', 'arousal_1', 'resp_1', 'limb_1',
        'stage_2', 'arousal_2', 'resp_2', 'limb_2',
    ]

    # --- Load raw signals (H5 or EDF, auto-detected) ---
    _unit_attr = None
    if file.lower().endswith('.edf') or is_zarr_input(file):
        sig_group = _load_arousal_edf_as_sig_group(file, target_fs=200)
    else:
        with h5.File(file, "r") as f:
            # Support BITS (single 2D `Xy`), nested (f["signals"]/channel), and flat (f/channel) layouts.
            _top = list(f.keys())
            if len(_top) == 1 and _top[0] == 'Xy' and isinstance(f['Xy'], h5.Dataset) and f['Xy'].ndim == 2:
                _xy = f['Xy'][:]
                if _xy.shape[0] != len(_BITS_XY_COLS):
                    raise ValueError(
                        f"BITS Xy has {_xy.shape[0]} rows; expected {len(_BITS_XY_COLS)} "
                        f"matching _BITS_XY_COLS."
                    )
                sig_group = {name: _xy[i, :] for i, name in enumerate(_BITS_XY_COLS)}
            else:
                # Materialise to a dict so the h5 file handle can close cleanly.
                _src = f["signals"] if "signals" in f else f
                sig_group = {k: np.array(_src[k]) for k in _src.keys()}
            # Capture voltage-unit hint (e.g. f.attrs['unit_voltage'] = 'V'|'uV').
            _ua = f.attrs.get("unit_voltage", None)
            if _ua is not None:
                try:
                    _ua = _ua.decode() if isinstance(_ua, bytes) else _ua
                    _unit_attr = str(_ua).strip().lower().replace("μ", "u")
                except Exception:
                    _unit_attr = None

    available = set(sig_group.keys())
    missing_idx = [i for i, ch in enumerate(channels) if ch not in available]
    substitute_map: dict = {}  # missing ch -> available substitute ch
    if missing_idx:
        # Try non-EEG fallbacks first
        non_eeg_missing = []
        for i in missing_idx:
            if channel_type[i] == "EEG":
                continue
            ch = channels[i]
            sub = next((s for s in _NON_EEG_FALLBACK.get(ch, []) if s in available), None)
            if sub:
                substitute_map[ch] = sub
            else:
                non_eeg_missing.append(ch)
        if non_eeg_missing:
            raise ValueError(
                f"Required non-EEG channel(s) missing: {non_eeg_missing}. "
                f"Available: {sorted(available)}"
            )
        eeg_missing = [channels[i] for i in missing_idx if channel_type[i] == "EEG"]
        substituted, no_substitute = [], []
        for ch in eeg_missing:
            sub = next((s for s in _EEG_FALLBACK.get(ch, []) if s in available), None)
            if sub:
                substitute_map[ch] = sub
                substituted.append(f"{ch} -> {sub}")
            else:
                no_substitute.append(ch)
        if substitute_map:
            warnings.warn(f"Channel(s) missing, using substitute: {list(substitute_map.items())}.")
        if no_substitute:
            warnings.warn(f"EEG channel(s) {no_substitute} not in file and no substitute — skipping.")
        keep_idx = [i for i, ch in enumerate(channels)
                    if ch in available or ch in substitute_map]
        channels = [channels[i] for i in keep_idx]
        channel_type = [channel_type[i] for i in keep_idx]
    if not any(ct == "EEG" for ct in channel_type):
        raise ValueError(
            f"No EEG channels available. Available signals: {sorted(available)}"
        )
    _first_src = substitute_map.get(channels[0], channels[0])
    n_samples = len(sig_group[_first_src])
    data = np.zeros((len(channels), n_samples))
    for i, ch in enumerate(channels):
        src = substitute_map.get(ch, ch)
        data[i, :] = np.squeeze(np.array(sig_group[src]))

    # --- Auto-detect input voltage scale (V vs uV) and convert to V ---
    # Trust f.attrs['unit_voltage'] if present; otherwise use amplitude.
    # qli-prepared data is clipped to ~±20 dimensionless, raw V EEG is
    # ~±1e-3 V, raw uV EEG ranges up to ~±300 uV — so a 99th-percentile
    # threshold of 30 separates uV from {qli, V} cleanly.
    VOLTAGE_TYPES = {"EEG", "EYE", "CHIN", "ECG", "CHIN_RAW"}
    voltage_idxs = [i for i, ct in enumerate(channel_type) if ct in VOLTAGE_TYPES]
    input_in_uV = False
    if _unit_attr in ("uv", "microvolt", "microvolts"):
        input_in_uV = True
    elif _unit_attr not in ("v", "volt", "volts") and voltage_idxs:
        sample_p99 = float(np.nanpercentile(np.abs(data[voltage_idxs[0], :]), 99))
        input_in_uV = sample_p99 > 30
    if input_in_uV:
        for i in voltage_idxs:
            data[i, :] = data[i, :] / 1e6

    # --- Detect pre-normalised (qli) vs raw (EDF/HSP) data ---
    # JMLR qli-prepared H5 files are clipped and RobustScaler-normalised to
    # IQR ≈ 1, range ≈ [-20, 20].  Raw EDF/HSP signals arrive in Volts with
    # EEG IQR ~ 8–35 µV (8e-6 – 3.5e-5).  A threshold of 0.1 reliably
    # separates the two: pre-normalised data → IQR ≈ 1, raw data → IQR ≪ 0.1.
    EEG_EOG_TYPES = {"EEG", "EYE"}
    eeg_idxs = [i for i, ct in enumerate(channel_type) if ct in EEG_EOG_TYPES]
    is_prenormalized = False
    if eeg_idxs:
        sample_iqr = float(np.median([iqr(data[i, :]) for i in eeg_idxs]))
        is_prenormalized = sample_iqr > 0.1  # IQR ≈ 1 → qli; IQR ≈ 8–35 µV → raw

    # --- ECG artefact removal for chin channel (if raw chin is available) ---
    if "CHIN_RAW" in channel_type:
        idx_ecg = channel_type.index("ECG")
        ecg_unprocessed = data[idx_ecg, :].copy()

        idx_chin_col = [i for i, c in enumerate(channels) if "chin" in c][-1]
        channels[idx_chin_col] = "chin_raw"

        idx_chin_raw = channel_type.index("CHIN_RAW")
        chin_raw_unprocessed = data[idx_chin_raw, :].copy()

        chin_ecg = np.vstack((chin_raw_unprocessed, ecg_unprocessed))
        chin_filtered, chin_model_input = chin_unscaled(chin_ecg)

        idx_chin = channel_type.index("CHIN")
        data[idx_chin, :] = chin_model_input

    # --- Filter and resample ---
    df = pd.DataFrame(data.T, columns=channels)
    df = do_initial_preprocessing(df, 200, chin_filter)
    data = df[channels].values.T

    # Replace the CHIN column with the ECG-cleaned filtered signal
    idx_chin_raw = channel_type.index("CHIN_RAW")
    data[idx_chin_raw, :] = chin_filtered

    # Final resample to 128 Hz
    data = resample(data, int(data.shape[1] / 200 * 128), axis=1)

    # --- Write processed HDF5 ---
    with h5.File(path_write, "w") as hf:
        hf.attrs["sample_rate"] = 128
        for i, (ch, ct) in enumerate(zip(channels, channel_type)):
            if "raw" in ch:
                continue  # Skip unprocessed raw channels
            ch_key = "chin1-chin2" if ch == "chin" else ch
            hf.create_dataset(
                f"channels/{ch_key.upper()}",
                data=apply_pre_process(data[i, :], ct, time_min=temporal_resolution, use_robust_scaler=is_prenormalized),
                dtype="float32",
                compression="gzip",
            )
        hf.create_dataset(
            "channels/CHIN_REM",
            data=data[idx_chin_raw, :],
            dtype="float32",
            compression="gzip",
        )

    return path_write


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def predict(
    f,
    model_path: str,
    channels_to_load: List[str],
    hypno: np.ndarray,
    EEG_chan: int = 6,
) -> pd.DataFrame:
    """
    Run the arousal detection model on a pre-processed HDF5 file handle.

    The model is evaluated across multiple EEG derivation groups (frontal,
    central, occipital when ``EEG_chan=6``; central only when ``EEG_chan=2``).
    Predictions are averaged across groups before post-processing.

    Post-processing includes
    - Square-root transform and moving average of raw probabilities.
    - Stage-specific thresholding (N1/N2/N3/REM).
    - REM post-processing using chin-EMG (CHIN_REM dataset).
    - CAISER label assignment and short-event removal.

    Parameters
    ----------
    f               : h5py.File  Open HDF5 file handle.
    model_path      : str        Directory containing ``Best_Model.h5``.
    channels_to_load: list       Channel name list (overridden internally by EEG_chan).
    hypno           : ndarray    Hypnogram at 128 Hz (integer sleep stages).
    EEG_chan        : int        Number of EEG channels: 6 (default) or 2.

    Returns
    -------
    DataFrame with columns:
        start_idx, end_idx, arousal, prob_no, prob_arousal, pp_trace
    """
    # Reset the default TF graph each call. Without this, every model rebuild
    # adds nodes to the previous graph, accumulating memory across subjects
    # and OOMing tasks that process more than ~300 sequential files (observed
    # at ~32 GB).
    tf.keras.backend.clear_session()
    import gc; gc.collect()
    hparams = YAMLHParams(HPARAMS_PATH)
    INPUT_HZ = 128
    OUTPUT_HZ = 2

    # Raw chin for REM post-processing
    emg = f["channels"]["CHIN_REM"][()]

    # Define channel groups per EEG coverage mode
    if EEG_chan == 2:
        channel_groups = [["C3-M2", "C4-M1", "CHIN1-CHIN2", "E1-M2", "ECG"]]
    else:  # 6-channel mode: frontal, central, occipital
        channel_groups = [
            ["F3-M2", "F4-M1", "CHIN1-CHIN2", "E1-M2", "ECG"],
            ["C3-M2", "C4-M1", "CHIN1-CHIN2", "E1-M2", "ECG"],
            ["O1-M2", "O2-M1", "CHIN1-CHIN2", "E1-M2", "ECG"],
        ]

    n_groups = len(channel_groups)
    n_samples = len(f["channels"][channel_groups[0][0]])
    prediction = np.zeros((n_groups, int(n_samples / INPUT_HZ * OUTPUT_HZ), 2))

    # --- Build model once for the full recording length ---
    first_group_eeg = np.stack(
        [np.array(f["channels"][c]) for c in channel_groups[0]], axis=-1
    )
    first_group_eeg = first_group_eeg[: first_group_eeg.shape[0] // 3840 * 3840, :]

    hparams["build"]["batch_shape"][1] = int(first_group_eeg.shape[0] / 3840)
    hparams["build"]["batch_shape"][0] = 1
    model = init_model(hparams["build"])
    model.load_weights(os.path.join(model_path, "Best_Model.h5"))

    # --- Predict for each channel group ---
    for k, group in enumerate(channel_groups):
        eeg = np.stack([np.array(f["channels"][c]) for c in group], axis=-1)
        eeg = eeg[: eeg.shape[0] // 3840 * 3840, :]
        n_epochs = eeg.shape[0] // 64
        if hparams["build"]["data_per_prediction"] == 64:
            prediction[k, :n_epochs, :] = model.predict(
                eeg.reshape((1, -1, hparams["data"]["data_Hz"] * 30, len(group)))
            )

    # Average predictions across channel groups
    prediction = prediction.mean(axis=0)

    # --- Post-processing ---
    arousal_prob = np.sqrt(prediction[:, 1])
    arousal_prob = arousal_prob.repeat(64)
    arousal_prob = movav(arousal_prob, window_width=192 * 3, center=True)

    # Align hypno and emg to arousal_prob length to avoid index-out-of-bounds
    # when the hypnogram or chin signal extends slightly beyond the EEG signal.
    n = len(arousal_prob)
    hypno = hypno[:n]
    emg = emg[:n]

    # Build REM post-processing trace (binary, threshold=0.2)
    pp_binary = (arousal_prob >= 0.2).astype(float)
    pp_trace = rem_post_processing(emg, pp_binary, hypno, output_pp_trace=True)
    pp_trace_epoch = np.round(np.mean(pp_trace.reshape(-1, 64), axis=1))

    # Stage-specific thresholds
    THRESHOLDS = {1: 0.42, 2: 0.45, 3: 0.35, 4: 0.51}  # N3, N2, N1, REM
    threshold = np.full(len(arousal_prob), 1.1)
    for stage, thres in THRESHOLDS.items():
        threshold[hypno == stage] = thres

    arousal_binary = (arousal_prob >= threshold).astype(float)
    arousal_binary = rem_post_processing(emg, arousal_binary, hypno)

    # CAISER label assignment
    input_dict = {
        "hypno": hypno,
        "arousal": arousal_binary,
        "limp": hypno * 0,
        "resp": hypno * 0,
    }
    pred_lab = CAISER_labels(input_dict, fs=INPUT_HZ)
    pred_lab.set_uncertainty_label(0)
    pred_lab.set_unstable_sleep_label(0)
    pred_lab = pred_lab.get_masked_arousal_label_eval()
    pred_lab = post_process_after_smoothing(
        pred_lab, [1], N_remove_length=2.5, N_max_len=30, Hz=INPUT_HZ
    )

    # Downsample to OUTPUT_HZ
    arousal_epoch = np.round(np.mean(arousal_binary.reshape(-1, 64), axis=1))
    t1 = (np.arange(len(arousal_epoch)) / OUTPUT_HZ * 200).astype(int)
    t2 = ((np.arange(len(arousal_epoch)) + 1) / OUTPUT_HZ * 200).astype(int)

    return pd.DataFrame(
        {
            "start_idx": t1,
            "end_idx": t2,
            "arousal": arousal_epoch,
            "prob_no": prediction[:, 0],
            "prob_arousal": prediction[:, 1],
            "pp_trace": pp_trace_epoch,
        }
    )


# ---------------------------------------------------------------------------
# Single-file processing entry-point (called by both sequential and parallel)
# ---------------------------------------------------------------------------

def process_single_arousal_file(
    input_path: str,
    write_path: str,
    temp_dir_path: str,
    file_info: Tuple[int, int],
    channels: List[str],
    channel_type: List[str],
    temporal_resolution: int,
) -> None:
    """
    Pre-process and score one PSG recording for arousal events.

    This function is self-contained so it can safely be called by a
    ``multiprocessing.Pool`` worker.

    Parameters
    ----------
    input_path         : str   Path to the raw input .h5 file.
    write_path         : str   Destination path for the output arousal CSV.
    temp_dir_path      : str   Directory for intermediate HDF5 files.
    file_info          : (int, int)  (0-based index, total) for progress display.
    channels           : list  Channel names.
    channel_type       : list  Semantic types (same order as *channels*).
    temporal_resolution: int   Window size (minutes) for IQR scaling.
    """
    num, total = file_info
    subject_id = input_path.split("/")[-1].split(".")[0]
    display_id = subject_id if len(subject_id) <= 20 else subject_id[:20] + ".."
    print(f"(# {num + 1}/{total}) Processing '{display_id}' [PID:{os.getpid()}]", flush=True)

    # Derive path to the matching hypnogram CSV
    hypno_path = write_path.replace("arousal", "stage")
    # Each process writes to a uniquely-named temp file to avoid collisions
    temp_h5_path = opj(temp_dir_path, f"{subject_id}.h5")

    if not os.path.exists(hypno_path):
        print(f"  -> No stage file found for '{display_id}'. Skipping.")
        return

    try:
        # Select model path: use split-specific fold if available, else default
        df_splits = pd.read_csv(MODEL_SPLIT_CSV)
        if (df_splits["File_name"] == subject_id).any():
            rel_path = df_splits.loc[
                df_splits["File_name"] == subject_id, "Model_path"
            ].values[0]
            model_path = opj(AROUSAL_PKG_DIR, rel_path)
            # Fall back to default if fold-specific weights don't exist
            if not os.path.isfile(os.path.join(model_path, "Best_Model.h5")):
                model_path = DEFAULT_MODEL_DIR
        else:
            model_path = DEFAULT_MODEL_DIR

        # Pre-process raw signals → scaled intermediate HDF5
        pre_process_temporal_scaling(
            input_path, temp_h5_path, channels.copy(), channel_type,
            temporal_resolution, chin_filter=False,
        )

        # Load hypnogram at 128 Hz
        hypnogram = (
            pd.read_csv(hypno_path)["stage"].fillna(5).values.repeat(128)
        )

        # 6-channel default (frontal+central+occipital); fall back to
        # 2-channel (C3/C4 only) when frontal/occipital are missing or
        # otherwise cause predict() to fail.
        with h5.File(temp_h5_path, "r") as hf:
            try:
                df = predict(hf, model_path, channels, hypnogram, EEG_chan=6)
            except Exception as exc6:
                print(f"  -> 6-chn predict failed for '{display_id}': {exc6}. Falling back to 2-chn.", flush=True)
                df = predict(hf, model_path, channels, hypnogram, EEG_chan=2)

        df.to_csv(write_path, index=False)

    except ValueError as exc:
        print(f"  -> SKIP '{display_id}': {exc}", flush=True)
        return
    except Exception as exc:
        print(f"  -> ERROR processing '{display_id}': {exc}", flush=True)

    finally:
        # Always clean up the per-subject temporary file
        if os.path.exists(temp_h5_path):
            os.remove(temp_h5_path)


# ---------------------------------------------------------------------------
# Dispatcher: sequential vs. parallel
# ---------------------------------------------------------------------------

def CAISR_arousal(
    in_paths: List[str],
    save_paths: List[str],
    temp_dir_path: str,
    multiprocess: bool,
) -> None:
    """
    Dispatch arousal detection over all subjects, sequentially or in parallel.

    Parameters
    ----------
    in_paths      : list of str  Raw input .h5 file paths.
    save_paths    : list of str  Output CSV paths (one per subject).
    temp_dir_path : str          Shared directory for intermediate HDF5 files.
    multiprocess  : bool         If True, use ``multiprocessing.Pool``.
    """
    # PSG channels used for model input (order must match channel_type).
    # Default = 6-EEG (frontal+central+occipital); pre_process_temporal_scaling
    # drops F3/F4/O1/O2 when missing (no substitution), so predict() with
    # EEG_chan=6 will fail-fast and process_single_arousal_file falls back to
    # EEG_chan=2 using C3-M2 / C4-M1.
    channels = [
        "f3-m2", "f4-m1", "c3-m2", "c4-m1", "o1-m2", "o2-m1",
        "e1-m2", "chin1-chin2", "ecg", "chin1-chin2",
    ]
    channel_type = [
        "EEG", "EEG", "EEG", "EEG", "EEG", "EEG",
        "EYE", "CHIN", "ECG", "CHIN_RAW",
    ]
    temporal_resolution = 5  # minutes

    if multiprocess and len(in_paths) > 1:
        num_workers = min(MAX_WORKERS, len(in_paths))
        print(f">> Multiprocessing enabled — launching {num_workers} workers …")
        tasks = [
            (path, save, temp_dir_path, (i, len(in_paths)), channels, channel_type, temporal_resolution)
            for i, (path, save) in enumerate(zip(in_paths, save_paths))
        ]
        with multiprocessing.Pool(processes=num_workers) as pool:
            results = pool.starmap_async(process_single_arousal_file, tasks)
            try:
                results.get()  # This will raise if any worker was killed at OS level
            except Exception as e:
                print(f"  -> Pool-level ERROR: {e}", flush=True)
    else:
        print(">> Sequential processing …")
        for i, (path, save) in enumerate(zip(in_paths, save_paths)):
            process_single_arousal_file(
                path, save, temp_dir_path, (i, len(in_paths)),
                channels, channel_type, temporal_resolution,
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run CAISR arousal-event detection on prepared PSG HDF5 files."
    )
    parser.add_argument(
        "--input_data_dir",
        type=str,
        required=True,
        help="Folder containing pre-prepared .h5 PSG data files.",
    )
    parser.add_argument(
        "--output_csv_dir",
        type=str,
        required=True,
        help="Root folder where arousal CSV outputs will be written.",
    )
    parser.add_argument(
        "--param_dir",
        type=str,
        required=True,
        help="Folder containing 'arousal.csv' with run parameters.",
    )
    args = parser.parse_args()

    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"GPUs available: {len(gpu_devices)}")

    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp(prefix="caisr_arousal_")
        print(f"Temporary directory: {temp_dir}")

        timer('Starting CAISR arousal detection')

        input_files = sorted(
            glob(os.path.join(args.input_data_dir, "*.h5")) +
            glob(os.path.join(args.input_data_dir, "*.edf")) +
            glob(os.path.join(args.input_data_dir, "*.zarr"))
        )
        overwrite, multiprocess = extract_run_parameters(
            os.path.join(args.param_dir, "arousal.csv")
        )
        in_paths, save_paths = set_output_paths(input_files, args.output_csv_dir, overwrite)

        if in_paths:
            CAISR_arousal(in_paths, save_paths, temp_dir, multiprocess)
        else:
            print(">> No files to process.")

        timer('CAISR arousal detection complete')

    finally:
        if temp_dir and os.path.exists(temp_dir):
            print(f"Cleaning up temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
