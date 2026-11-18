"""
CAISR Limb: Limb Movement Detection
Author: Samaneh Nasiri, PhD
Cleaned/Refactored for Public Repository

Description:
    Runs limb movement detection logic (PLM/LM) on EMG channels.

    Accepts either pre-processed HDF5 files (legacy CAISR format, see
    README) or raw EDF recordings. Input format is auto-detected per
    file based on extension; both code paths produce the same output.
"""

import os
import sys
import glob
import time
import argparse
import multiprocessing
from math import gcd
import h5py
import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import resample_poly
from typing import List, Tuple


def _input_is_microvolt(path: str, sample: np.ndarray) -> bool:
    """
    Decide whether a CAISR input is stored in microvolts (vs volts).
    Trust ``f.attrs['unit_voltage']`` when present (H5 only); otherwise
    fall back to amplitude — EMG in V has 99th-pct |x| ~ 1e-5–1e-4,
    EMG in uV ~ 10–100. EDF inputs go through the EDF loader which
    returns Volts, so the amplitude path correctly classifies them.
    """
    unit = None
    if path.lower().endswith('.h5'):
        try:
            with h5py.File(path, "r") as f:
                unit = f.attrs.get("unit_voltage", None)
        except Exception:
            unit = None
    if unit is not None:
        try:
            unit = unit.decode() if isinstance(unit, bytes) else unit
            unit = str(unit).strip().lower().replace("μ", "u")
        except Exception:
            unit = None
    if unit in ("uv", "microvolt", "microvolts"):
        return True
    if unit in ("v", "volt", "volts"):
        return False
    p99 = float(np.nanpercentile(np.abs(np.asarray(sample)), 99))
    return p99 > 0.5


sys.path.insert(1, './limb')
from utils_limb import *
from utils_docker import *
# Keep an explicit handle on the H5 loader so the unified dispatcher below
# can fall back to it when the input is a .h5 file.
from utils_docker import select_signals_cohort as _select_signals_cohort_h5

# EDF loader (only imported here, not at function call time, so a missing
# pyedflib install fails fast with a clear error if EDF inputs are present).
try:
    from edf_loader_pyedflib import load_edf_for_caisr as _load_edf_for_caisr
except ImportError:
    _load_edf_for_caisr = None  # surface a clear error only if EDF input is fed in

# TS-Zarr loader (BDSP primary stores); needs `pip install bdsp`.
from zarr_loader import is_zarr_input, load_raw_for_caisr


def _select_signals_cohort_edf(path: str, intermediate_fs: int = 200) -> Tuple[np.ndarray, int]:
    """EDF counterpart of utils_docker.select_signals_cohort for caisr_limb.

    Loads the two leg-EMG channels (lat, rat), resamples each to
    ``intermediate_fs`` (default 200 Hz, matching the H5 path), and stacks
    them into the (n_samples, 2) array the downstream limb pipeline expects.
    """
    if _load_edf_for_caisr is None and not is_zarr_input(path):
        raise ImportError(
            "EDF input detected but `pyedflib` is not installed. "
            "Run `pip install pyedflib` inside the caisr_limb environment."
        )
    the_id = os.path.splitext(os.path.basename(path))[0]
    channel_data, channel_fs = load_raw_for_caisr(path, _load_edf_for_caisr)

    REQUIRED = ['lat', 'rat']
    missing = [ch for ch in REQUIRED if ch not in channel_data]
    if missing:
        raise ValueError(
            f'"{the_id}" is missing required EMG channel(s) {missing}. '
            f'Subject will be skipped.'
        )

    resampled = []
    for ch in REQUIRED:
        sig = np.asarray(channel_data[ch], dtype=np.float64)
        src_fs = float(channel_fs[ch])
        if src_fs != float(intermediate_fs):
            g = gcd(int(intermediate_fs), int(src_fs))
            sig = resample_poly(sig, int(intermediate_fs) // g, int(src_fs) // g)
        resampled.append(sig)

    n_samples = min(len(s) for s in resampled)
    EMGs = np.column_stack([s[:n_samples] for s in resampled])
    return EMGs, intermediate_fs


def select_signals_cohort(path: str):
    """Unified dispatcher: load limb EMG signals from either an .h5 or .edf file."""
    if path.lower().endswith('.edf') or is_zarr_input(path):
        return _select_signals_cohort_edf(path)
    return _select_signals_cohort_h5(path)

def timer(tag: str) -> None:
    print(tag)
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))
    print()

def extract_run_parameters(param_path: str) -> Tuple[bool, int]:
    if not os.path.exists(param_path):
        return False, 1 # Default
    params = pd.read_csv(param_path)
    overwrite = params['overwrite'].values[0]
    val = params.get('multiprocess', pd.Series([False])).values[0]
    
    if str(val).lower() == 'true': workers = 0
    elif str(val).lower() == 'false': workers = 1
    else:
        try: workers = int(val)
        except: workers = 1
    return overwrite, workers

def set_output_paths(input_paths: List[str], csv_folder: str, overwrite: bool) -> Tuple[List[str], List[str]]:
    limb_out_dir = os.path.join(csv_folder, 'limb')
    os.makedirs(limb_out_dir, exist_ok=True)
    
    IDs = [p.split(os.sep)[-1].split('.')[0] for p in input_paths]
    csv_paths = [os.path.join(limb_out_dir, f'{ID}_limb.csv') for ID in IDs]
    input_paths, csv_paths = filter_already_processed_files(input_paths, csv_paths, overwrite)
    return input_paths, csv_paths

def filter_already_processed_files(input_paths: List[str], csv_paths: List[str], overwrite: bool) -> Tuple[List[str], List[str]]:
    total = len(input_paths)
    if not overwrite:
        todo_indices = [i for i, path in enumerate(csv_paths) if not os.path.exists(path)]
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        processed_count = total - len(todo_indices)
    else:
        processed_count = 0

    tag = '(overwrite)' if overwrite else ''
    print(f'>> {processed_count}/{total} files already processed\n>> {len(input_paths)} to go.. {tag}\n')
    return input_paths, csv_paths

def create_empty_output(total_samples, save_path):
    print(' - Creating empty limb movement file.')
    fs_input = 200 
    num_epochs = total_samples // fs_input
    zeros_1hz = np.zeros(num_epochs, dtype=int)
    ones_1hz = np.ones(num_epochs, dtype=float)
    t1 = (np.arange(num_epochs) * fs_input).astype(int)
    t2 = ((np.arange(num_epochs) + 1) * fs_input).astype(int)
    
    df = pd.DataFrame({
        'start_idx': t1, 'end_idx': t2, 'limb': zeros_1hz, 'plm': zeros_1hz,
        'prob_no': ones_1hz, 'prob_limb': zeros_1hz.astype(float)
    })
    df.to_csv(save_path, index=False)

def process_single_limb_file(path: str, save_path: str, file_info: Tuple[int, int]):
    num, total_files = file_info
    the_id = path.split(os.sep)[-1].split('.')[0]
    tag = the_id if len(the_id) < 21 else the_id[:20] + '..'
    print(f'(# {num + 1}/{total_files}) Processing "{tag}" [PID:{os.getpid()}]')
    
    try:
        EMGs, Fs = select_signals_cohort(path)
        if EMGs is None or np.all(EMGs == 0):
            print(f' - EMG empty for "{tag}".')
            # Assuming standard length if file load fails is tricky, but here we assume Fs=200
            # Ideally select_signals_cohort returns length even if signals are flat.
            return 
            
        new_Fs = 100
        emg_signal_resampled = resample_poly(EMGs, new_Fs, Fs) if new_Fs != Fs else EMGs
    except Exception as e:
        print(f'ERROR loading signals for "{the_id}": {e}')
        return
    
    signal_emg = np.array(emg_signal_resampled.T)
    if not _input_is_microvolt(path, signal_emg):
        signal_emg = signal_emg * 1e6
    sample_rate = 100

    lower_threshold_uV = 2
    upper_threshold_uV = 10
    min_duration_sec = 0.5
    max_duration_sec = 10.0
    filter_hp_freq_Hz = 15
    sliding_window_len_sec = 0.5
    filter_order = 100
    merge_lm_within_sec = 0.5

    all_channels_lm_events = []
    num_chns = signal_emg.shape[0]

    for ch in range(num_chns):
        b = signal.firwin(filter_order + 1, filter_hp_freq_Hz / (sample_rate / 2), pass_zero=False)
        delay = int(filter_order / 2)
        LM_line = signal.lfilter(b, 1, signal_emg[ch, :], axis=0)
        LM_line = np.concatenate((np.abs(LM_line[delay:]), np.zeros(delay)))
        
        LM_above_thresh_high = np.where(LM_line >= upper_threshold_uV)[0]
        if len(LM_above_thresh_high) == 0:
            continue

        window_len = int(sliding_window_len_sec * sample_rate)
        max_dur = int(max_duration_sec * sample_rate)
        avg_LM_line = signal.lfilter(np.ones(window_len) / window_len, 1, LM_line)

        LM_below_thresh_low = LM_above_thresh_high.copy()
        last_crossed_index = 0
        for k in range(len(LM_above_thresh_high)):
            start = LM_above_thresh_high[k] + 1
            if start > last_crossed_index:
                cand = np.where(avg_LM_line[start : start + max_dur] < lower_threshold_uV)[0]
                if len(cand) > 0:
                    LM_below_thresh_low[k] = start + cand[0]
                    last_crossed_index = LM_below_thresh_low[k]
        
        LM_candidates = np.column_stack((LM_above_thresh_high, LM_below_thresh_low))
        lm_dur = LM_candidates[:, 1] - LM_candidates[:, 0]
        LM_dur_range = np.ceil(np.array([min_duration_sec, max_duration_sec]) * sample_rate)
        valid_indices = np.where((lm_dur >= LM_dur_range[0]) & (lm_dur <= LM_dur_range[1]))[0]
        LM_candidates = LM_candidates[valid_indices]

        if len(LM_candidates) > 0:
            min_samples = int(round(merge_lm_within_sec * sample_rate))
            # FIX: Correctly call merge_nearby_events and extract the first element
            LM_candidates = merge_nearby_events(LM_candidates, min_samples)
            LM_candidates = LM_candidates[0]
            all_channels_lm_events.extend(LM_candidates.tolist())

    if not all_channels_lm_events:
        print(f" - No valid limb movements found for {tag}.")
        create_empty_output(EMGs.shape[0], save_path)
        return

    final_lm_events = np.array(all_channels_lm_events)
    final_lm_events = final_lm_events[final_lm_events[:, 0].argsort()]
    min_samples = int(round(merge_lm_within_sec * sample_rate))
    # FIX: Correctly call merge_nearby_events and extract the first element here as well
    final_lm_events = merge_nearby_events(final_lm_events, min_samples)
    final_lm_events = final_lm_events[0]

    PLM_events = []
    if len(final_lm_events) >= 4:
        PLM_min_interval_sec = 5
        PLM_max_interval_sec = 90
        PLM_min_LM_req = 4
        PLM_candidates = np.zeros(len(final_lm_events), dtype=bool)
        interval = np.diff(final_lm_events[:, 0]) / sample_rate
        is_in_range = (interval >= PLM_min_interval_sec) & (interval <= PLM_max_interval_sec)
        for i in range(len(is_in_range) - (PLM_min_LM_req - 2)):
            if np.all(is_in_range[i : i + PLM_min_LM_req - 1]):
                PLM_candidates[i : i + PLM_min_LM_req] = True
        PLM_events = final_lm_events[PLM_candidates]

    LM_binary = np.zeros(emg_signal_resampled.shape[0])
    for start, end in final_lm_events:
        LM_binary[start:end+1] = 1
    
    PLM_binary = np.zeros(emg_signal_resampled.shape[0])
    for start, end in PLM_events:
        PLM_binary[start:end+1] = 1
    
    fs_input = 200
    LM_binary_resampled = resample_poly(LM_binary, fs_input, new_Fs)
    PLM_binary_resampled = resample_poly(PLM_binary, fs_input, new_Fs)
    
    LM_1Hz_matrix = reshape_array(LM_binary_resampled, window_size=fs_input)
    PLM_1Hz_matrix = reshape_array(PLM_binary_resampled, window_size=fs_input)

    majority_LM = (np.sum(LM_1Hz_matrix, axis=1) > fs_input / 2).astype(int)
    majority_PLM = (np.sum(PLM_1Hz_matrix, axis=1) > fs_input / 2).astype(int)

    one_hot_labels = np.eye(2)[majority_LM]
    
    t1 = (np.arange(len(majority_LM)) * fs_input).astype(int)
    t2 = ((np.arange(len(majority_LM)) + 1) * fs_input).astype(int)
    
    df = pd.DataFrame({
        'start_idx': t1, 'end_idx': t2, 'limb': majority_LM, 'plm': majority_PLM,
        'prob_no': one_hot_labels[:, 0], 'prob_limb': one_hot_labels[:, 1]
    })
    df.to_csv(save_path, index=False)

def process_limb_dispatcher(in_paths: List[str], save_paths: List[str], worker_count: int):
    if not in_paths:
        print(">> No files to process.")
    else:
        num_workers = min(os.cpu_count(), len(in_paths)) if worker_count == 0 else worker_count

        tasks = [(path, save_path, (num, len(in_paths))) for num, (path, save_path) in enumerate(zip(in_paths, save_paths))]

        if num_workers > 1:
            print(f">> Multiprocessing enabled ({num_workers} workers).")
            with multiprocessing.Pool(processes=num_workers) as pool:
                pool.starmap(process_single_limb_file, tasks)
        else:
            print(">> Sequential processing.")
            for task in tasks:
                process_single_limb_file(*task)

    timer('* Finishing "caisr_limb"')


def CAISR_limb(
    in_paths: List[str],
    csv_folder: str,
    overwrite: bool = False,
    worker_count: int = 1,
) -> None:
    timer('* Starting "caisr_limb" (created by Samaneh Nasiri, PhD)')
    in_paths, save_paths = set_output_paths(in_paths, csv_folder, overwrite)
    process_limb_dispatcher(in_paths, save_paths, worker_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAISR limb movement detection.")
    parser.add_argument("--input_data_dir", type=str, required=True,
                        help="Folder containing .h5 and/or .edf PSG data files.")
    parser.add_argument("--output_csv_dir", type=str, required=True,
                        help="Folder where limb CSV outputs will be written.")
    parser.add_argument("--param_dir", type=str, required=True,
                        help="Folder containing 'limb.csv' with run parameters.")
    args = parser.parse_args()

    input_files = sorted(
        glob.glob(os.path.join(args.input_data_dir, '*.h5')) +
        glob.glob(os.path.join(args.input_data_dir, '*.edf')) +
        glob.glob(os.path.join(args.input_data_dir, '*.zarr'))
    )

    param_file = os.path.join(args.param_dir, 'limb.csv')
    if not os.path.exists(param_file):
        os.makedirs(args.param_dir, exist_ok=True)
        pd.DataFrame({'overwrite': [False], 'multiprocess': [False]}).to_csv(param_file, index=False)

    overwrite, worker_count = extract_run_parameters(param_file)

    CAISR_limb(input_files, args.output_csv_dir, overwrite=overwrite, worker_count=worker_count)
