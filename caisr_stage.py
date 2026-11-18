"""
CAISR Stage: Sleep Staging Module
Author: Samaneh Nasiri, PhD
Cleaned/Refactored for Public Repository

Description:
    Runs the sleep staging model (GraphSleepNet/ProductGraphSleepNet) on PSG
    input files.
    Supports both sequential and multiprocessing execution.
    Accepts either pre-processed .h5 files (legacy CAISR format) or raw .edf
    recordings; format is auto-detected per file by extension.

Requirements:
    - Input data must be .h5 and/or .edf in the input directory.
    - Parameter file (stage.csv) must exist.
    - Pre-trained model weights must be available in `model_dir`.
"""

import sys
import os
import time
import glob
import warnings
import argparse
import logging
import multiprocessing
from math import gcd
import numpy as np
import pandas as pd
import h5py
import tensorflow as tf
from tensorflow import keras
from scipy.signal import resample_poly
from sklearn.preprocessing import RobustScaler

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
tf.get_logger().setLevel(logging.ERROR)

# --- Local Imports ---
# NOTE: Ensure the folders 'stage' and 'stage/graphsleepnet' are present in the root directory.
sys.path.insert(1, 'stage/graphsleepnet')
sys.path.insert(1, 'stage')

try:
    from DE_PSD import *
    from gcn_features import *
    from utils_docker import *
    from ProductGraphSleepNet import *
    from utils_model import *
    # Keep an explicit handle on the H5 loader so the unified dispatcher below
    # can fall back to it when the input is a .h5 file.
    from utils_docker import select_signals_cohort as _select_signals_cohort_h5
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure 'stage' and 'stage/graphsleepnet' directories exist and contain required scripts.")
    sys.exit(1)

# EDF loader (only required when an .edf input is encountered).
try:
    from edf_loader_pyedflib import load_edf_for_caisr as _load_edf_for_caisr
except ImportError:
    _load_edf_for_caisr = None

# TS-Zarr loader (BDSP primary stores); needs `pip install bdsp`.
from zarr_loader import is_zarr_input, load_raw_for_caisr


def _select_signals_cohort_edf(path: str):
    """EDF counterpart of utils_docker.select_signals_cohort for caisr_stage.

    Returns the same ``(sigs, new_Fs, channels, length_at_200Hz)`` tuple as
    the H5 loader, so ``process_single_file`` is format-agnostic.
    """
    if _load_edf_for_caisr is None and not is_zarr_input(path):
        raise ImportError(
            "EDF input detected but `pyedflib` is not installed. "
            "Run `pip install pyedflib` inside the caisr_stage environment."
        )
    new_Fs = 100
    Fs = 100
    period_length_sec = 0.5
    fs_output_ref = 200

    channel_data, channel_fs = load_raw_for_caisr(path, _load_edf_for_caisr)

    eeg = ['c3-m2', 'c4-m1']
    common_sigs = ['e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg']
    expected_channels = eeg + common_sigs

    available_channels = set(channel_data.keys())
    channels_found = [ch for ch in expected_channels if ch in available_channels]
    channels_missing = [ch for ch in expected_channels if ch not in available_channels]

    if channels_missing:
        the_id = os.path.splitext(os.path.basename(path))[0]
        raise ValueError(
            f"SKIP '{the_id}': missing required channels: {channels_missing}"
        )
    if not channels_found:
        raise ValueError(f"No recognisable channels found in {path}.")

    resampled = {}
    for ch in channels_found:
        sig = channel_data[ch].astype(np.float64)
        src_fs = float(channel_fs[ch])
        if src_fs != float(new_Fs):
            g = gcd(int(new_Fs), int(src_fs))
            sig = resample_poly(sig, int(new_Fs) // g, int(src_fs) // g)
        resampled[ch] = sig

    n_samples = min(len(v) for v in resampled.values())
    duration_sec = n_samples / new_Fs
    length_data = int(duration_sec * fs_output_ref)

    signal_matrix = {
        ch: (resampled[ch][:n_samples] if ch in resampled else np.zeros(n_samples))
        for ch in expected_channels
    }
    signals = pd.DataFrame(signal_matrix)[expected_channels]

    step = int(Fs * period_length_sec)
    n_steps = len(signals) // step
    signals = signals.iloc[: n_steps * step, :]

    sigs = signals[expected_channels].values
    sigs, _chan_inds = clip_noisy_values(sigs, new_Fs, period_length_sec)
    sigs = RobustScaler().fit_transform(sigs).T

    return sigs, new_Fs, expected_channels, length_data


def select_signals_cohort(path: str):
    """Unified dispatcher: load + pre-process from either .h5 or .edf."""
    if path.lower().endswith('.edf') or is_zarr_input(path):
        return _select_signals_cohort_edf(path)
    return _select_signals_cohort_h5(path)


def timer(tag: str) -> None:
    print(tag)
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))
    print()


def extract_run_parameters(param_csv: str) -> 'tuple[bool, bool]':
    """Extracts run parameters from a CSV file."""
    if not os.path.exists(param_csv):
        raise FileNotFoundError(f'Run parameter file not found at {param_csv}.')
        
    params = pd.read_csv(param_csv)
    overwrite = params['overwrite'].values[0]
    
    # Get the multiprocess flag.
    multiprocess_val = params.get('multiprocess', pd.Series([False])).values[0]
    multiprocess = str(multiprocess_val).strip().lower() == 'true'
    
    return overwrite, multiprocess


def set_output_paths(input_paths: 'list[str]', csv_folder: str, overwrite: bool) -> 'tuple[list[str], list[str]]':
    IDs = [p.split(os.sep)[-1].split('.')[0] for p in input_paths]
    
    # Ensure output directory exists
    stage_out_dir = os.path.join(csv_folder, 'stage')
    os.makedirs(stage_out_dir, exist_ok=True)
    
    csv_paths = [os.path.join(stage_out_dir, f'{ID}_stage.csv') for ID in IDs]
    
    assert len(input_paths) == len(csv_paths), 'SETUP ERROR: Number of input and CSV files mismatch.'
    input_paths, csv_paths = filter_already_processed_files(input_paths, csv_paths, overwrite)
    return input_paths, csv_paths


def filter_already_processed_files(input_paths: 'list[str]', csv_paths: 'list[str]', overwrite: bool) -> 'tuple[list[str], list[str]]':
    total = len(input_paths)
    if not overwrite:
        todo_indices = [i for i, path in enumerate(csv_paths) if not os.path.exists(path)]
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        processed_count = total - len(todo_indices)
    else:
        processed_count = 0 
    
    tag = '(overwrite)' if overwrite else ''
    print(f'>> {processed_count}/{total} files already processed')
    print(f'>> {len(input_paths)} files to process {tag}\n')
    return input_paths, csv_paths


def _build_stage_model(model_path: str, model_params: dict):
    """Build ProductGraphSleepNet and load fold-3 weights."""
    sample_shape = (model_params['context'], model_params['w'], model_params['h'])
    model = build_ProductGraphSleepNet(
        model_params['cheb_k'], model_params['num_of_chev_filters'], model_params['num_of_time_filters'],
        model_params['time_conv_strides'], model_params['cheb_polynomials'],
        model_params['time_conv_kernel'], sample_shape, model_params['num_block'],
        model_params['optimizer'],
        model_params['conf_adj'] == 'GL', model_params['GLalpha'], model_params['regularizer'],
        model_params['GRU_Cell'], model_params['attn_heads'], model_params['dropout'],
    )
    weights_file = os.path.join(model_path, 'weights_fold_3.h5')
    if not os.path.exists(weights_file):
        raise FileNotFoundError(f"Weights file not found at {weights_file}")
    model.load_weights(weights_file)
    return model


def process_single_file(path: str, save_path: str, model_path: str, model_params: dict, file_info: 'tuple[int, int]', model=None):
    """
    Processes a single input file for sleep staging.
    Builds its own model when ``model`` is None (multiprocess workers); otherwise
    reuses the caller's pre-built TF model — saves ~30 s per subject in the
    sequential path by skipping rebuild + weight reload.
    """
    num, total = file_info

    if model is None:
        try:
            model = _build_stage_model(model_path, model_params)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return

    context = model_params['context']

    the_id = path.split(os.sep)[-1].split('.')[0]
    tag = the_id if len(the_id) < 21 else the_id[:20] + '..'
    print(f'(# {num + 1}/{total}) Processing "{tag}" [PID:{os.getpid()}]')

    try:
        signals, Fs, sig_tages, length_data = select_signals_cohort(path)
    except Exception as e:
        print(f'Error loading signals for {tag}: {e}', flush=True)
        return

    window = 30
    fs_input = 200
    
    try:
        segs = segment_data_unseen(signals)
        MYpsd, MYde = graph_feat_extraction_unseen_docker(segs, sig_tages, Fs, window)
        image = AddContext(MYde, context)
        image = np.squeeze(np.array(image))
        
        # Prediction
        prediction = model.predict(image, verbose=0) 
        pred_per_subject = prediction.argmax(axis=1) + 1
        
        # Padding predictions to match input length (context window offset)
        pred_per_subject = np.concatenate([[np.nan] * 3, pred_per_subject, [np.nan] * 3])
        pred_per_subject = np.repeat(pred_per_subject, window, axis=0) # Expand to seconds/samples if needed
        
        nan_row = np.empty((1, prediction.shape[1])); nan_row[:] = np.nan
        for index in [0, 1, 2]:
            prediction = np.insert(prediction, index, nan_row, axis=0)
        for i in range(3):
            prediction = np.vstack((prediction, nan_row))
        prediction = np.repeat(prediction, window, axis=0)

        # Create DataFrame
        t1 = (np.arange(len(prediction)) * fs_input).astype(int)
        t2 = (np.arange(len(prediction)) + 1) * fs_input
        
        df = pd.DataFrame({
            'start_idx': t1, 'end_idx': t2, 'stage': pred_per_subject,
            'prob_n3': prediction[:, 0], 'prob_n2': prediction[:, 1],
            'prob_n1': prediction[:, 2], 'prob_r': prediction[:, 3],
            'prob_w': prediction[:, 4]
        })
        
        # Ensure output length matches original data length
        t1_full = (np.arange(length_data / fs_input) * fs_input).astype(int)
        t2_full = ((np.arange(length_data / fs_input) + 1) * fs_input).astype(int)
        
        df_matched = pd.DataFrame({
            'start_idx': t1_full, 'end_idx': t2_full, 'stage': np.nan,
            'prob_n3': np.nan, 'prob_n2': np.nan, 'prob_n1': np.nan, 'prob_r': np.nan, 'prob_w': np.nan
        })
        
        # Copy valid predictions into matched dataframe
        limit = min(len(df), len(df_matched))
        df_matched.iloc[0:limit] = df.iloc[0:limit]
        
        df_matched.to_csv(save_path, index=False)
        
    except Exception as error:
        print(f'({num}) Failure during feature extraction/prediction for {tag}: {error}')


def CAISR_stage(in_paths: 'list[str]', save_paths: 'list[str]', model_path: str, multiprocess: bool):
    timer('* Starting "caisr_stage" (created by Samaneh Nasiri, PhD)')
    
    # --- Define Model Parameters ---
    try:
        # Check TF version for optimizer compatibility
        if keras.optimizers.Adam(learning_rate=0.0001).get_config()['name'] == "adam": 
             opt = keras.optimizers.Adam(learning_rate=0.0001, decay=0.0, clipnorm=1)
        else:
             opt = keras.optimizers.Adam(lr=0.0001, decay=0.0, clipnorm=1)
    except:
        opt = keras.optimizers.Adam(learning_rate=0.0001, clipnorm=1)

    model_params = {
        'optimizer': opt,
        'regularizer': keras.regularizers.l1_l2(l1=0.001, l2=0.001),
        'w': 7, 'h': 9, 'context': 7,
        'conf_adj': 'GL', 'GLalpha': 0.0,
        'num_of_chev_filters': 128, 'num_of_time_filters': 128,
        'time_conv_strides': 1, 'time_conv_kernel': 3,
        'num_block': 1, 'cheb_k': 3,
        'cheb_polynomials': None, 'dropout': 0.60,
        'GRU_Cell': 256, 'attn_heads': 40,
    }

    # --- Dispatch jobs ---
    if multiprocess and len(in_paths) > 1:
        # Cap at 12 (24 OOMed at ~900 GB on a single node; each TF worker holds ~10 GB).
        # maxtasksperchild recycles worker procs to release accumulated TF state.
        num_workers = min(12, multiprocessing.cpu_count())
        print(f">> Multiprocessing enabled. Starting parallel processing with {num_workers} workers...")

        tasks = [
            (path, save_path, model_path, model_params, (num, len(in_paths)))
            for num, (path, save_path) in enumerate(zip(in_paths, save_paths))
        ]

        with multiprocessing.Pool(processes=num_workers, maxtasksperchild=20) as pool:
            pool.starmap(process_single_file, tasks)

    else:
        print(">> Starting sequential processing...")
        # Build model ONCE and reuse across subjects — avoids ~30 s/subject of
        # rebuild + weight-reload overhead.
        try:
            shared_model = _build_stage_model(model_path, model_params)
        except FileNotFoundError as exc:
            print(f"Error: {exc}")
            return
        print(">> Model built once; reusing across all subjects.")
        for num, (path, save_path) in enumerate(zip(in_paths, save_paths)):
            process_single_file(path, save_path, model_path, model_params, (num, len(in_paths)), model=shared_model)
    
    timer('* Finishing "caisr_stage"')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAISR sleep staging.")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Path to the input data folder (containing .h5 files)")
    parser.add_argument("--output_csv_dir", type=str, required=True, help="Path to save the output features")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pre-trained model directory")
    parser.add_argument("--param_dir", type=str, required=True, help="Folder containing the run parameters file (stage.csv)")
    args = parser.parse_args()

    # GPU Check
    if tf.test.is_built_with_cuda():
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f">> GPU detected: {len(gpus)} device(s)")
        else:
            print(">> TensorFlow built with CUDA, but no GPU detected at runtime.")
    else:
        print(">> TensorFlow is CPU only.")

    input_files = sorted(
        glob.glob(os.path.join(args.input_data_dir, '*.h5')) +
        glob.glob(os.path.join(args.input_data_dir, '*.edf')) +
        glob.glob(os.path.join(args.input_data_dir, '*.zarr'))
    )
    
    # Load parameters
    param_file = os.path.join(args.param_dir, 'stage.csv')
    if not os.path.exists(param_file):
         # Create default if missing for user convenience
         print(f">> Parameter file not found at {param_file}. Creating default.")
         os.makedirs(args.param_dir, exist_ok=True)
         pd.DataFrame({'overwrite': [False], 'multiprocess': [False]}).to_csv(param_file, index=False)

    overwrite, multiprocess = extract_run_parameters(param_file)
    
    in_paths, save_paths = set_output_paths(input_files, args.output_csv_dir, overwrite)
    
    if in_paths:
        CAISR_stage(in_paths, save_paths, args.model_dir, multiprocess)
    else:
        print(">> No files to process.")
        