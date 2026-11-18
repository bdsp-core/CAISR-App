"""
CAISR Resp: Respiratory Event Detection
Author: Thijs Nassi, PhD
Cleaned/Refactored for Public Repository

Description:
    Runs rule-based respiratory event detection (Apnea, Hypopnea, RERA).
    Accepts either pre-processed .h5 files (legacy CAISR format) or raw .edf
    recordings; format is auto-detected per file by extension.
    Uses Ray for parallel processing.
"""

import os
import ray
import time
import logging
import argparse
import glob
from math import gcd
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.signal import resample_poly

# --- Local Imports ---
# Ensure 'resp' folder exists
from resp.rule_based_functions.SpO2_drop_analysis import *
from resp.rule_based_functions.Ventilation_drop_analysis import *
from resp.rule_based_functions.Hypopnea_analysis import *
from resp.rule_based_functions.RERA_analysis import *
from resp.rule_based_functions.Resp_event_hierarchy import *
from resp.rule_based_functions.Post_processing import *
from resp.utils_functions.Data_loaders import *
from resp.utils_functions.Data_writers import *

# EDF loader (only required when an .edf input is encountered).
try:
    from edf_loader_pyedflib import load_edf_for_caisr as _load_edf_for_caisr
except ImportError:
    _load_edf_for_caisr = None

# TS-Zarr loader (BDSP primary stores); needs `pip install bdsp`.
from zarr_loader import is_zarr_input, load_raw_for_caisr


def _load_breathing_signals_from_edf(
    path: str,
    csv_folder: str,
    add_CAISR: bool = True,
    verbose: int = 5,
    original_fs: int = 200,
    new_fs: int = 10,
) -> Tuple[pd.DataFrame, dict]:
    """EDF counterpart of load_breathing_signals_from_prepared_data.

    Drop-in replacement: same signature, same return tuple ``(signals_df, hdr)``.
    Loads the EDF, resamples each channel individually to ``original_fs``
    (200 Hz), then runs the identical post-load pipeline as the H5 loader.
    """
    if _load_edf_for_caisr is None and not is_zarr_input(path):
        raise ImportError(
            "EDF input detected but `pyedflib` is not installed. "
            "Run `pip install pyedflib` inside the caisr_resp environment."
        )
    the_id = os.path.splitext(os.path.basename(path))[0]
    channel_data, channel_fs = load_raw_for_caisr(path, _load_edf_for_caisr)
    if not channel_data:
        raise ValueError(f'No channels could be loaded from {path}.')

    resampled = {}
    for ch, sig in channel_data.items():
        sig = np.asarray(sig, dtype=np.float64)
        src_fs = float(channel_fs[ch])
        if src_fs > float(original_fs):
            g = gcd(int(original_fs), int(src_fs))
            sig = resample_poly(sig, int(original_fs) // g, int(src_fs) // g)
        elif src_fs < float(original_fs):
            sig = np.repeat(sig, int(round(original_fs / src_fs)))
        resampled[ch] = sig

    n_samples = min(len(v) for v in resampled.values())
    resampled = {ch: v[:n_samples] for ch, v in resampled.items()}

    REQUIRED = ['abd', 'chest', 'spo2']
    missing = [ch for ch in REQUIRED if ch not in resampled]
    if missing:
        raise ValueError(
            f'"{the_id}" is missing required channel(s) {missing}. '
            f'Subject will be skipped.'
        )

    WANTED = ['ptaf', 'cflow', 'abd', 'chest', 'airflow', 'spo2', 'ecg']
    signals_df = pd.DataFrame(index=range(n_samples))
    for ch in WANTED:
        if ch in resampled:
            signals_df[ch] = resampled[ch]

    if add_CAISR:
        stage_csv = os.path.join(csv_folder, 'stage', f'{the_id}_stage.csv')
        if os.path.exists(stage_csv):
            try:
                stage_vals = pd.read_csv(stage_csv)['stage'].fillna(5).values
                stage_up = np.repeat(stage_vals, original_fs)
                if len(stage_up) >= n_samples:
                    signals_df['stage'] = stage_up[:n_samples]
                else:
                    signals_df['stage'] = np.pad(
                        stage_up, (0, n_samples - len(stage_up)), constant_values=5
                    )
            except Exception as exc:
                if verbose > 0:
                    print(f'   [WARNING] Could not load stage CSV for "{the_id}": {exc}')
        elif verbose > 0:
            print(f'   [WARNING] Stage CSV not found for "{the_id}" (expected: {stage_csv})')

        arousal_csv = os.path.join(csv_folder, 'arousal', f'{the_id}_arousal.csv')
        if os.path.exists(arousal_csv):
            try:
                ar_df = pd.read_csv(arousal_csv)
                arousal_arr = np.zeros(n_samples, dtype=np.float32)
                for _, row in ar_df.iterrows():
                    scale = original_fs / 200.0
                    s = int(row['start_idx'] * scale)
                    e = min(int(row['end_idx'] * scale), n_samples)
                    if s < n_samples and not np.isnan(row.get('arousal', np.nan)):
                        arousal_arr[s:e] = float(row['arousal'])
                signals_df['arousal'] = arousal_arr
            except Exception as exc:
                if verbose > 0:
                    print(f'   [WARNING] Could not load arousal CSV for "{the_id}": {exc}')
        elif verbose > 0:
            print(f'   [WARNING] Arousal CSV not found for "{the_id}" (expected: {arousal_csv})')
    else:
        signals_df['stage'] = signals_df.get('stage', 0)
        signals_df['arousal'] = signals_df.get('arousal', 0)

    signals_df = do_initial_preprocessing(signals_df, new_fs, original_fs)

    bad_channels = []
    for channel in ['ptaf', 'cflow']:
        if channel in signals_df.columns and signals_df[channel].std() < 1e-4:
            if verbose:
                print(f'   [{the_id}] Channel "{channel}" is flat — ignored.')
            bad_channels.append(channel)
    if bad_channels:
        signals_df = signals_df.drop(columns=bad_channels)

    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_fs, verbose)
    tag = 'signal morphology'

    if 'cpap_on' in signals_df.columns:
        br_trace_cpap, split_loc_cpap = cpapOn_select_breathing_trace(signals_df)
        if br_trace_cpap:
            br_trace, split_loc = br_trace_cpap, split_loc_cpap
            tag = 'cpap_on'
        signals_df = signals_df.drop(columns=['cpap_on'])
    elif verbose == 2:
        print(f'   [{the_id}] No "cpap_on" channel found.')

    if isinstance(split_loc, (int, float)):
        split_loc = int(split_loc * new_fs / original_fs)

    signals_df = clip_normalize_signals(signals_df, new_fs, br_trace, split_loc)

    if split_loc is None:
        if br_trace and br_trace[0] in signals_df.columns:
            signals_df['breathing_trace'] = signals_df[br_trace[0]].values
        else:
            br_trace = []
        if not br_trace or signals_df[br_trace[0]].std() < 1e-4:
            if verbose:
                print(f'   [{the_id}] No valid flow trace — using "abd+chest" fallback.')
            abd = signals_df['abd'].rolling(int(0.5 * new_fs), center=True, min_periods=1).median().fillna(0)
            chest = signals_df['chest'].rolling(int(0.5 * new_fs), center=True, min_periods=1).median().fillna(0)
            signals_df['breathing_trace'] = abd + chest
            br_trace = ['abd+chest']
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    if 'airflow' not in signals_df.columns or signals_df['airflow'].std() < 1e-4:
        signals_df['airflow'] = signals_df['breathing_trace']

    hdr = setup_header(path, new_fs, original_fs, br_trace, split_loc)
    if verbose == 2:
        print(f'   [{the_id}] Study: {hdr["test_type"]} (based on "{tag}").')

    signals_df['patient_asleep'] = np.logical_and(
        signals_df['stage'] > 0, signals_df['stage'] < 5
    )
    return signals_df, hdr


def load_breathing_signals_dispatch(
    path: str,
    csv_folder: str,
    add_CAISR: bool = True,
    verbose: int = 5,
) -> Tuple[pd.DataFrame, dict]:
    """Unified data_loader: dispatches on extension between H5 and EDF loaders."""
    if path.lower().endswith('.edf') or is_zarr_input(path):
        return _load_breathing_signals_from_edf(
            path, csv_folder, add_CAISR=add_CAISR, verbose=verbose
        )
    return load_breathing_signals_from_prepared_data(
        path, csv_folder, add_CAISR=add_CAISR, verbose=verbose
    )

def timer(tag: str) -> None:
    print(tag)
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))
    print()

def extract_run_parameters(param_csv: str) -> Tuple[bool, bool]:
    if not os.path.exists(param_csv):
        return False, False # Default values
    params = pd.read_csv(param_csv)
    multiprocess = params['multiprocess'].values[0]
    overwrite = params['overwrite'].values[0]
    return multiprocess, overwrite

def set_output_paths(input_paths: List[str], csv_folder: str, raw_folder: str, overwrite: bool) -> Tuple[List[str], List[str], List[str]]:
    # Create directories
    resp_csv_dir = os.path.join(csv_folder, 'resp')
    os.makedirs(resp_csv_dir, exist_ok=True)
    os.makedirs(raw_folder, exist_ok=True)

    IDs = [p.split(os.sep)[-1].split('.')[0] for p in input_paths]
    csv_paths = [os.path.join(resp_csv_dir, f'{ID}_resp.csv') for ID in IDs]
    raw_paths = [os.path.join(raw_folder, f'{ID}.hf5') for ID in IDs]

    input_paths, csv_paths, raw_paths = filter_already_processed_files(input_paths, csv_paths, raw_paths, overwrite)
    return input_paths, csv_paths, raw_paths

def filter_already_processed_files(input_paths, csv_paths, raw_paths, overwrite):
    total = len(input_paths)
    todo_indices = [i for i, path in enumerate(csv_paths) if not os.path.exists(path)]

    if not overwrite:
        input_paths = np.array(input_paths)[todo_indices].tolist()
        csv_paths = np.array(csv_paths)[todo_indices].tolist()
        raw_paths = np.array(raw_paths)[todo_indices].tolist()

    tag = '(overwrite) ' if overwrite else ''
    print(f'>> {total - len(todo_indices)}/{total} files already processed\n>> {len(input_paths)} to go.. {tag}\n')
    return input_paths, csv_paths, raw_paths

def filter_todo_files(input_paths: List[str], csv_paths: List[str], raw_paths: List[str], keep_indices: List[int]) -> Tuple[List[str], List[str], List[str]]:
    """
    Filters input, CSV, and raw paths based on indices of files to process.
    
    Args:
    - input_paths (List[str]): List of input file paths.
    - csv_paths (List[str]): List of CSV output file paths.
    - raw_paths (List[str]): List of raw output file paths.
    - keep_indices (List[int]): Indices of files to keep for processing.

    Returns:
    - Tuple[List[str], List[str], List[str]]: Filtered lists of input, CSV, and raw paths.
    """

    input_paths = np.array(input_paths)[keep_indices].tolist()
    csv_paths = np.array(csv_paths)[keep_indices].tolist()
    raw_paths = np.array(raw_paths)[keep_indices].tolist()

    return input_paths, csv_paths, raw_paths

def format_dataframe(data: pd.DataFrame, hdr: dict, ID: str) -> Tuple[pd.DataFrame, dict]:
    """
    Formats the input dataframe by renaming columns and validating the breathing trace.
    
    Args:
    - data (pd.DataFrame): The input data.
    - hdr (dict): Header information.
    - ID (str): The patient ID.

    Returns:
    - Tuple[pd.DataFrame, dict]: The formatted dataframe and updated header.
    
    Raises:
    - ValueError: If no valid/usable breathing trace can be found.
    """

    # FIX: The data loader has already performed robust breathing trace selection.
    # Instead of re-implementing the logic, we validate its output.
    # This prevents flat signals from being passed to the core algorithm.
    if 'breathing_trace' not in data.columns or np.nanstd(data['breathing_trace']) < 1e-6:
        # Provide a clear error message indicating why the file cannot be processed.
        raise ValueError(f"No usable breathing trace found for {ID}. All available channels (ptaf, cflow, abd, chest) might be flat or have insufficient signal.")

    # Rename specific columns for consistency
    old_names = ['resp', 'stage', 'arousal', 'abd', 'chest', 'spo2']
    new_names = ['Apnea', 'Stage', 'EEG_arousals', 'ABD', 'CHEST', 'SpO2']

    for old_name, new_name in zip(old_names, new_names):
        if old_name in data.columns:
            data = data.rename(columns={old_name: new_name})

    # Add a combined ventilation column
    data['Ventilation_combined'] = data['breathing_trace']
    hdr['patient_ID'] = ID

    check_cols = ['ABD', 'CHEST', 'SpO2', 'Stage', 'EEG_arousals', 'breathing_trace', 'Ventilation_combined']
    if 'Apnea' in data.columns:
        check_cols.insert(0, 'Apnea')
    
    for col in check_cols:
        assert col in data.columns, f'ERROR: {col} not in dataframe'

    return data, hdr
    
# save functions
def save_raw_output(data: pd.DataFrame, hdr: dict, out_file: str) -> None:
    """
    Saves processed data, including breathing, effort belts, SpO2, and algorithmic predictions to an HDF5 file.

    Args:
    - data (pd.DataFrame): DataFrame containing the input signals and predictions.
    - hdr (dict): Dictionary containing metadata or header information.
    - out_file (str): The file path to save the output data.
    """
    # Create an empty DataFrame to store data
    df = pd.DataFrame()

    # Fill the DataFrame with relevant data columns
    df['breathing'] = data['breathing_trace']
    if 'ptaf' in data.columns:
        df['ptaf'] = data['ptaf']
    if 'cflow' in data.columns:
        df['cflow'] = data['cflow']
    df['abd'] = data['ABD']
    df['chest'] = data['CHEST']
    df['spo2'] = data['SpO2']
    df['sleep_stages'] = data['Stage']
    df['y_CAISR'] = data['algo_apneas']
    df['EEG_arousals'] = data['EEG_arousals']

    # Add 'airflow' if it exists in the DataFrame
    if 'airflow' in data.columns:
        df['airflow'] = data['airflow']
    
    # Add 'Apnea' (original labels) if it exists in the DataFrame
    if 'Apnea' in data.columns:
        df['y_original'] = data['Apnea']

    # Add additional header information to the DataFrame (except patient ID)
    for key, value in hdr.items():
        if key != 'patient_ID':
            df[key] = value

    # Save data to HDF5 using the custom writer function
    from Data_writers import write_to_hdf5_file
    write_to_hdf5_file(df, out_file, overwrite=True)

def export_to_csv(pred: np.ndarray, csv_path: str, verbose: int = 5, Fs: int = 10, originalFs: int = 200) -> None:
    """
    Exports the predicted respiration events to a CSV file, resampled to 1Hz.

    Args:
    - pred (np.ndarray): Array of predicted respiration events.
    - csv_path (str): Path to save the CSV file.
    - verbose (int): Verbosity level for logging.
    - Fs (int): The current sampling frequency in Hz.
    - originalFs (int): The original sampling frequency of the signal.
    """
    # Create an empty DataFrame for the predictions
    data = pd.DataFrame(columns=['start_idx', 'end_idx', 'resp'])

    # Resample predictions to 1Hz (from Fs) and insert into the DataFrame
    data['resp'] = pred[::Fs]

    # Calculate start and end indices based on the original sampling frequency
    factor = originalFs // 1  # Resampling factor for converting to 1Hz
    ind0 = np.arange(0, len(data)) * factor
    ind1 = np.concatenate([ind0[1:], [ind0[-1] + factor]])

    # Add the calculated indices to the DataFrame
    data['start_idx'] = ind0
    data['end_idx'] = ind1

    # Save the DataFrame to a CSV file
    data.to_csv(csv_path, index=False, mode='w+')

    # Verify the saved CSV if verbosity is enabled
    if verbose > 0:
        print(f'Saved to "{csv_path}"')

# RUN ALGO
@ray.remote
def set_multiprocess_run(p: int, path: str, csv_out: str, raw_out: str, csv_folder: str, data_loader, total_num: int, verbose: int=5) -> bool:
    """
    Function for running the algorithm in parallel using Ray.
    
    Args:
    - p (int): Current index for progress tracking.
    - path (str): Input file path.
    - csv_out (str): CSV output file path.
    - raw_out (str): HDF5 output file path.
    - csv_folder (str): Folder path to store the CSV.
    - data_loader: Function to load data.
    - total_num (int): Total number of files to process.
    
    Returns:
    - bool: Whether the processing was successful.
    """

    try:
        success = set_individual_run(p, path, csv_out, raw_out, csv_folder, data_loader, total_num, verbose=verbose)
        return success
    except Exception:
        return False

def set_individual_run(p: int, path: str, csv_out: str, raw_out: str, csv_folder: str, data_loader, total_num: int, verbose: int) -> bool:
    """
    Function for running the algorithm on a single file.
    
    Args:
    - p (int): Current index for progress tracking.
    - path (str): Input file path.
    - csv_out (str): CSV output file path.
    - raw_out (str): HDF5 output file path.
    - csv_folder (str): Folder path to store the CSV.
    - data_loader: Function to load data.
    - total_num (int): Total number of files to process.
    - verbose (int): Verbosity level (0: silent, >0: detailed logging).
    
    Returns:
    - bool: Whether the processing was successful.
    """

    # Print progress
    patient_ID = path.split('/')[-1].split('.')[0]
    ratio = f'# {p + 1}/{total_num}'
    tag = patient_ID if len(patient_ID)<21 else patient_ID[:20]+'..'
    print(f'(# {p + 1}/{total_num}) Processing "{tag}"')

    try:
        # Load data (note: csv_folder used for side-loading other features like stage/arousal if needed)
        data, hdr = data_loader(path, csv_folder, add_CAISR=True, verbose=verbose)
        data, hdr = format_dataframe(data, hdr, patient_ID)
    except Exception as error:
        if verbose > 1: print(f'ERROR loading ({patient_ID}): {error}')
        return False

    try:
        run_algorithm_pipeline(data, hdr, csv_out, raw_out, verbose=verbose)
    except Exception as error:
        if verbose > 1: print(f'ERROR algorithm ({patient_ID}): {error}')
        return False

    return True
    
def run_algorithm_pipeline(data: pd.DataFrame, hdr: dict, csv_out: str, raw_out: str, save_raw: bool = False, verbose: bool = True) -> None:
    """
    The main pipeline to run the algorithm on data, save results as CSV and HDF5.
    
    Args:
    - data (pd.DataFrame): Input data for the algorithm.
    - hdr (dict): Metadata or header information.
    - csv_out (str): Path to save the CSV file.
    - raw_out (str): Path to save the HDF5 file.
    - save_raw (bool): Whether to save raw output to HDF5.
    - verbose (bool): Whether to print logs.
    """

    # Compute desaturation drops and merge events
    data = compute_saturation_drops(data, hdr, sat_drop=3)
    data = merge_connecting_saturation_drops(data, hdr)

    # Identify flow reductions and match potential hypopneas
    # import pdb; pdb.set_trace()
    data = find_flow_reductions(data, hdr)
    data = match_saturation_and_ventilation_drops(data, hdr)
    data = match_EEG_with_ventilation_drops(data, hdr)

    # Detect RERAs (Respiratory Effort-Related Arousals)
    data = RERA_detection(data, hdr)

    # Combine desaturation-hypopneas with EEG-hypopneas
    data['algo_hypopneas_three'] = (data['accepted_saturation_hypopneas'] + data['accepted_EEG_hypopneas']) * 4
    data.loc[data['algo_hypopneas_three'] == 8, 'algo_hypopneas_three'] = 4

    # Rank events based on AASM hierarchy
    data = do_apnea_multiclassification(data, hdr)

    # Post-processing: merge small detections, split large events
    data = post_processing(data, hdr)

    # Save output data to CSV and HDF5 if needed
    export_to_csv(data['algo_apneas'].values, csv_out, verbose=verbose)
    if save_raw:
        raw_folder = './resp/raw_outputs/'
        os.makedirs(raw_folder, exist_ok=True)
        save_raw_output(data, hdr, raw_out)

def CAISR_resp(in_paths: List[str], csv_folder: str, raw_output_dir: str, multiprocess: bool = True, overwrite: bool = False, verbose: int = 5) -> None:
    timer('* Starting "caisr_resp" (created by Thijs Nassi, PhD)')
    data_loader = load_breathing_signals_dispatch
    
    in_paths, csv_paths, raw_paths = set_output_paths(in_paths, csv_folder, raw_output_dir, overwrite)
    finished = [False] * len(in_paths)
    total = len(in_paths)
    
    if multiprocess:
        try:
            if verbose > 0: print(f'Parallel processing {total} files..')
            # Initialize Ray (adjust num_cpus as needed)
            ray.init(num_cpus=min(20, os.cpu_count()), logging_level=logging.CRITICAL)
            
            futures = [set_multiprocess_run.remote(p, pi, pc, pr, csv_folder, data_loader, total, verbose)
                       for p, (pi, pc, pr) in enumerate(zip(in_paths, csv_paths, raw_paths))]
            
            finished = ray.get(futures)
            ray.shutdown()
            
            if verbose > 0: print(f'  Successful for {sum(finished)}/{total} recordings\n')
        except Exception as e:
            if verbose > 0: print(f'Parallel processing failed or interrupted: {e}')
            if ray.is_initialized(): ray.shutdown()

    # Retry failed or process sequentially if MP disabled
    todo_indices = np.where([not f for f in finished])[0]
    if len(todo_indices) > 0:
        print("Processing remaining files sequentially...")
        for idx in todo_indices:
            finished[idx] = set_individual_run(idx, in_paths[idx], csv_paths[idx], raw_paths[idx], csv_folder, data_loader, total, verbose)

    print(f'{sum(finished)}/{total} recordings were successfully processed.')
    timer('* Finishing "caisr_resp"')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CAISR respiratory event detection.")
    parser.add_argument("--input_data_dir", type=str, required=True, help="Folder containing prepared .h5 and/or raw .edf data files.")
    parser.add_argument("--output_csv_dir", type=str, required=True, help="Folder where CAISR's output CSV files will be stored.")
    parser.add_argument("--param_dir", type=str, required=True, help="Folder containing the run parameters file.")
    args = parser.parse_args()
        
    input_files = sorted(
        glob.glob(os.path.join(args.input_data_dir, '*.h5')) +
        glob.glob(os.path.join(args.input_data_dir, '*.edf')) +
        glob.glob(os.path.join(args.input_data_dir, '*.zarr'))
    )
    param_file = os.path.join(args.param_dir, 'resp.csv')
    
    if not os.path.exists(param_file):
        os.makedirs(args.param_dir, exist_ok=True)
        pd.DataFrame({'overwrite': [False], 'multiprocess': [True]}).to_csv(param_file, index=False)
        
    multiprocess, overwrite = extract_run_parameters(param_file)
    
    # Define raw output dir based on CSV output dir
    raw_out = os.path.join(args.output_csv_dir, 'resp', 'raw_outputs')
    
    CAISR_resp(input_files, args.output_csv_dir, raw_out, multiprocess=multiprocess, overwrite=overwrite, verbose=5)
    