import os, sys, h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional

# import utils
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Preprocessing import *
from load_caisr_annotation_functions import load_annotation_in_original_fs

## Main CAISR prepared data loader ##
def load_prepared_data(path: str, signals: List[str] = [], fs: int = 200) -> pd.DataFrame:
    """
    Load prepared CAISR data from an .h5 file.

    Args:
        path (str): Path to the .h5 file containing the data.
        signals (list[str], optional): List of signal names to load. If empty, all signals are loaded. Defaults to [].
        fs (int, optional): Sampling frequency. Defaults to 200.

    Returns:
        pd.DataFrame: DataFrame containing the loaded signals.

    Raises:
        AssertionError: If there is a mismatch in the length of the signals or if not all requested signals are found.
    """
    # Read in signals from the .h5 file
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        vals = None
        cols = []

        for key in keys:
            subkeys = f[key].keys()
            for subkey in subkeys:
                # Filter based on requested signals
                if signals and subkey not in signals:
                    continue

                val = np.squeeze(f[key][subkey][:])
                
                # Initialize vals array if it's the first signal
                if vals is None:
                    vals = val
                else:
                    diff = np.abs(len(val) - max(vals.shape))
                    assert diff < fs / 2, f'Incorrect column lengths in prepared data.'
                    vals = np.vstack([vals, val[:max(vals.shape)]])
                
                cols.append(subkey)

    # Create DataFrame from loaded data
    data = pd.DataFrame(vals.T, columns=cols)

    # Ensure all requested signals are loaded
    if signals:
        assert len(signals) == len(data.columns), f'Not all requested signals found for recording {path}.'

    return data

def load_CAISR_output(path: str, signals_df: pd.DataFrame, csv_folder: str, fs: int, verbose: bool = False) -> pd.DataFrame:
    """
    Load CAISR output (e.g., stage and arousal labels) and append them to the signals DataFrame.

    Args:
        path (str): Path to the original .h5 file.
        signals_df (pd.DataFrame): DataFrame containing the signals data.
        csv_folder (str): Folder containing the CAISR output .csv files.
        fs (int): Sampling frequency.
        verbose (bool, optional): If True, prints additional information. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with the CAISR labels (stage, arousal) added.

    Raises:
        Exception: If the corresponding CAISR output .csv file is not found.
    """
    # Run over all tasks (e.g., stage, arousal)
    for task in ['stage', 'arousal']:
        # Setup corresponding CSV path for CAISR output
        tag = path.split('/')[-1].replace('.h5', f'_{task}.csv')
        csv_path = os.path.join(csv_folder, task, tag)

        # Check if the CAISR output file exists
        if not os.path.exists(csv_path):
            raise Exception(f'No matching "caisr_{task}" output found.')

        # Load CAISR labels and add to signals DataFrame
        labels = load_annotation_in_original_fs(task, path, csv_path, fs_original=fs, verbose=verbose)
        signals_df[f'{task}_CAISR'] = labels[task]

    return signals_df


## Breathing Trace Selection based on CPAP On/Off ##
def cpapOn_select_breathing_trace(signals_df: pd.DataFrame) -> Tuple[List[str], Optional[int]]:
    """
    Select the breathing trace based on whether CPAP is on or off during the sleep study.

    Args:
        signals_df (pd.DataFrame): DataFrame containing the signal data, including the 'cpap_on' column.

    Returns:
        Tuple[List[str], Optional[int]]: Selected breathing trace and the split location for CPAP on/off.
    
    Raises:
        Exception: If 'cpap_on' column is not present in the signals DataFrame.
    """

    # Determine split location and selection
    split_loc = np.where(signals_df.cpap_on == 1)[0]
    
    if len(split_loc) == 0:
        # If no CPAP on is found
        split_loc = None
        selection = ['ptaf']
    else:
        split_loc = split_loc[0]
        first_sleep = np.where(np.logical_and(signals_df.stage > 0, signals_df.stage < 5))[0]
        assert len(first_sleep) > 0, 'No sleep found for this patient!'

        # Decide trace based on the location of CPAP on and first sleep
        if split_loc < first_sleep[0]:
            split_loc = None
            selection = ['cflow']
        else:
            selection = ['ptaf', 'cflow']

    return selection, split_loc

## Breathing Trace Selection based on Morphology ##
def morphology_select_breathing_trace(signals: pd.DataFrame, Fs: int, verbose: int) -> Tuple[List[str], Optional[int]]:
    """
    Select the breathing trace based on signal morphology and noise analysis.
    
    Args:
        signals (pd.DataFrame): DataFrame containing signal data.
        Fs (int): Sampling frequency.
        verbose (int): Verbosity level for logging.

    Returns:
        Tuple[List[str], Optional[int]]: Selected breathing trace and potential split location.
    """

    try:
        st = np.where(signals.stage < 5)[0][0]
        end = len(signals) - np.where(np.flip(signals.stage.values) < 5)[0][0]
    except IndexError:
        if verbose == 2:
            print('No sleep detected.')
        return ['ptaf'], None

    # Standard deviation check for breathing traces
    br_traces = ['ptaf', 'cflow']
    stds = np.array([np.std(signals.ptaf), np.std(signals.cflow)])
    if sum(stds == 0) == 1:
        return [br_traces[np.where(stds > 0)[0][0]]], None

    # Noise check
    noise = []
    for col in br_traces:
        is_noise, _, _ = check_for_noise(signals.loc[st:end, col], Fs)
        noise.append(is_noise)
    
    if sum(noise) == 1:
        return [br_traces[np.where(np.array(noise) == False)[0][0]]], None

    # Split-night check for potential split
    window = 30 * 60 * Fs
    ptaf_std = signals['ptaf'].rolling(int(60 * Fs), center=True).std()
    cflow_std = signals['cflow'].rolling(int(60 * Fs), center=True).std()

    ptaf_on = (ptaf_std / np.max(ptaf_std)).rolling(window, center=True).max().values > .1
    cflow_on = (cflow_std / np.max(cflow_std)).rolling(window, center=True).max().values > .1

    loc1 = st + np.where(ptaf_on[st:end] == 0)[0] - window / 2
    loc2 = end - np.where(np.flip(cflow_on[st:end]) == 0)[0] + window / 2

    # Determine split location if applicable
    if len(loc1) == 0 and len(loc2) == 0:
        return ['ptaf'], None
    elif len(loc1) > 0 and len(loc2) == 0:
        loc1, loc2 = loc1[0], loc1[0]
    elif len(loc2) > 0 and len(loc1) == 0:
        loc1, loc2 = loc2[0], loc2[0]
    else:
        loc1, loc2 = loc1[0], loc2[0]

    # Split location must be within 30 minutes in both traces
    if np.abs(loc2 - loc1) < Fs * 60 * 30:
        split_loc = int(np.mean([loc1, loc2]))
    else:
        return [], None

    # Unlikely split-night scenarios correction
    if split_loc < 0.1 * len(signals):
        return ['cflow'], None
    elif split_loc > 0.9 * len(signals):
        return ['ptaf'], None

    return ['ptaf', 'cflow'], split_loc

## Noise Check for Breathing Signal ##
def check_for_noise(sig: np.ndarray, Fs: int) -> Tuple[bool, float, float]:
    """
    Check whether the signal is noisy using a frequency-based method.
    
    Args:
        sig (np.ndarray): Signal to check.
        Fs (int): Sampling frequency.

    Returns:
        Tuple[bool, float, float]: 
            - is_noise (bool): Whether the signal is classified as noise.
            - peak (float): Peak frequency.
            - percentage (float): Percentage of power over 75% of the peak.
    """

    # Remove median to normalize
    sig = sig - np.median(sig)

    # Frequency analysis
    ps = np.abs(np.fft.fft(sig))**2
    freqs = np.fft.fftfreq(sig.size, 1 / Fs)
    idx = np.argsort(freqs)

    min_freq, max_freq = 0.01, 0.5
    locs = (freqs[idx] > min_freq) & (freqs[idx] < max_freq)

    f_range = freqs[idx][locs]
    power = ps[idx][locs]

    # Power peak
    peak = f_range[np.argmax(power)]
    
    # Compute percentage of power over 75% of the peak
    percentage = len(np.where(power > 0.75 * np.max(power))[0]) / len(power) * 100

    # Determine if the signal is classified as noise
    is_noise = peak < 0.1

    return is_noise, peak, percentage

## data formating ##
def setup_header(path: str, new_Fs: int, original_Fs: int, br_trace: List[str], split_loc: Optional[int]) -> Dict[str, object]:
    """
    Set up the header metadata for a sleep study recording.

    Args:
        path (str): File path of the recording.
        new_Fs (int): New sampling frequency after resampling.
        original_Fs (int): Original sampling frequency.
        br_trace (List[str]): Selected breathing trace channels.
        split_loc (Optional[int]): Split location for split-night recordings or None.

    Returns:
        Dict[str, object]: Header dictionary with metadata about the sleep study recording.
    """
    # Initialize the header dictionary
    hdr = {
        'newFs': new_Fs,
        'Fs': original_Fs,
        'test_type': 'diagnostic',  # Default value for test_type
        'rec_type': 'CAISR',        # Recording type
        'cpap_start': split_loc,    # Location where CPAP starts (if applicable)
        'patient_tag': path.split('/')[-1].split('.')[0]  # Patient ID based on the file name
    }

    # Determine the test type based on the breathing trace selection
    if len(br_trace) == 0:
        hdr['test_type'] = 'unknown'
    elif len(br_trace) == 2:
        hdr['test_type'] = 'split-night'
    elif br_trace[0] == 'cflow':
        hdr['test_type'] = 'titration'

    return hdr



######################
## main data-loader ##
######################

def load_breathing_signals_from_prepared_data(path: str, csv_folder: str, original_Fs: int = 200, new_Fs: int = 10, 
                                                channels: Optional[List[str]] = [], add_CAISR: bool = True, verbose: bool = True) -> Tuple[pd.DataFrame, dict]:
    """
    Load and preprocess breathing signals from prepared data.

    Args:
        path (str): File path to the prepared data.
        csv_folder (str): Folder path for CAISR output CSV files.
        original_Fs (int): Original sampling frequency of the data (default 200 Hz).
        new_Fs (int): New sampling frequency for resampling (default 10 Hz).
        channels (Optional[List[str]]): List of channels to load (optional).
        add_CAISR (bool): Flag to include CAISR outputs (default True).
        verbose (bool): Flag for verbosity in logging and printing (default True).

    Returns:
        Tuple[pd.DataFrame, dict]: Processed signals DataFrame and header information.
    """

    # Load signals from the prepared data file
    signals_df = load_prepared_data(path)

    # Add available CAISR outputs if requested
    if add_CAISR:
        try:
            signals_df = load_CAISR_output(path, signals_df, csv_folder, original_Fs)
        except Exception as error:
            raise Exception(f"Error loading CAISR output: {error}")
        
        # Set default label arrays for stage and arousal
        signals_df['stage'] = signals_df['stage_CAISR']
        signals_df['arousal'] = signals_df['arousal_CAISR']
    else:
        signals_df['stage'] = signals_df['stage_expert_0']
        signals_df['arousal'] = signals_df['arousal_expert_0']

    # Remove unnecessary channels
    if 'resp-h3_expert_0' in signals_df.columns:
        signals_df['resp'] = signals_df['resp-h3_expert_0']
    elif 'resp-h3_converted_1' in signals_df.columns:
        signals_df['resp'] = signals_df['resp-h3_converted_1']
    for col in signals_df.columns:
        if col not in ['arousal', 'stage', 'resp', 'ptaf', 'airflow', 'cflow', 'spo2', 'abd', 'chest', 'cpap_on']:
            signals_df = signals_df.drop(columns=col)

    # Select breathing trace based on morphology
    br_trace, split_loc = morphology_select_breathing_trace(signals_df, new_Fs, verbose)
    tag = 'signal morphology'

    # Apply initial preprocessing
    signals_df = do_initial_preprocessing(signals_df, new_Fs, original_Fs)

    # Correct the split location if necessary
    if isinstance(split_loc, int):
        split_loc = int(split_loc * new_Fs / original_Fs)

    # Set the breathing trace based on CPAP status
    if 'cpap_on' in signals_df.columns:
        br_trace, split_loc = cpapOn_select_breathing_trace(signals_df)
        tag = 'cpap_on'

        # Drop the 'cpap_on' column if present
        signals_df = signals_df.drop(columns=['cpap_on'])
    else:
        if verbose == 2:
            print('No "cpap_on" channel was found.')

    # Clip and normalize the signals
    signals_df = clip_normalize_signals(signals_df, new_Fs, br_trace, split_loc)

    # Set breathing trace logic if split location is None
    if split_loc is None:
        if np.all(signals_df[['ptaf', 'cflow']] == 0) or len(br_trace) == 0:
            abd = signals_df['abd'].rolling(int(0.5 * new_Fs), center=True).median().fillna(0)
            chest = signals_df['chest'].rolling(int(0.5 * new_Fs), center=True).median().fillna(0)
            signals_df['breathing_trace'] = abd + chest
            br_trace, split_loc = ['ptaf'], None
            if verbose == 2:
                print('Abd + Chest is used as breathing trace (replacing ptaf/cflow).')
        else:
            signals_df['breathing_trace'] = signals_df[br_trace].values
    else:
        signals_df['breathing_trace'] = np.nan
        signals_df.loc[:split_loc, 'breathing_trace'] = signals_df.loc[:split_loc, br_trace[0]]
        signals_df.loc[split_loc:, 'breathing_trace'] = signals_df.loc[split_loc:, br_trace[1]]

    # Set up header and prepare the DataFrame
    hdr = setup_header(path, new_Fs, original_Fs, br_trace, split_loc)
    if verbose == 2:
        print('Study: ' + hdr['test_type'] + f' (based on "{tag}").')

    # Define 'patient_asleep' status
    signals_df['patient_asleep'] = np.logical_and(signals_df.stage > 0, signals_df.stage < 5)

    return signals_df, hdr

