import os, mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from sklearn.preprocessing import RobustScaler
from scipy.signal import resample_poly
from mne.filter import filter_data, notch_filter


# Preprocess signals from original sampling rate to new sampling rate (e.g., 200Hz --> 10Hz)
def do_initial_preprocessing(signals: pd.DataFrame, new_Fs: int, original_Fs: int, 
                             extreme_resp_filtering: bool = False) -> pd.DataFrame:
    """
    Applies notch filtering, bandpass filtering, and resampling to the input signals.

    Args:
        signals (pd.DataFrame): Input signal data with various channels.
        new_Fs (int): New sampling frequency to resample to.
        original_Fs (int): Original sampling frequency of the input signals.
        extreme_resp_filtering (bool): Whether to apply extreme respiratory filtering (default: False).

    Returns:
        pd.DataFrame: Preprocessed signals with applied filters and resampling.
    """
    
    # Set frequency parameters for different types of filtering
    notch_freq_us = 60.0  # [Hz] for US power line
    bandpass_freq_eeg = [0.1, 20]  # [Hz]
    
    # Respiratory filtering settings based on the flag
    bandpass_freq_breathing = [0.1, 2] if extreme_resp_filtering else [0., 10]
    bandpass_freq_ecg = [0.3, None]  # [Hz] for ECG
    
    # Create a new DataFrame for processed signals
    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        image = signals[sig].values
        not_zero = not np.all(image == 0)

        # Apply notch filter to remove powerline noise
        if not_zero and sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 
                                'e1-m2', 'chin1-chin2', 'abd', 'chest', 'airflow', 
                                'ptaf', 'cflow', 'breathing_trace', 'ecg']:
            if original_Fs >= 2 * notch_freq_us:
                image = notch_filter(image.astype(float), original_Fs, notch_freq_us, verbose=False)

        # Apply bandpass filters based on signal type (EEG, Breathing, ECG)
        if not_zero and sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
            image = filter_data(image, original_Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        elif not_zero and sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            image = filter_data(image, original_Fs, bandpass_freq_breathing[0], bandpass_freq_breathing[1], verbose=False)
        elif not_zero and sig == 'ecg':
            image = filter_data(image, original_Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # Resample the data if the sampling frequencies are different
        if new_Fs != original_Fs:
            if not_zero and sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 
                                    'chin1-chin2', 'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = np.repeat(image, new_Fs)[::original_Fs]

        # Insert processed signal into new DataFrame
        new_df[sig] = image.astype(np.float32)
    
    return new_df


# Clip and normalize signals for further analysis
def clip_normalize_signals(signals: pd.DataFrame, sample_rate: int, br_trace: List[str], 
                           split_loc: int = None, min_max_times_global_iqr: int = 20) -> pd.DataFrame:
    """
    Clips and normalizes EEG, ECG, and breathing signals to remove outliers and standardize data.

    Args:
        signals (pd.DataFrame): Input signal data.
        sample_rate (int): Sample rate of the signals.
        br_trace (list[str]): List containing the respiratory trace channels.
        split_loc (int): Index location to split recordings for split-night studies (default: None).
        min_max_times_global_iqr (int): Multiplier for the interquartile range (default: 20).

    Returns:
        pd.DataFrame: Clipped and normalized signals.
    """
    
    skip_cols = ['stage', 'arousal', 'resp', 'spo2_desat', 'cpap_pressure', 'cpap_on', 
                 'spo2_artifact', 'effort_artifact', 'start', 'end']

    for chan in signals.columns:
        if any(skip in chan for skip in skip_cols) or np.all(signals[chan] == 0):
            continue

        signal = signals[chan].values

        # Special clipping for SpO2 signal
        if chan == 'spo2':
            signals[chan] = np.clip(signal.astype(float).round(), 60, 100)
            continue

        # Handle EEG and ECG traces
        if chan in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'ecg']:
            iqr = np.subtract(*np.percentile(signal, [75, 25]))  # Calculate IQR
            threshold = iqr * min_max_times_global_iqr
            signal_clipped = np.clip(signal, -threshold, threshold)

            # Normalize the channel
            transformer = RobustScaler().fit(np.atleast_2d(signal_clipped).T)
            signal_normalized = np.squeeze(transformer.transform(np.atleast_2d(signal_clipped).T))

        # Handle respiratory traces
        elif chan in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            region = np.arange(len(signal))

            if split_loc is not None:
                replacement_signal = np.empty(len(signals)) * np.nan
                if chan in [br_trace[0], 'airflow']:
                    region = region[:split_loc]
                elif chan == br_trace[1]:
                    region = region[split_loc:]
                replacement_signal[region] = signal[region]
                signal = replacement_signal
            
            if np.all(signal[region] == 0):
                continue

            # Z-score Normalize
            signal_clipped = np.clip(signal, np.nanpercentile(signal[region], 5), np.nanpercentile(signal[region], 95))
            if np.all(signal_clipped[region] == 0):
                continue
            
            signal_normalized = (signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped)
            
            # Clip extreme values
            clp = 0.01 if chan in ['abd', 'chest'] else 0.001
            factor = 10 if chan in ['abd', 'chest'] else 20
            quan = 0.2
            thresh = factor * np.mean([np.abs(np.nanquantile(signal_normalized[region], quan)), 
                                       np.abs(np.nanquantile(signal_normalized[region], 1 - quan))])
            signal_normalized = np.clip(signal_normalized, -thresh, thresh)

        # Replace original signal with normalized data
        signals[chan] = signal_normalized

    return signals
