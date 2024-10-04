import os, sys
import numpy as np
import pandas as pd
from typing import List

# Set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import custom event functions
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Event_array_modifiers import find_events

def bad_signal_check(data: pd.DataFrame, Fs: int, flat_std: float = 0.05) -> pd.DataFrame:
    """
    Identifies bad signal segments based on low standard deviation in ventilation
    and flat SpO2 signals.

    Args:
        data (pd.DataFrame): Input DataFrame containing breathing and SpO2 signals.
        Fs (int): Sampling frequency.
        flat_std (float, optional): Threshold for flat standard deviation in the breathing signal.

    Returns:
        pd.DataFrame: DataFrame with additional columns indicating bad signal, flat signal, and low standard deviation.
    """
    
    # Check for flat breathing signal
    breathing_std = data['Ventilation_combined'].rolling(Fs, center=True).std().fillna(-1)
    data['low_std'] = np.logical_and(breathing_std < 0.01, breathing_std > -0.01)

    # Check for flat SpO2 signal
    spo2 = data['SpO2'] if 'smooth_saturation' not in data.columns else data['smooth_saturation']
    data['flat_spo2'] = np.logical_or(spo2 <= 60, np.isnan(spo2))

    # Combined bad signal detection
    data['bad_sig'] = np.logical_and(data['low_std'], data['flat_spo2'])
    data['bad_signal'] = data['bad_sig'].rolling(3 * Fs, center=True).min().fillna(1)

    # Mark segments with flat Ventilation_combined signal
    for st, end in find_events(data['Ventilation_combined'].rolling(2).std() == 0):
        if end - st > 0.5 * Fs:
            data.loc[st:end, 'bad_signal'] = 1

    # Add flat signal detection based on standard deviation
    data['low_std'] = np.logical_and(breathing_std < flat_std, breathing_std > -flat_std)
    flat_signal = data['low_std'].rolling(2 * Fs, center=True).max().fillna(1)

    # Mark flat signals lasting more than 5 minutes
    data['flat_signal'] = 0
    for st, end in find_events(flat_signal > 0):
        if (end - st) > 5 * 60 * Fs:
            data.loc[st:end, 'flat_signal'] = 1

    # Remove intermediate columns used for calculation
    data = data.drop(columns=['bad_sig', 'flat_spo2', 'low_std'])

    return data

def remove_error_events(data: pd.DataFrame, trace: str, columns: List[str]) -> pd.DataFrame:
    """
    Removes error events from specified columns based on flat breathing segments and unavailable SpO2 data.

    Args:
        data (pd.DataFrame): Input DataFrame containing event and signal data.
        trace (str): Trace name to check flat and bad signal conditions.
        columns (list[str]): List of column names to clean error events from.

    Returns:
        pd.DataFrame: DataFrame with error events removed from specified columns.
    """

    for col in columns:
        for st, end in find_events(data[col].values > 0):
            # Remove events that occur in flat breathing segments
            flat_cols = [f'flat_signal_{trace}', f'bad_signal_{trace}']
            if np.any(data.loc[st:end, flat_cols] > 0):
                data.loc[st:end, col] = 0

            # Remove events where more than 75% of the segment has unavailable SpO2 data
            if sum(data.loc[st:end, 'SpO2'].isna()) > 0.75 * (end - st):
                data.loc[st:end, col] = 0

    return data
