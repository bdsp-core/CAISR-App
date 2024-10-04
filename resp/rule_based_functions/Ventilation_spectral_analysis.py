import os
import sys
import numpy as np
import pandas as pd
from typing import List, Tuple

# Set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Import utils
sys.path.insert(0, os.path.join(RB_folder, 'utils_functions'))
from Event_array_modifiers import find_events

# Spectral assessment functions
def spectral_assess_effort_limitation(data: pd.DataFrame, start: int, end: int, fs: float, tag: str) -> List[bool]:
    """
    Assesses the effort limitation based on spectral analysis.

    Args:
        data (pd.DataFrame): Data containing the signal.
        start (int): Start index of the segment.
        end (int): End index of the segment.
        fs (float): Sampling frequency of the signal.
        tag (str): Signal type (e.g., 'airflow', 'breathing_trace').

    Returns:
        List[bool]: List of boolean checks based on the spectral peaks.
    """
    
    mid = int(np.mean([start, end]))
    regions = [np.arange(start, end), np.arange(start, mid)]
    peaks: List[float] = []

    for region in regions:
        sig = data.loc[region, tag].values - np.median(data.loc[region, tag].values)

        # FFT and power spectrum calculation
        ps = np.abs(np.fft.fft(sig))**2
        freqs = np.fft.fftfreq(sig.size, d=1/fs)
        
        # Frequency range filter
        freq_filter = (0.01 < freqs) & (freqs < 10)
        filtered_freqs = freqs[freq_filter]
        filtered_power = ps[freq_filter]

        # Power peak detection
        peaks.append(filtered_freqs[np.argmax(filtered_power)])

    if tag == 'airflow':
        checks = [0.5 < p or p < 0.12 for p in peaks]
    elif tag == 'breathing_trace':
        checks = [0.5 < p or p < 0.08 for p in peaks]
    else:
        checks = [(0.5 < p or p < 0.05) if i == 0 else (0.8 < p or p < 0.05) for i, p in enumerate(peaks)]

    return checks

def spectral_assess_thermistor_hypopnea(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Assesses thermistor hypopneas using spectral analysis and converts non-breathing thermistor hypopneas into apneas.

    Args:
        data (pd.DataFrame): Data containing the signal and event information.
        hdr (dict): Header information including the sampling frequency.

    Returns:
        pd.DataFrame: Updated data with converted non-breathing thermistor hypopneas.
    """

    # Array to mark non-breathing thermistor hypopneas
    non_breathing_therm_hyps = np.zeros(len(data))

    # Assess all thermistor hypopneas for breathing frequencies
    for start, end in find_events(data.flow_reductions_br == 1):
        if not np.any(data.loc[start:end, 'flow_reductions'] == 2):
            continue
        
        not_breathing = spectral_assess_effort_limitation(data, start, end, hdr['newFs'], 'airflow')
        if not_breathing[0]:
            non_breathing_therm_hyps[start:end] = 1

    # Convert non-breathing thermistor hypopneas into apneas
    for start, end in find_events(data['flow_reductions'] == 2):
        if np.any(non_breathing_therm_hyps[start:end] == 1):
            data.loc[start:end, 'flow_reductions'] = 1

    return data
