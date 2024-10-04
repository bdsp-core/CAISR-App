import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple

# set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import utils
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Event_array_modifiers import find_events

# SpO2 Analysis functions:
def compute_saturation_drops(data: pd.DataFrame, hdr: dict, sat_drop: int, two_drops: bool = True) -> pd.DataFrame:
    """
    Computes the saturation drops within a dataset.

    Args:
    - data: DataFrame containing the SpO2 data.
    - hdr: Dictionary containing metadata, including 'newFs' for sampling rate.
    - sat_drop: Threshold value for the drop in saturation.
    - two_drops: Boolean to determine whether to add a 2% desaturation check.

    Returns:
    - Updated DataFrame with saturation drop information.
    """
    Fs = hdr['newFs']
    ranges = np.array([5, 10, 20, 30, 40, 50]) * Fs
    data['saturation_drop'] = 0

    # Compute smooth saturation
    smooth_saturation, error_array = compute_smooth_saturation(data['SpO2'].values, Fs)
    data['smooth_saturation'] = smooth_saturation
    data['error_saturation'] = error_array

    # Find saturation drops using various windows in sequence
    for ran in ranges:
        data = find_desats_within_range(data, sat_drop, ran, Fs, 'saturation_drop')

    # Add 2% desats, but below 3% <baseline> (default 2min)
    if sat_drop == 3 and two_drops:
        base_win = 60 * Fs
        data = add_quick_two_desats(data, base_win, Fs)

    # Number the desaturations
    data['saturation_drop_count'] = 0
    for i, (st, end) in enumerate(find_events(data['saturation_drop'] > 0)):
        data.loc[st:end, 'saturation_drop_count'] = i

    return data

def compute_smooth_saturation(x: np.ndarray, Fs: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes a smooth SpO2 signal and identifies error regions.

    Args:
    - x: Raw SpO2 signal as a numpy array.
    - Fs: Sampling frequency (Hz).

    Returns:
    - Tuple containing the smoothed SpO2 signal and an error array.
    """
    sat_trace = np.array(x).astype(float)
    
    # Remove obvious artifacts (SpO2 values below 61 are considered artifacts)
    sat_trace[np.less(sat_trace, 61, where=np.isfinite(sat_trace))] = np.nan
    
    # Cut out segments with unrealistic drops
    shift = int(0.5 * Fs)
    drops = find_events(np.diff(sat_trace) > 3)
    diff = np.squeeze(pd.DataFrame(data=sat_trace, columns=['sat_trace']).diff(periods=int(0.5 * Fs)).values)
    drops += find_events(np.abs(diff) > 5)
    
    # Mark the regions with unrealistic changes as NaN
    for st, end in drops:
        sat_trace[st - shift:end + shift] = np.nan
    
    # Further remove segments with short valid regions (< 3 seconds)
    for st, end in find_events(np.isfinite(sat_trace)):
        if end - st < 3 * Fs:
            sat_trace[st:end] = np.nan
    
    # Mark all values below 65 as NaN
    sat_trace[sat_trace < 65] = np.nan

    # Interpolate small gaps < 30 seconds
    smooth_saturation = pd.DataFrame(sat_trace).interpolate(method='nearest', limit=30 * Fs, limit_area='inside')
    
    # Interpolate small gaps < 20 seconds
    smooth_saturation = smooth_saturation.interpolate(method='nearest', limit=20 * Fs, limit_area='inside')
    
    # Smooth the signal with a 1-second rolling median
    smooth_saturation = np.squeeze(smooth_saturation.rolling(Fs, center=True).median().values)
    
    # Compute error regions using a sliding window of 20 minutes with a stride of 5 minutes
    win, stride = 5 * 60 * Fs, 1 * 60 * Fs
    error_array = np.zeros(len(x))
    
    for start in np.arange(0, len(x) - win, stride):
        reg = list(range(start, start + win))
        events = find_events(x[reg] < 61)
        
        # Count the number of error segments > 8 seconds in the region
        val = len([s for s, e in events if (e - s) > 8 * Fs])
        
        # Mark regions with more than 5 error events as invalid (> 1 per minute)
        if val > 5:
            error_array[reg] = 1

    return smooth_saturation, error_array

def find_desats_within_range(data: pd.DataFrame, sat_drop: float, ran: int, Fs: int, tag: str) -> pd.DataFrame:
    """
    Finds desaturation events within a specific range.

    Args:
    - data: DataFrame containing the saturation data and other signal information.
    - sat_drop: Threshold value for saturation drop detection.
    - ran: Window size (in seconds) for which to detect desaturation events.
    - Fs: Sampling frequency (Hz).
    - tag: Column name to store the detected desaturation events.

    Returns:
    - The updated DataFrame with detected desaturation events.
    """

    # Set thresholds for minimum and maximum saturation values
    smooth_sat = data['smooth_saturation'].values
    thresh = Fs / ran
    h_w = ran // 2
    
    # Compute rolling quantiles to find min/max saturation values within the range
    sat_min = data['smooth_saturation'].rolling(ran, center=True).quantile(thresh, interpolation='lower')
    sat_max = data['smooth_saturation'].rolling(ran, center=True).quantile(1 - thresh, interpolation='higher')

    # Determine potential desaturation drops
    data['potential_saturation_drop'] = np.array((sat_max - sat_min) >= sat_drop).astype(int)

    # Run through all potential drops found in 'potential_saturation_drop'
    data[tag] = 0
    for drop in find_events(data['potential_saturation_drop'] > 0):
        region = list(range(drop[0], drop[1]))

        # Skip if the drop is already found using a smaller window in previous iterations
        if not np.all(data.saturation_drop[region] == 0):
            beg = np.where(data.saturation_drop[region] == 1)[0][0]
            end = np.where(data.saturation_drop[region] == 1)[0][-1]

            if beg == 0:
                area = list(range(drop[0] + end + 1, region[-1]))
            else:
                area = list(range(drop[0], drop[0] + beg))

            # Skip invalid areas based on SpO2 and region length
            if (data.loc[area, 'SpO2'].max() - data.loc[area, 'SpO2'].min() < sat_drop) or len(area) <= 2 * Fs:
                continue
        else:
            area = list(range(drop[0] - h_w, drop[1] + h_w))

        # Cut area if it exceeds signal length
        if area[-1] > len(data) - 1:
            area = list(range(area[0], len(data)))

        # Define the first and last half of the area
        f_area = area[: len(area) // 2]
        l_area = area[len(area) // 2 :]

        # Locate the start ('s') and end ('e') location of the desaturation drop
        s = np.where(smooth_sat[f_area] == np.nanmax(smooth_sat[f_area]))[0][-1]
        if s > (len(f_area) - Fs):
            s = np.where(smooth_sat[area] == np.nanmax(smooth_sat[area]))[0][-1]

        e = np.where(smooth_sat[l_area] == np.nanmin(smooth_sat[l_area]))[0][-1]
        fill = list(range(area[0] + s, len(area) // 2 + area[0] + e))

        # Remove drop if it does not meet length or threshold criteria
        check1 = len(fill) <= 2 * Fs
        check2 = data.loc[fill[: Fs], 'smooth_saturation'].max() - data.loc[fill[-Fs:], 'smooth_saturation'].min() < sat_drop
        if check1 or check2:
            continue
        else:
            data.loc[fill, tag] = 1

    # Remove the temporary 'potential_saturation_drop' column
    data = data.drop(columns='potential_saturation_drop')

    return data

def add_quick_two_desats(data: pd.DataFrame, base_win: int, Fs: int) -> pd.DataFrame:
    """
    Detects 2% desaturations and adds corresponding soft 3% desaturation events.

    Args:
    - data: DataFrame containing the saturation data.
    - base_win: Window size (in samples) for determining the baseline saturation level.
    - Fs: Sampling frequency (Hz).

    Returns:
    - The updated DataFrame with detected 2% and soft 3% desaturation events.
    """

    # Find 2% desaturations across different ranges
    ranges = np.array([5, 10, 15]) * Fs
    for ran in ranges:
        data = find_desats_within_range(data, 2, ran, Fs, '2%_desats')

    # Initialize the column for soft 3% desaturations
    data['soft_3%_desats'] = 0

    # Iterate over all detected 2% desaturations
    for st, end in find_events(data['2%_desats'] > 0):
        # Skip if the 2% desaturation overlaps with an existing desaturation
        if np.any(data.loc[st:end, 'saturation_drop'] == 1):
            continue
        
        # Define the region to search for a 3% desaturation delta
        start = max(0, end - base_win)
        ending = min(len(data), end + (base_win // 3))

        # Check if a 3% desaturation occurs within the defined region
        if data.loc[start:ending, 'smooth_saturation'].max() - data.loc[st:end, 'smooth_saturation'].min() >= 3:
            data.loc[st:end, 'saturation_drop'] = 1
            data.loc[st:end, 'soft_3%_desats'] = 1

    # Remove the temporary column for 2% desaturations
    data = data.drop(columns='2%_desats')

    return data

def merge_connecting_saturation_drops(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Merges adjacent or close saturation drops if the saturation continues to decrease between the events.

    Args:
    - data: DataFrame containing the saturation data and saturation drop events.
    - hdr: Dictionary containing metadata, including the sampling frequency ('newFs').

    Returns:
    - The updated DataFrame with merged saturation drops.
    """
    
    # Find all saturation drop events
    sat_drops = find_events(data['saturation_drop'] > 0)
    
    # Compute smoothed saturation with a 2-second window
    smooth_saturation = data['SpO2'].rolling(2 * hdr['newFs'], center=True).median()
    
    # Iterate over consecutive saturation drop events
    for i, (st, end) in enumerate(sat_drops[:-1]):
        next_st, next_end = sat_drops[i + 1]
        
        # Check if the saturation keeps going down between the current and next event
        keep_going_down = all(np.diff(smooth_saturation[st:next_end:2 * hdr['newFs']]) < 0)
        
        # If the gap between events is smaller than 15 seconds and saturation keeps decreasing, merge the events
        if next_st - end < 15 * hdr['newFs'] and keep_going_down:
            data.loc[st:next_end, 'saturation_drop'] = 1

    return data
