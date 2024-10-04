import os, glob
import numpy as np
import pandas as pd


import numpy as np
import pandas as pd
from typing import List, Tuple

# Find Events in a boolean signals array
def find_events(sig: np.ndarray, Fs: int = 10) -> List[Tuple[int, int]]:
    """
    Finds start and end points of events in a boolean signal array.

    Args:
        sig (np.ndarray): A boolean array representing events.
        Fs (int, optional): Sampling frequency. Defaults to 10.

    Returns:
        List[Tuple[int, int]]: A list of tuples where each tuple contains the start and end indices of an event.
    """
    
    signal = np.array(sig)
    
    # Initialize lists for starts and ends of events
    starts, ends = [], []

    # Add start if the signal starts with an active event
    if signal[0] > 0:
        starts.append(0)

    # Compute the difference of the signal
    diff_drops = pd.DataFrame(signal).diff()

    # Find starts and ends of events using a helper function
    starts, ends = define_events_start_ends(signal, diff_drops, starts, ends)

    # Add end if the signal ends with an active event
    if signal[-1] > 0:
        ends.append(len(signal))
        signal[-1] = 0

    # If no events found, return empty list
    if len(ends) == len(starts) == 0:
        return []

    # If starts and ends do not match, perform label correction
    if len(ends) != len(starts):
        print('Attempting label correction...')
        starts, ends = label_correction(starts, ends, signal, Fs)
        
    assert len(ends) == len(starts), "Mismatched starts and ends after correction."

    # Zip starts and ends into a list of event tuples
    grouped_events = list(zip(starts, ends))
    
    return grouped_events

# Helper function to define event start and end points
def define_events_start_ends(signal: np.ndarray, diff_drops: pd.DataFrame, starts: List[int], ends: List[int]) -> Tuple[List[int], List[int]]:
    """
    Helper function to define the start and end points of events in a signal.

    Args:
        signal (np.ndarray): Boolean array representing the signal.
        diff_drops (pd.DataFrame): Difference of the signal.
        starts (List[int]): List of start indices of events.
        ends (List[int]): List of end indices of events.

    Returns:
        Tuple[List[int], List[int]]: Updated lists of start and end indices of events.
    """
    
    for v in np.where(diff_drops[1:])[0]:
        loc = v + 1
        step = signal[loc]
        step_min_one = signal[loc - 1]

        if step > step_min_one:
            starts.append(loc)
        elif step < step_min_one:
            ends.append(loc)
        else:
            raise ValueError('Unexpected event error in find_events.')

    return starts, ends

# Helper function to correct mismatched event labels
def label_correction(starts: List[int], ends: List[int], signal: np.ndarray, Fs: int) -> Tuple[List[int], List[int]]:
    """
    Corrects mismatched event labels when starts and ends are not aligned.

    Args:
        starts (List[int]): List of start indices of events.
        ends (List[int]): List of end indices of events.
        signal (np.ndarray): Boolean array representing the signal.
        Fs (int): Sampling frequency.

    Returns:
        Tuple[List[int], List[int]]: Corrected lists of start and end indices of events.
    """
    
    # Search for merged labels (overlapping events)
    merged_locs = search_for_merged_labels(signal)
    
    for p, loc, n in merged_locs:
        # Split the merged events
        event1 = signal[loc - 1]
        event2 = signal[loc + 1]

        # Check which event has priority (decide which event to keep)
        priority_loc = np.argmin([event1, event2])

        if priority_loc == 0:
            # Keep the first event, modify the ends accordingly
            starts = [s for s in starts if p != s]
            if n in ends:
                ends[ends.index(n)] = loc
        else:
            # Keep the second event, modify the starts accordingly
            ends = [e for e in ends if loc != e]
            if p in starts:
                starts[starts.index(p)] = loc

    return starts, ends

# Helper function to find merged event labels
def search_for_merged_labels(signal: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Searches for locations where events are merged (overlapping).

    Args:
        signal (np.ndarray): Boolean array representing the signal.

    Returns:
        List[Tuple[int, int, int]]: List of tuples where events are merged.
    """
    
    locs = []
    diff_drops = pd.DataFrame(signal).diff().fillna(0)
    
    # Drop zero values (no change in signal)
    diff_drops = diff_drops[diff_drops != 0].dropna()

    # Loop through the signal and find overlapping events
    for i in range(len(diff_drops) - 1):
        current_val = diff_drops.iloc[i].values[0]
        next_val = diff_drops.iloc[i + 1].values[0]

        if current_val > 0 and next_val > 0:
            locs.append((diff_drops.index[i], diff_drops.index[i + 1], diff_drops.index[i + 2]))
        elif current_val < 0 and next_val < 0:
            locs.append((diff_drops.index[i - 1], diff_drops.index[i], diff_drops.index[i + 1]))
    
    return locs



# Convert list of events (start, end) to an array
def events_to_array(events: List[Tuple[int, int]], len_array: int, labels: List[int] = []) -> np.ndarray:
    """
    Converts a list of events (start, end) to a 1D array where the array values represent labels for events.

    Args:
        events (List[Tuple[int, int]]): List of tuples containing event start and end indices.
        len_array (int): Length of the resulting array.
        labels (List[int], optional): List of labels corresponding to each event. Defaults to [] (all events assigned label 1).

    Returns:
        np.ndarray: Array with event labels applied to the event regions.
    """
    
    array = np.zeros(len_array)
    if len(labels) == 0: 
        labels = [1] * len(events)  # Default all labels to 1 if not provided
    
    for i, (st, end) in enumerate(events):
        array[st:end] = labels[i]
        
    return array

# Correct window after applying a moving window
def window_correction(array: np.ndarray, window_size: int) -> np.ndarray:
    """
    Corrects events in an array by extending the label regions based on a window size.

    Args:
        array (np.ndarray): Input array with event labels.
        window_size (int): The size of the window to extend the events.

    Returns:
        np.ndarray: Array with window-corrected event labels.
    """
    
    half_window = int(window_size // 2)
    events = find_events(array)
    corr_array = np.array(array)
    
    # Apply window correction for each event
    for st, end in events:
        label = array[st]
        corr_array[max(0, st - half_window):min(len(array), end + half_window)] = label  # Handle boundary cases

    return corr_array.astype(int)

# Connect nearby events
def connect_events(events: List[Tuple[int, int]], win: int, Fs: int, max_dur: bool = False, labels: List[int] = []) -> Tuple[List[Tuple[int, int]], List[int]]:
    """
    Connects nearby events if they occur within a given window size and merges them into one event.

    Args:
        events (List[Tuple[int, int]]): List of events (start, end).
        win (int): The window size to connect events (in seconds).
        Fs (int): The sampling frequency of the signal.
        max_dur (bool, optional): Whether to limit the maximum duration of merged events. Defaults to False.
        labels (List[int], optional): Labels for each event. Defaults to [] (all events assigned label 1).

    Returns:
        Tuple[List[Tuple[int, int]], List[int]]: Merged events and their corresponding labels.
    """
    
    new_events, new_labels = [], []
    if len(events) > 0:
        cnt = 0
        win = win * Fs
        if len(labels) == 0:
            labels = [1] * len(events)
        
        while cnt < len(events) - 1:
            st = events[cnt][0]
            end = events[cnt][1]
            dist_to_next_event = events[cnt + 1][0] - end
            
            # Condition to merge events if within the window and optionally limited by max duration
            if (dist_to_next_event < win) and (not max_dur or (events[cnt + 1][1] - st < max_dur * Fs)):
                new_events.append((st, events[cnt + 1][1]))  # Merge events
                # Assign label based on the longer duration event
                label = labels[cnt] if (end - st) > (events[cnt + 1][1] - events[cnt + 1][0]) else labels[cnt + 1]
                new_labels.append(label)
                cnt += 2  # Move past the merged events
            else:
                new_events.append((st, end))
                new_labels.append(labels[cnt])
                cnt += 1
        
        # Add the last event if not merged
        new_events.append(events[-1])
        new_labels.append(labels[-1])

    return new_events, new_labels

