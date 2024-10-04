import os, sys
import numpy as np

# set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import utils
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Event_array_modifiers import *


# Post-Processing functions:
def post_processing(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Perform post-processing on the apnea event data.

    Args:
    - data: DataFrame containing event data.
    - hdr: Header dictionary containing metadata, e.g., sampling frequency.

    Returns:
    - Updated DataFrame with post-processed apnea events.
    """

    events = find_events(data['algo_apneas'] > 0)
    labels = [data['algo_apneas'].iloc[st] for st, _ in events]
    
    # Connect very close together events
    new_events, new_labels = connect_events(events, 3, hdr['newFs'], max_dur=20, labels=labels)
    data['algo_apneas'] = events_to_array(new_events, len(data), labels=new_labels)

    # Split long events according to arousals
    data = split_long_events(data, hdr['newFs'])

    # Remove short events
    data['algo_apneas'] = remove_short_events(data['algo_apneas'], 6 * hdr['newFs'], skip_RERA=True)

    # Remove long events beyond a certain duration
    data = remove_long_events(data, 'algo_apneas', hdr['newFs'], max_duration=90)

    return data

def split_long_events(data: pd.DataFrame, Fs: int, tag: str = 'algo_apneas') -> pd.DataFrame:
    """
    Splits long events (>30s) based on arousals or saturation drops.

    Args:
    - data: DataFrame containing event data.
    - Fs: Sampling frequency.
    - tag: Column name for the event data (default 'algo_apneas').

    Returns:
    - Updated DataFrame with split events.
    """
    # Find all events in the specified column
    events = find_events(data[tag] > 0)

    for st, end in events:
        # Only process events longer than 30 seconds
        if (end - st) <= 30 * Fs:
            continue

        # Define region of interest, extending 10 seconds after the event if possible
        region_end = min(end + 10 * Fs, len(data))
        region = list(range(st, region_end))
        event_type = np.max(data.loc[st:end, tag])  # Identify event type

        # Look for arousals in the region (ignoring those in the first 10 seconds)
        arousals = [a for a in find_events(data.loc[region, 'EEG_arousals'] > 0) if a[0] > 10 * Fs]

        # If no arousals are found, look for saturation drops
        if len(arousals) <= 1:
            arousals = [a for a in find_events(data.loc[region, 'saturation_drop'] > 0) if a[0] > 10 * Fs]
            if len(arousals) <= 1:
                continue

        # Split the long event based on the detected arousals or drops
        for i, arousal in enumerate(arousals):
            # Zero out the event in the arousal region
            data.loc[region[arousal[0]:arousal[1] + 5 * Fs], tag] = 0

            # Split into two regions: before and after the arousal
            if i == 0:
                first_region = region[:arousal[0]]
            else:
                first_region = region[arousals[i - 1][1] + 5 * Fs:arousal[0]]

            if i == len(arousals) - 1:
                second_region = region[arousal[1] + 5 * Fs:]
            else:
                second_region = region[arousal[1] + 5 * Fs:arousals[i + 1][0]]

            # Update the data for each split region
            data.loc[first_region, tag] = event_type
            data.loc[second_region, tag] = event_type

    return data

def remove_long_events(data: pd.DataFrame, tag: str, Fs: int, max_duration: int = 60) -> pd.DataFrame:
    """
    Removes events with duration longer than `max_duration` from the specified column.
    If the event is associated with a saturation drop or arousal, only a portion of the event is retained.

    Args:
    - data: DataFrame containing event data.
    - tag: Column name containing the event data to check.
    - Fs: Sampling frequency (samples per second).
    - max_duration: Maximum allowable event duration in seconds (default is 60).

    Returns:
    - Updated DataFrame with long events removed or shortened.
    """

    data['too_long_events'] = 0
    # Find all events in the specified column that exceed the max_duration
    events = [ev for ev in find_events(data[tag] > 0) if (ev[1] - ev[0]) > max_duration * Fs]

    for st, end in events:
        region = list(range(st, end))
        data.loc[region, 'too_long_events'] = 1

        # For 'algo_apneas' events with type 4, try to keep part based on saturation drops or arousals
        if tag == 'algo_apneas' and data.loc[st, tag] == 4:
            data.loc[region, tag] = 0  # Clear the event initially

            # Check for a saturation drop within the region
            drops = [a for a in find_events(data.loc[region, 'saturation_drop'] > 0) if a[0] > 10 * Fs]
            arousals = [a for a in find_events(data.loc[region, 'EEG_arousals'] > 0) if a[0] > 10 * Fs]

            if len(drops) == 1:
                loc = int(st + np.mean(drops[0]))
                fill = list(range(loc - 30 * Fs, loc))
                data.loc[fill, tag] = 4
            elif len(arousals) == 1:
                loc = int(st + np.mean(arousals[0]))
                fill = list(range(loc - 30 * Fs, loc))
                data.loc[fill, tag] = 4
        else:
            # Completely remove the event if it doesn't match the special case
            data.loc[region, tag] = 0

    return data

def remove_short_events(array: np.ndarray, duration: int, skip_RERA: bool = False) -> np.ndarray:
    """
    Removes events with duration shorter than the specified 'duration'.
    Optionally skips events that are labeled as RERA (type 5).

    Args:
    - array: 1D numpy array of event labels.
    - duration: Minimum event duration in samples.
    - skip_RERA: If True, skip removing RERA events (type 5).

    Returns:
    - Updated numpy array with short events removed.
    """
    array = np.array(array)  # Ensure input is a numpy array

    for st, end in find_events(array > 0):
        # Skip RERA events if required
        if array[st] == 5 and skip_RERA:
            continue
        region = list(range(st, end))
        if len(region) < duration:
            array[region] = 0  # Remove the event if it's too short

    return array

def remove_events_when_bad_signal(data: pd.DataFrame, Fs: int) -> pd.DataFrame:
    """
    Removes events from 'algo_apneas' if they overlap with regions marked as having bad signal.

    Args:
    - data: DataFrame containing event and bad signal information.
    - Fs: Sampling frequency (samples per second).

    Returns:
    - Updated DataFrame with bad signal events removed.
    """
    # Check if 'bad_signal' exists in the DataFrame
    if 'bad_signal' not in data.columns:
        return data

    # Loop through all apnea events and remove them if overlapping with bad signal
    for st, end in find_events(data['algo_apneas'] > 0):
        if np.any(data.loc[st:end, 'bad_signal']):
            data.loc[st:end, 'algo_apneas'] = 0  # Remove event if bad signal found

    return data

def remove_wake_events(data: pd.DataFrame) -> pd.DataFrame:
    """
    Removes flow reduction events that occur primarily when the patient is awake.

    Args:
    - data: DataFrame containing event and patient sleep state information.

    Returns:
    - Updated DataFrame with wake events removed.
    """
    # If all data points indicate the patient is asleep, return the data unchanged
    if np.all(data['patient_asleep'] == 1):
        return data

    # Find all flow reduction events
    all_flow_reductions = find_events(data['flow_reductions'] > 0)

    # Remove flow reduction events if most of the event occurs during wakefulness
    for st, end in all_flow_reductions:
        region = list(range(st, end))
        if np.sum(data.loc[region, 'patient_asleep'] == 0) > 0.75 * len(region):  # More than 75% during wake
            data.loc[region, 'flow_reductions'] = 0  # Remove the event

    return data

