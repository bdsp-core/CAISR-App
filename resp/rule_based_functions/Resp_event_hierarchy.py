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

# Import additional algorithm functions
sys.path.insert(0, os.path.join(RB_folder, 'rule_based_functions'))
from Ventilation_spectral_analysis import spectral_assess_effort_limitation


# Apnea Comparison functions
def overlapping_event_check(data: pd.DataFrame, hdr: dict, Fs: float) -> pd.DataFrame:
    """
    Checks for overlapping events and adjusts RERA and hypopnea events.

    Args:
        data (pd.DataFrame): Data containing the signal and event information.
        hdr (dict): Header information, including the sampling frequency.
        Fs (float): Sampling frequency.

    Returns:
        pd.DataFrame: Updated data with modified RERA and hypopnea events.
    """

    # Define apnea location arrays
    all_apneas = data['Ventilation_drop_apnea'].fillna(0)
    soft_apneas = data['soft_ventilation_drop_apnea'].fillna(0)
    threes = data['algo_hypopneas_three'].fillna(0)
    
    hyps = ['algo_hypopneas_three', 'algo_hypopneas_four'] if 'algo_hypopneas_four' in data.columns else ['algo_hypopneas_three']
    data['hypopneas'] = data[hyps].any(axis=1, skipna=False).astype(int)
    hypopneas = data['hypopneas'].fillna(0)
    
    reras = data['algo_reras'].fillna(0)
    arousals = data['EEG_arousals'] + data['added_arousals'] if 'added_arousals' in data.columns else data['EEG_arousals']
    
    data['3%_reras'] = 0

    # Remove all RERAs that overlap with an apnea
    grouped_reras = find_events(reras > 0)
    for i, (st, end) in enumerate(grouped_reras):
        try:
            end += np.where(np.diff(arousals[st:]) < 0)[0][0]
        except:
            assert i == len(grouped_reras) - 1, 'Somehow no arousals found after RERA'
            end = len(data)
        region = range(max(0, st - int(10 * Fs)), min(end, len(data)))
        if np.any(threes.loc[region] > 0):
            data.loc[region, '3%_reras'] = 1
        if np.any(all_apneas.loc[region] > 0) or np.any(hypopneas.loc[region] > 0):
            data.loc[region, 'algo_reras'] = 0

    # Remove RERAs that overlap with EEG hypopneas
    EEG_hypopneas = find_events(data.accepted_EEG_hypopneas > 0)
    for i, (st, end) in enumerate(EEG_hypopneas):
        try:
            end += np.where(np.diff(arousals[st:]) < 0)[0][0]
        except:
            assert i == len(EEG_hypopneas) - 1, 'No arousals found after EEG hypopnea'
            end = len(data)
        if np.any(data.loc[st:end, 'algo_reras'] > 0):
            if data.loc[end, 'algo_reras'] > 0:
                end += np.where(data.loc[end:, 'algo_reras'].diff() < 0)[0][0]
                end = min(end, len(data))
            data.loc[st:end, 'algo_reras'] = 0

    # Remove hypopneas that overlap with an apnea
    grouped_hypopneas = find_events(hypopneas > 0)
    for st, end in grouped_hypopneas:
        if np.any(all_apneas.loc[st - 1:end] > 0):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 1
            data.loc[st:end, hyps] = 0
            data.loc[st:end, 'accepted_saturation_hypopneas'] = 0
            data.loc[st:end, 'accepted_EEG_hypopneas'] = 0

    data = data.drop(columns=['hypopneas'])
    return data

def do_apnea_multiclassification(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Performs multi-classification of apnea events.

    Args:
        data (pd.DataFrame): Data containing the signal and event information.
        hdr (dict): Header information, including the sampling frequency.

    Returns:
        pd.DataFrame: Updated data with multi-classification of apnea events.
    """

    Fs = hdr['newFs']
    
    if 'algo_reras' not in data.columns:
        data['algo_reras'] = 0

    data = overlapping_event_check(data, hdr, Fs)

    all_apneas = data['Ventilation_drop_apnea'].fillna(0)
    grouped_apneas = find_events(all_apneas > 0)

    central_events = find_events(data.central_component > 0)
    for st, end in grouped_apneas:
        middle = (st + end) // 2
        start, ending = st + int(2 * Fs), end
        region = range(start, ending)

        # Compute central component ratio
        thresh = 'xsmall' if all(data.loc[st:end, ['ABD_tiny_exc', 'CHEST_tiny_exc']] == 0) else 'tiny'
        cols = [f'ABD_{thresh}_exc', f'CHEST_{thresh}_exc']
        left_weight = data.loc[st:middle, cols].sum().sum()
        right_weight = data.loc[middle:end, cols].sum().sum()

        ratio = (left_weight - right_weight) / (left_weight + right_weight) if (left_weight + right_weight) > 0 else 0

        if np.any(data.loc[st:end, 'central_component'] == 1):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 2
            if np.any(data.loc[st:end, 'mixed_component'] == 1):
                not_breathing = spectral_assess_effort_limitation(data, st, end, Fs, 'breathing_trace')
                if not not_breathing[0] and not not_breathing[1]:
                    continue
                data.loc[st:end, 'Ventilation_drop_apnea'] = 3
                continue

        if data.loc[st, 'Ventilation_drop_apnea'] == 2 and ratio > 0.14:
            not_breathing = spectral_assess_effort_limitation(data, st, end, Fs, 'breathing_trace')
            if not not_breathing[0] and not not_breathing[1]:
                continue
            data.loc[st:end, 'Ventilation_drop_apnea'] = 3

    data = set_hypopneas(data)
    return data

def set_hypopneas(data: pd.DataFrame) -> pd.DataFrame:
    """
    Sets hypopneas based on % desaturation rules.

    Args:
        data (pd.DataFrame): Data containing the signal and event information.

    Returns:
        pd.DataFrame: Updated data with hypopneas set based on desaturation.
    """
    
    if 'algo_hypopneas_four' in data.columns:
        print('Check hypopnea computation setup!')
        import pdb; pdb.set_trace()

        three = 'algo_hypopneas_three'
        four = 'algo_hypopneas_four'
        threes = find_events(data[three] > 0)

        for st, end in threes:
            if np.any(data.loc[st:end, four] == 4):
                data.loc[st:end, three] = 0

        data.loc[data[four] == 4, four] = 6

        data['algo_apneas'] = data['Ventilation_drop_apnea'] + data['algo_hypopneas_three'] + \
                              data['algo_hypopneas_four'] + data['algo_reras']
    else:
        data['algo_apneas'] = data['Ventilation_drop_apnea'] + data['algo_hypopneas_three'] + \
                              data['algo_reras']

    return data
