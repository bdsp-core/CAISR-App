import os, sys
import numpy as np
import pandas as pd
from typing import List, Tuple

# set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import utils
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Event_array_modifiers import find_events

# Hypopnea Analysis Functions
def match_saturation_and_ventilation_drops(data: pd.DataFrame, hdr: dict, hyp_range: int = 45) -> pd.DataFrame:
    """
    Matches saturation drops with ventilation drops (hypopneas).

    Args:
    - data: A pandas DataFrame containing saturation and ventilation data.
    - hdr: Header dictionary containing metadata, such as newFs (sampling frequency).
    - hyp_range: Hypopnea-saturation matching range in seconds.

    Returns:
    - Updated DataFrame with accepted/rejected hypopneas based on saturation drops.
    """
    
    data['potential_saturation_drops'] = data['saturation_drop']
    data['accepted_saturation_hypopneas'] = 0
    data['rejected_saturation_hypopneas'] = 0
    potential_hyp = find_events(data['Ventilation_drop_hypopnea'].fillna(0) > 0)

    # Convert hyp_range to samples
    hyp_range_samples = hyp_range * hdr['newFs']

    # Run backwards through all events
    for p_h in potential_hyp:
        hyp = list(range(p_h[0], p_h[1]))  # Define ventilation drop as potential hypopnea
        region = define_search_region(p_h, hyp_range_samples, data, hdr['newFs'])  # Define search region

        # Match potential hypopnea event with saturation drop
        if hypopnea_match_making(data, hyp, region, 'potential_saturation_drops'):
            data.loc[hyp, 'accepted_saturation_hypopneas'] = 1
            continue
        elif sum(data.loc[region, 'error_saturation'] == 1) > 0.8 * len(region):
            if any(data.loc[region, 'patient_asleep'] == 1):
                print(hdr['patient_tag'], p_h)
                data.loc[hyp, 'accepted_saturation_hypopneas'] = 1
                continue
        data.loc[hyp, 'rejected_saturation_hypopneas'] = 1

    # Remove multiple hypopneas referring to the same desaturation
    data = remove_multiple_satuturation_references(data, hdr, max_match=2)

    # Remove temporary columns
    data = data.drop(columns='potential_saturation_drops')

    return data

def match_EEG_with_ventilation_drops(data: pd.DataFrame, hdr: dict, EEG_range: int = 25) -> pd.DataFrame:
    """
    Matches EEG arousals with ventilation drops (hypopneas).

    Args:
    - data: A pandas DataFrame containing EEG and ventilation data.
    - hdr: Header dictionary containing metadata, such as newFs (sampling frequency).
    - EEG_range: EEG-arousal matching range in seconds.

    Returns:
    - Updated DataFrame with accepted/rejected hypopneas based on EEG arousals.
    """
    
    data['potential_hypopnea_arousals'] = data['EEG_arousals']
    if 'added_arousals' in data.columns:
        data['potential_hypopnea_arousals'] += data['added_arousals']
    data['accepted_EEG_hypopneas'] = 0
    data['rejected_EEG_hypopneas'] = 0
    potential_hyp = find_events(data['Ventilation_drop_hypopnea'].fillna(0) > 0)
    EEG_range_samples = EEG_range * hdr['newFs']

    # Run over all potential hypopneas
    for p_h in potential_hyp:
        hyp = list(range(p_h[0], p_h[1]))
        region = define_search_region(p_h, EEG_range_samples, data, hdr['newFs'])

        if hypopnea_match_making(data, hyp, region, 'potential_hypopnea_arousals'):
            data.loc[hyp, 'accepted_EEG_hypopneas'] = 1
        else:
            data.loc[hyp, 'rejected_EEG_hypopneas'] = 1

    # Remove temporary columns
    data = data.drop(columns='potential_hypopnea_arousals')

    return data

def define_search_region(p_h: Tuple[int, int], search_range: int, data: pd.DataFrame, Fs: int) -> List[int]:
    """
    Defines a search region for hypopnea matching based on the range and sampling frequency.

    Args:
    - p_h: Tuple of the start and end indices of a potential hypopnea.
    - search_range: Search range in samples.
    - data: A pandas DataFrame containing the data.
    - Fs: Sampling frequency in Hz.

    Returns:
    - List of indices defining the search region.
    """
    
    st, end = p_h[0], p_h[1]
    hyp = list(range(st, end + 10 * Fs))
    region = list(range(st, int((st + end) // 2 + search_range)))

    if hyp[-1] > region[-1]:
        region = hyp
    if region[-1] >= len(data) - 1:
        region = list(range(st, len(data)))

    # Find and group possible upcoming events
    all_apneas = data.loc[region, 'Ventilation_drop_apnea'].mask(data.loc[region, 'Ventilation_drop_apnea'] == 0).fillna(0)
    hypopneas = data.loc[region, 'Ventilation_drop_hypopnea'].mask(data.loc[region, 'Ventilation_drop_hypopnea'] == 0).fillna(0)
    events = all_apneas + hypopneas
    events = find_events(events > 0)

    if len(events) > 1:
        region = list(range(st, st + events[1][0] + int(4 * Fs)))

    return region

def hypopnea_match_making(data: pd.DataFrame, hyp: List[int], region: List[int], tag: str) -> bool:
    """
    Matches a hypopnea event with a desaturation or arousal event.

    Args:
    - data: A pandas DataFrame containing the data.
    - hyp: List of indices for the hypopnea event.
    - region: List of indices for the search region.
    - tag: Column name to search for desaturation or arousal events.

    Returns:
    - True if a match is found, False otherwise.
    """
    
    desats_or_arousals = find_events(data.loc[region, tag].fillna(0) > 0)
    found = False

    for st, end in desats_or_arousals:
        if st == 0 and (hyp[-1] - hyp[0]) > end:
            continue
        found = True

    return found

def remove_multiple_satuturation_references(data: pd.DataFrame, hdr: dict, max_match: int) -> pd.DataFrame:
    """
    Removes multiple hypopneas referring to the same desaturation event, keeping the most significant ones.

    Args:
    - data: A pandas DataFrame containing hypopneas and desaturation data.
    - hdr: Header dictionary containing metadata, such as newFs (sampling frequency).
    - max_match: Maximum number of hypopneas that can refer to the same desaturation.

    Returns:
    - Updated DataFrame with excess references removed.
    """
    
    Fs = hdr['newFs']
    desats = find_events(data['saturation_drop'] > 0)

    for i, (st, end) in enumerate(desats):
        prev_end = desats[i - 1][1] if i > 0 else 0
        start = max(st - 60 * Fs, prev_end, 0)

        depth = data.loc[st:end, 'smooth_saturation'].max() - data.loc[st:end, 'smooth_saturation'].min()
        desat_two = True if depth == 2 else False

        hyps = find_events(data.loc[start:end, 'accepted_saturation_hypopneas'] > 0)

        if len(hyps) > max_match or (len(hyps) > 1 and desat_two):
            excursions = []
            for j in range(len(hyps)):
                ss = start if j == 0 else hyps[j][0] + start
                ee = end if j == len(hyps) - 1 else hyps[j][1] + start
                pos_env = data.loc[ss:ee, 'Ventilation_pos_envelope']
                neg_env = data.loc[ss:ee, 'Ventilation_neg_envelope']
                excursions.append(np.mean(pos_env - neg_env))

            keep = 1 if desat_two else max_match
            remove = np.array(hyps)[np.argsort(excursions)[keep:]] + start
            for r0, r1 in remove:
                data.loc[r0:r1, 'accepted_saturation_hypopneas'] = 0

    return data

def remove_multiple_arousal_references(data: pd.DataFrame, hdr: dict, max_match: int) -> pd.DataFrame:
    """
    Removes multiple hypopneas referring to the same EEG arousal event, keeping the most significant ones.

    Args:
    - data: A pandas DataFrame containing hypopneas and EEG arousal data.
    - hdr: Header dictionary containing metadata, such as newFs (sampling frequency).
       - max_match: Maximum number of hypopneas that can refer to the same EEG arousal.

    Returns:
    - Updated DataFrame with excess references removed.
    """
    
    Fs = hdr['newFs']
    arousals = find_events(data['EEG_arousals'] > 0)

    for i, (st, end) in enumerate(arousals):
        prev_end = arousals[i - 1][1] if i > 0 else 0
        start = max(st - 60 * Fs, prev_end, 0)

        # Find all hypopneas in the window
        hyps = find_events(data.loc[start:end, 'accepted_EEG_hypopneas'] > 0)

        if len(hyps) > max_match:
            excursions = []
            for j in range(len(hyps)):
                ss = start if j == 0 else hyps[j][0] + start
                ee = end if j == len(hyps) - 1 else hyps[j][1] + start
                pos_env = data.loc[ss:ee, 'Ventilation_pos_envelope']
                neg_env = data.loc[ss:ee, 'Ventilation_neg_envelope']
                excursions.append(np.mean(pos_env - neg_env))

            # Only keep the <max_match> with the smallest excursion
            remove = np.array(hyps)[np.argsort(excursions)[max_match:]] + start
            for r0, r1 in remove:
                data.loc[r0:r1, 'accepted_EEG_hypopneas'] = 0

    return data

