import os, sys, mne, h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample_poly, butter, filtfilt
mne.set_log_level(verbose='CRITICAL')

from bdsp_sleep_functions import (
    load_bdsp_signal, 
    annotations_preprocess, 
    vectorize_respiratory_events, 
    vectorize_sleep_stages, 
    vectorize_arousals, 
    vectorize_limb_movements
)

import neurokit2 as nk

def compute_hr_from_ecg(ecg: np.ndarray, fs: int = 200):
    """
    Compute heart rate from a 1D ECG signal using NeuroKit2.

    Args:
        ecg (np.ndarray): 1D ECG trace.
        fs (int): Sampling frequency of the ECG signal in Hz. Default is 200 Hz.

    Returns:
        np.ndarray: Estimated heart rate (in bpm) at the same sampling frequency as the ECG signal.
    """
    # Process the ECG signal to extract R-peaks and HR
    ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=fs)
    # signals, info = nk.ecg_process(ecg_cleaned, sampling_rate=fs, hrv_features=None)
    # info = nk.ecg_findpeaks(ecg_cleaned, sampling_rate=fs)
    # Identify and Correct Peaks using "Kubios" Method
    rpeaks_uncorrected = nk.ecg_findpeaks(ecg_cleaned, method="pantompkins", sampling_rate=fs)
    info, rpeaks_corrected = nk.signal_fixpeaks(
        rpeaks_uncorrected, sampling_rate=fs, iterative=True, method="Kubios", show=False
    )
    rate_corrected = nk.signal_rate(rpeaks_corrected, desired_length=len(ecg))

    # Extract heart rate from processed signals
    heart_rate = rate_corrected
    
    return heart_rate


def preprocess_signals(signal, fs_original, fs_target=200, autoscale_signals=True, verbose=False):

    verbose = False
    # Standardize channel names
    signal = standardize_channel_names(signal, verbose)

    # Check if all needed channels are present and create derived channels if necessary
    signal = check_channels_present(signal, verbose)

    # Resample signals to 200 Hz
    signal = resample_signals(signal, fs_original, fs_target)

    if autoscale_signals:
        # Detect signal unit (uV or V) and scale accordingly
        unit = detect_signal_unit(signal['c4-m1'].values, fs=200)
        if unit == 'uV':
            if verbose:
                print('Scaling signal channels from uV to V.')
            # Scale signal channels from uV to V
            signal = scale_signal_to_voltage(signal)

    auto_scale_spo2 = True
    if auto_scale_spo2:
        spo2 = signal['spo2'].values
        scale_factor = 1
        while (np.percentile(spo2, 97) < 1) and (np.median(spo2) > 0) and (np.median(spo2) < 1):
            scale_factor *= 100
            spo2 *= 100
        if scale_factor > 1:
            signal['spo2'] = spo2
        # clip to 0-100:
        signal['spo2'] = np.clip(signal['spo2'], 0, 100)
        
    return signal
    
def preprocess_annotations(annotations, fs_original, signal_len, fs_target=200, verbose=False):
    """ Preprocess the annotations to vectorize the sleep stages and events. """
    
    results = {}
    
    # Preprocess annotations
    annotations, annotations_quality = annotations_preprocess(annotations, fs_original, return_quality=True)
    
    # Vectorize events
    stage = vectorize_sleep_stages(annotations, signal_len)[:, np.newaxis]
    arousal = vectorize_arousals(annotations, signal_len)[:, np.newaxis]
    resp = vectorize_respiratory_events(annotations, signal_len)[:, np.newaxis]
    limb = vectorize_limb_movements(annotations, signal_len)[:, np.newaxis]

    annotations = pd.DataFrame(data=np.concatenate([stage, arousal, resp, limb], axis=1), 
                                columns=['stage', 'arousal', 'resp', 'limb'])
    
    if all(np.isnan(stage)):
        if verbose:
            print('Sleep stages not available.')
        results['stage_available'] = False
        return None, results
    else:
        results['stage_available'] = True

    # Resample signals and annotations to 200 Hz
    annotations = resample_annotations(annotations, fs_original, fs_target, signal_len)
    
    return annotations, results

def preprocess_data(signal, annotations, params, autoscale_signals=True , verbose=False):
    """ Preprocess the signal data and annotations (if annotations should be saved into prepared data).
    Args:
        signal (pd.DataFrame): Raw signal data.
        annotations (pd.DataFrame): Annotations corresponding to the signals.
        params (dict): Parameters like sampling frequency.
        verbose (bool): If True, prints additional information.

    Returns:
        signal (pd.DataFrame), annotations (pd.DataFrame), results (dict)
    """

    # Get the original sampling frequency and signal length
    fs = params['Fs']
    signal_len = signal.shape[0]
    
    # Preprocess signals:
    signal = preprocess_signals(signal, fs, fs_target=200, autoscale_signals=autoscale_signals, verbose=verbose)
    
    # Preprocess annotations:
    if annotations is not None:
        annotations, results = preprocess_annotations(annotations, fs, signal_len, fs_target=200, verbose=verbose)
    else:
        annotations = None
        results = {}
        
    return signal, annotations, results


def determine_cpap_start(cpres, fs):
    # find CPAP start based on permanently increase pressure
    cpap_on = pd.Series(cpres).rolling(int(120*fs*2), center=True, min_periods=1).median().round(decimals=1) > 0
    cpap_on = cpap_on.astype(int).values
    return cpap_on

def standardize_channel_names(signal, verbose=False):
    """ Standardizes the channel names to ensure consistent naming across all files. """
    
    signal.columns = signal.columns.str.lower()
    # also remove "_pds" from the end of the channel names
    signal.columns = signal.columns.str.replace('_pds', '')
    
    rename_rules = {
        'ecg': ['ekg', 'ecg-la', 'ecg-v1', 'ecg l', 'ecgl', 'ecg ii'],
        'spo2': ['sao2', 'osat', 'o2sat', 'o2 sat', 'o2-sat', 'o2-saturation'],
        'e2-m1': ['e2-m2', 'loc', 'eog(r)', 'eog-r', 'eog r', 'eog2', 'rt eye (e2)', 'rt. eye (e2)', 'rt. eye (e2)', 'lt. eye (e1)', 'eog loc-a2'],
        'e1-m2': ['e1-m1', 'roc', 'eog(l)', 'eog-l', 'eog l', 'eog1', 'lt eye (e1)', 'lt. eye (e1)', 'lt. eye (e1)', 'rt. eye (e2)', 'eog roc-a2'],
        'hr': ['heart rate', 'heartrate', 'h.r.'],
        'f3-m2': ['f3m2', 'f3-m2+m1', 'eeg f3-a2'],
        'f4-m1': ['f4m1', 'f4-m1+m2', 'eeg f4-a1'],
        'c3-m2': ['c3m2', 'c3-m2+m1', 'eeg c3-a2'],
        'c4-m1': ['c4m1', 'c4-m1+m2', 'eeg c4-a1'],
        'o1-m2': ['o1m2', 'o1-m2+m1', 'eeg o1-a2'],
        'o2-m1': ['o2m1', 'o2-m1+m2', 'eeg o2-a1'],
        'chin1-chin2': ['chin1chin2', 'chin', 'chin_emg', 'emg.subm', 'emg', 'emg1', 'emg2', 'chin emg', 'chin_emg1', 'chin_emg2', 'chin emg 2', 'chin emg 2', 'emg chin'],
        'chin 1': ['l chin', 'chin1'],
        'chin 2': ['r chin', 'chin2'],
        'abd': ['abdomen', 'abdominal', 'abd res', 'abdo res', 'abdo', 'effort abd'],
        'chest': ['thorax', 'thoracic', 'thor res', 'thoracic res', 'chest res', 'thor', 'effort tho'],
        'ptaf': ['npt', 'nptaf', 'pap', 'pap flow', 'papflow', 'nasal_pressure', 'nasal', 'cannula_flow'],
        'airflow': ['flow', 'air flow', 'thermal airflow', 'airflow thermal'],
        'cpres': ['c press', 'c_press', 'cpap pressure', 'cpap', 'cpap_press'],
        'm1': ['a1'],
        'm2': ['a2'],
        'position': ['pos', 'pos.'],
    }
    
    for new_name, old_names in rename_rules.items():
        for old_name in old_names:
            if old_name in signal.columns and new_name not in signal.columns:
                signal.rename(columns={old_name: new_name}, inplace=True)

    if ('pulse' in signal.columns) and ('hr' not in signal.columns):
        signal.rename(columns={'pulse': 'hr'}, inplace=True)
    if ('pr' in signal.columns) and ('hr' not in signal.columns):
        signal.rename(columns={'pr': 'hr'}, inplace=True)
        
    # if 'm1' and 'm2' are in the channels, there's a chance that the other channels (as above) are not yet referenced to them.
    # do the reference (subtraction) here:
    if 'm1' in signal.columns:
        # check the typical '-m1' referenced channels, as above:
        for ch in ['e2', 'f4', 'c4', 'o2']:
            if (ch in signal.columns) & (f'{ch}-m1' not in signal.columns):
                signal[f'{ch}-m1'] = signal[ch] - signal['m1']
    if 'm2' in signal.columns:
        # check the typical '-m2' referenced channels, as above:
        for ch in ['e1', 'f3', 'c3', 'o1']:
            if (ch in signal.columns) & (f'{ch}-m2' not in signal.columns):
                signal[f'{ch}-m2'] = signal[ch] - signal['m2']
                
    # remove any duplicate channels:
    signal = signal.loc[:, ~signal.columns.duplicated()]
    
    eeg_channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
    # if none of those are present, sometimes there is "eeg", which will be c4-m1 and "eeg(sec)" which will be c3-m2:
    if all(ch not in signal.columns for ch in eeg_channels):
        if 'eeg' in signal.columns:
            signal['c4-m1'] = signal['eeg']
        if 'eeg(sec)' in signal.columns:
            signal['c3-m2'] = signal['eeg(sec)']
        if 'eeg1' in signal.columns:
            signal['c4-m1'] = signal['eeg1']
        if 'eeg2' in signal.columns:
            signal['c3-m2'] = signal['eeg2']
        if 'eeg3' in signal.columns:
            signal['c4-m1'] = signal['eeg3']
        # use central channel if only available:
        if 'cz-oz' in signal.columns:
            signal['c4-m1'] = signal['cz-oz']
            
    eog_channels = ['e1-m2', 'e2-m1']
    if all(ch not in signal.columns for ch in ['m1', 'm2']):
        if (not 'e1-m2' in signal.columns) and ('e1' in signal.columns):
            signal.rename(columns={'e1': 'e1-m2'}, inplace=True)
        if (not 'e2-m1' in signal.columns) and ('e2' in signal.columns):
            signal.rename(columns={'e2': 'e2-m1'}, inplace=True)

    return signal


def check_channels_present(signal, verbose=False):
    """ 
    Check if all needed channels are present in the signal data and create derived channels if necessary.
    """
    
    # respiratory channels are not always present, it depends on the type of sleep study.
    # 'cpres' (CPAP pressure), 'ptaf' (PAP flow), 'airflow' (nasal airflow), 'cflow' (cannula flow)
    # if not present, set to 0 array.
    ch_present = signal.columns
    
    ch_respiratory = ['cpres', 'ptaf', 'airflow', 'cflow']
    # if none of these exist, at least 'abd' and 'chest' need to be there:
    assert 'abd' in ch_present and 'chest' in ch_present, \
        f"Not sufficient respiratory channels found: At least 'abd' and 'chest' need to be present in the file. Channels present: {list(ch_present)}"
    
    # set the missing ones to 0 array
    for ch in ch_respiratory:
        if ch not in ch_present:
            if verbose:
                print(f"Channel {ch} not present, setting to 0 array. Channel present: {list(ch_present)}")
            signal[ch] = 0
        
    if 'cpap_on' not in signal.columns:
        signal['cpap_on'] = determine_cpap_start(signal['cpres'].values, fs=200)
        
    ch_present = signal.columns
    
    # EEG: Create derived channels if necessary
    
    for ch in ['f3', 'c3', 'o1', 'e1']:
        if (ch in ch_present) and ('m2' in ch_present) and (f'{ch}-m2' not in ch_present):
            signal[f'{ch}-m2'] = signal[ch] - signal['m2']
    for ch in ['f4', 'c4', 'o2', 'e2']:
        if (ch in ch_present) and ('m1' in ch_present) and (f'{ch}-m1' not in ch_present):
            signal[f'{ch}-m1'] = signal[ch] - signal['m1']
        
    eeg_channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1']
    dict_contralateral = {
        'f3-m2': 'f4-m1',
        'f4-m1': 'f3-m2',
        'c3-m2': 'c4-m1',
        'c4-m1': 'c3-m2',
        'o1-m2': 'o2-m1',
        'o2-m1': 'o1-m2',
        'e2-m1': 'e1-m2',
        'e1-m2': 'e2-m1',
    }
    
    # at least one needs to be present. If not, raise an error. 
    if all(ch not in signal.columns for ch in eeg_channels):
        if 'eeg' in signal.columns:
            for ch in eeg_channels:
                signal[ch] = signal['eeg']
        else:
            raise ValueError(f"No EEG channels found. At least one of the following channels needs to be present: {eeg_channels}. Channels present: {list(ch_present)}")
    else:
        # if not all are present, set the missing ones to the contralateral channel or the first available EEG channel
        eeg_channels_present = [ch for ch in eeg_channels if ch in signal.columns]
        for ch in eeg_channels:
            if ch not in eeg_channels_present:
                ch_contralateral = dict_contralateral[ch]
                if ch_contralateral in signal.columns:
                    signal[ch] = signal[ch_contralateral]
                else:
                    signal[ch] = signal[eeg_channels_present[0]]
                
    # same for EOG:
    eog_channels = ['e1-m2', 'e2-m1']
    if all(ch not in signal.columns for ch in eog_channels):
        raise ValueError(f"No EOG channels found. At least one of the following channels needs to be present: {eog_channels}. Channels present: {list(ch_present)}")
    else:
        eog_channels_present = [ch for ch in eog_channels if ch in signal.columns]
        for ch in eog_channels:
            if ch not in eog_channels_present:
                ch_contralateral = dict_contralateral[ch]
                if ch_contralateral in signal.columns:
                    signal[ch] = signal[ch_contralateral]
                else:
                    signal[ch] = signal[eog_channels_present[0]]
    
    # chin EMG:
    if ('chin 1' in signal.columns) and ('chin 2' in signal.columns) and ('chin1-chin2' not in signal.columns):
        signal['chin1-chin2'] = signal['chin 1'] - signal['chin 2']
        
    # LIMB: Create derived channels if necessary
    ch_present = signal.columns
    if 'rleg+' in ch_present and 'rleg-' in ch_present and 'rat' not in ch_present:
        signal['rat'] = signal['rleg+'] - signal['rleg-']
    if 'lleg+' in ch_present and 'lleg-' in ch_present and 'lat' not in ch_present:
        signal['lat'] = signal['lleg+'] - signal['lleg-']
    # if still not present, set to 0 array
    if 'rat' not in ch_present:
        signal['rat'] = 0
    if 'lat' not in ch_present:
        signal['lat'] = 0
        
    if 'hr' in signal.columns:
        pass
    else:
        signal['hr'] = compute_hr_from_ecg(signal['ecg'].values, fs=200)

    ch_needed = [
        'f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1',
        'e1-m2', 'e2-m1', 'chin1-chin2', 'abd', 'chest', 'spo2', 'ecg', 'lat', 'rat',
    ]
    
    ch_optional = [
         'airflow', 'ptaf', 'cflow', 'cpres', 'hr', 'position', 'cpap_on',
    ]
    
    # Ensure all needed channels are present
    all_channels_avail = all(ch in signal.columns for ch in ch_needed)
    if not all_channels_avail:
        if verbose:
            print(f"Missing channels: {set(ch_needed) - set(signal.columns)}")
            print(f"Available channels: {signal.columns}")
        raise ValueError(f"Not all channels are available, missing: {set(ch_needed) - set(signal.columns)}. Available channels: {signal.columns}")

    ch_optional_available = [ch for ch in ch_optional if ch in signal.columns]
    
    signal = signal[ch_needed + ch_optional_available]

    return signal


def resample_signals(signal, fs, target_fs):
    """ Resamples the signals to the target frequency (200 Hz). """
    
    spo2_original = signal['spo2'].values
    signal_resampled = pd.DataFrame(resample_poly(signal, target_fs, fs, axis=0), columns=signal.columns)
    ## spo2 better be resampled by linear interpolation
    signal_resampled['spo2'] = np.interp(np.arange(0, len(signal_resampled), 1), np.arange(0, len(spo2_original), 1), spo2_original)
    
    return signal_resampled

def resample_annotations(annotations, fs, target_fs, signal_len):
    """ Resamples the annotations to the target frequency (200 Hz). """
    idx_resample = np.linspace(0, len(annotations['stage']) - 1, signal_len).astype(int)
    for key in annotations.columns:
        annotations[key] = annotations[key].iloc[idx_resample].reset_index(drop=True)
    return annotations


def scale_signal_to_voltage(signal):
    """ Scales specified channels from uV to V. """
    scale_channels = ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 
                      'e1-m2', 'e2-m1', 'chin1-chin2', 'lat', 'rat', 'ecg']
    for channel in scale_channels:
        signal[channel] = signal[channel] / 1e6
    
    return signal


# Helper function for band-pass filtering (e.g., focusing on 0.5-30Hz)
def bandpass_filter(signal, lowcut=0.5, highcut=30, fs=200, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def detect_signal_unit(signal, fs=200, window_length_sec=60, noise_threshold=500):
    # Step 1: Filter the signal to remove noise outside the EEG band
    # remove first 10% and last 10% of the signal:
    signal = signal[int(len(signal) * 0.1):int(len(signal) * 0.9)]
    filtered_signal = bandpass_filter(signal, fs=fs)
    
    # Step 2: Split the signal into windows
    window_length = window_length_sec * fs  # Convert seconds to samples
    num_windows = len(filtered_signal) // window_length
    
    valid_amplitudes = []

    for i in range(num_windows):
        window = filtered_signal[i * window_length : (i + 1) * window_length]
        
        # Step 3: Compute the absolute max amplitude for each window
        max_amplitude = np.max(np.abs(window))
        
        # Step 4: Ignore windows dominated by noise (e.g., extreme amplitudes)
        if max_amplitude < noise_threshold:  # Noise threshold to exclude extreme outliers
            valid_amplitudes.append(max_amplitude)
    
    # Step 5: Use robust statistics (e.g., 95th percentile) to determine the unit
    if len(valid_amplitudes) > 0:
        robust_amplitude = np.percentile(valid_amplitudes, 95)
    else:
        robust_amplitude = np.max(np.abs(filtered_signal))  # Fallback if no valid windows

    # Step 6: Decide based on the robust amplitude
    if robust_amplitude > 0.5:  # If greater than 0.5, signal likely in µV
        return 'uV'
    else:
        return 'V'


def save_prepared_data(path, signal, annotations=None):
    """ Saves the processed signal and annotations to an HDF5 file. """
    with h5py.File(path, 'w') as f:
        f.attrs['sampling_rate'] = 200
        f.attrs['unit_voltage'] = 'V'
        
        group_signals = f.create_group('signals')
        for name in signal.columns:
            group_signals.create_dataset(name, data=signal[name], shape=(len(signal), 1), 
                                         maxshape=(len(signal), 1), dtype='float32', compression="gzip")

        if annotations is not None:
            group_annotations = f.create_group('annotations')
            for name in annotations.columns:
                group_annotations.create_dataset(name, data=annotations[name], shape=(len(annotations), 1), 
                                                maxshape=(len(annotations), 1), dtype='float32', compression="gzip")
            

def process_file(path_input, path_output, add_existing_annotations=False, autoscale_signals=True):
    """ Process a single file and save the result to the output path.
    Note: This file is specifically designed for the BDSP Sleep Lab dataset.
    However, the functions can be modified to work with other datasets as well, the main points are:
    - Load the signal and annotations
    - Preprocess the signal and annotations:
    -- Standardize channel names
    -- Resample signals and annotations to the target frequency (200 Hz)
    -- Scale EEG/EOG to V from uV if necessary
    - Save the processed data to an HDF5 file.
    
    Args:
        path_input (str): The directory containing the input signal and annotation files.
        path_output (str): The path where the processed .h5 file will be saved.
    """
    signal_path = path_input + '.edf'
    
    
    # try:
    if 1:
        
        signal, params = load_bdsp_signal(signal_path)
        
        if add_existing_annotations:
            annot_path = signal_path.replace('_eeg.edf', '_annotations.csv')
            annotations = pd.read_csv(annot_path)
        else:
            annotations = None
            
        signal, annotations, results = preprocess_data(signal, annotations, params, autoscale_signals=autoscale_signals, verbose=False)

        save_prepared_data(path_output, signal, annotations)

    # except Exception as e:
    #     print(f"Error processing file {path_input}: {e}")


def process_files(path_dir_input, path_dir_output, autoscale_signals=True, overwrite=False, verbose=False):
    """ Process a range of files and save the results. """
    
    # Ensure the output directory exists
    if not os.path.exists(path_dir_output):
        os.makedirs(path_dir_output)
        print(f"Created output directory: {path_dir_output}")
    
    # List all .edf files in the input directory
    edf_files = [f for f in os.listdir(path_dir_input) if f.endswith('.edf')]
    print(f"Found {len(edf_files)} .edf files in input directory.")
    
    # Generate file IDs by stripping the .edf extension
    fileids = [f.replace('.edf', '') for f in edf_files]
    
    # check if any of those fileids are already processed (available as .h5 file)
    files_done = [fileid for fileid in fileids if os.path.exists(os.path.join(path_dir_output, fileid + '.h5'))]
    tag = ' (overwrite)' if overwrite else ''
    print(f"{len(files_done)} files already processed.{tag}")
    # Determine which files still need to be processed
    if overwrite:
        to_process = fileids
    else:
        to_process = [f for f in fileids if f not in files_done]
    print(f"{len(to_process)} files remaining to be processed.")

    # Iterate over the files to be processed
    for filename in tqdm(to_process, desc=f"Preparing files"):
        if verbose:
            print(f'Process study: {filename}')
        # Define the full input and output paths
        path_input = os.path.join(path_dir_input, filename)
        # if there is a '.' in the filename, replace with '_':
        if '.' in filename:
            print(f"Replacing filename: {filename} to {filename.replace('.', '_')}")
            filename = filename.replace('.', '_')
        path_output = os.path.join(path_dir_output, filename + '.h5')

        try:
            process_file(path_input, path_output, autoscale_signals=autoscale_signals)
            if verbose:
                print(f"Successfully processed and saved: {filename}.h5")
        except Exception as e:
            print(f"Error processing {filename}.edf: {e}. Skipping file.")

    print("Processing complete.")
