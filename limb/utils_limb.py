import numpy as np
import scipy.io as sio
import pandas as pd
from mne.filter import filter_data, notch_filter

def clip_noisy_values(psg, sample_rate, period_length_sec,
                      min_max_times_global_iqr=20):
    """
    Clips all values that are larger or smaller than +- min_max_times_global_iqr
    times to IQR of the whole channel.
    Args:
        psg:                      A ndarray of shape [N, C] of PSG data
        sample_rate:              The sample rate of data in the PSG
        period_length_sec:        The length of one epoch/period/segment in
                                  seconds
        min_max_times_global_iqr: Extreme value threshold; number of times a
                                  value in a channel must exceed the global IQR
                                  for that channel for it to be termed an
                                  outlier (in neg. or pos. direction).
    Returns:
        PSG, ndarray of shape [N, C]
        A list of lists, one sub-list for each channel, each storing indices
        of all epochs in which one or more values were clipped.
    """
    n_channels = psg.shape[-1]
    chan_inds = []
    for chan in range(n_channels):
        chan_psg = psg[..., chan]

        # Compute global IQR
        iqr = np.subtract(*np.percentile(chan_psg, [75, 25]))
        threshold = iqr * min_max_times_global_iqr

        # Reshape PSG to periods on 0th axis
        n_periods = int(chan_psg.shape[0]/(sample_rate*period_length_sec))
        temp_psg = chan_psg.reshape(n_periods, -1)

        # Compute IQR for all epochs
        inds = np.unique(np.where(np.abs(temp_psg) > threshold)[0])
        chan_inds.append(inds)

        # Zero out noisy epochs in the particular channel
        psg[:, chan] = np.clip(chan_psg, -threshold, threshold)
    return psg, chan_inds

def plm_preprocessing(emg):

    # processing variables
    notch_freq = 60.            # [Hz]
    bandpass_freq = [0.01, 75]  # [Hz]
    highpass_freq = 10          # Hz for EMG
    epoch_time = 30             # [min]
    epoch_step_time = 30        # [min]
    std_thres = 0.2             # [uV]
    std_thres2 = 1.             # [uV]
    flat_seconds = 5            # [s]
    amplitude_thres = 500       # [uV]
    Fs = 200

    # set signals
    # ecg = ecg.reshape((-1,1))
    # ecg = ecg.T
    # Remove DC bias
    center_emg = np.mean(emg, axis=1)
    emg_signal = (np.transpose(emg) - center_emg)
    emg_signal = np.transpose(emg_signal).astype(float)
    
    # EMG filtering: Notch Filter and High-pass filtering
    emg_signal = notch_filter(emg_signal, Fs, notch_freq, verbose=False)   # time x channel
    emg_signal = filter_data(emg_signal, Fs, highpass_freq, None, verbose=False)
     
    return emg_signal

def reshape_array(input_array, window_size):
    # Calculate the number of rows needed
    num_rows = len(input_array) // window_size

    # Trim the array to the nearest multiple of window_size
    trimmed_array = input_array[:num_rows * window_size]

    # Reshape the array
    reshaped_array = np.reshape(trimmed_array, (num_rows, window_size))

    return reshaped_array

def merge_nearby_events(event_mat_in, min_samples=100):
    num_events_in = event_mat_in.shape[0]
    merged_from_indices = np.zeros(num_events_in, dtype=bool)
    merged_to_indices = np.arange(0, num_events_in)

    if not event_mat_in.size == 0:
        merged_events = np.zeros_like(event_mat_in)
        num_events_out = 0
        merged_events[num_events_out, :] = event_mat_in[0, :]

        for k in range(1, num_events_in):
            if event_mat_in[k, 0] - merged_events[num_events_out, 1] < min_samples:
                merged_events[num_events_out, 1] = max(merged_events[num_events_out, 1], event_mat_in[k, 1])
                merged_from_indices[k] = True
            else:
                num_events_out += 1
                merged_events[num_events_out, :] = event_mat_in[k, :]
            merged_to_indices[k] = num_events_out

        merged_events = merged_events[:num_events_out + 1, :]
    else:
        merged_events = event_mat_in

    return merged_events, merged_from_indices, merged_to_indices

def _check_keys( dict):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for key in dict:
        if isinstance(dict[key], sio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, sio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def loadmat_(filename):
    """
    this function should be called instead of direct scipy.io .loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    data = sio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)