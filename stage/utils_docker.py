import sys
# sys.path.insert(1, '../../prepared_data')
# from load_data import get_cohort_channels

import numpy as np
import pandas as pd
import os
import h5py
import sys
import warnings
import matplotlib.pyplot as plt
from shutil import copyfile, rmtree
import mne

if not sys.warnoptions:
    warnings.simplefilter("ignore")

from collections import Counter
from sklearn.preprocessing import RobustScaler

def load_prepared_data(file_path:str, get_signals:list=[]):
    """ 
    load prepared data format (post March 22, 2023). Currently, all signals are read. 
    Inputs:
    file_path: path to prepared .h5 file.
    """
    # init DF
    Xy = pd.DataFrame([])
    
    # read file
    with h5py.File(file_path, 'r') as f:
        # Loop over each group and dataset within each group and get data
        group_names = list(f.keys())
        for group_name in group_names:        
            group = f[group_name]
            # list all existing datasets
            dataset_names = list(group.keys())
            # get all, or take from get_signals
            get_names = dataset_names if len(get_signals)==0 else [c for c in dataset_names if c in get_signals]
            for dataset_name in get_names:
                dataset = group[dataset_name][:]
                assert dataset.shape[1] == 1, "Only one-dimensional datasets expected"
                Xy[dataset_name] = dataset.flatten()
        
        # save attributes
        params = {}
        for attrs in f.attrs.keys():
            params[attrs] = f.attrs[attrs]
    return Xy, params


def set_out_paths(save_folder, dataset):
    
    ### GRAPH ###
    graph_out_1 = save_folder + 'graphsleepnet/features/' + dataset + '/'
    graph_out_2 = save_folder + 'graphsleepnet/labels/' + dataset + '/'
    graph_out_paths = [graph_out_1, graph_out_2]
    for out_path in graph_out_paths:
        os.makedirs(out_path, exist_ok=True)

    return graph_out_paths

def do_initial_preprocessing(signals, new_Fs, original_Fs):
    from mne.filter import filter_data, notch_filter
    from scipy.signal import resample_poly
    notch_freq_us = 60.                 # [Hz]
    notch_freq_eur = 50.                # [Hz]
    bandpass_freq_eeg = [0.1, 35]       # [Hz] [0.5, 40]
    bandpass_freq_airflow = [0., 10]    # [Hz]
    bandpass_freq_ecg = [0.3, 40]     # [Hz]

    # setup new signal DF
    new_df = pd.DataFrame([], columns=signals.columns)

    for sig in signals.columns:
        # 1. Notch filter
        image = signals[sig].values
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2',
                   'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
            image = notch_filter(image, original_Fs,
                                 notch_freq_us, verbose=False)
            # image = notch_filter(image, 200, notch_freq_eur, verbose=False)

        # 2. Bandpass filter
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
            image = filter_data(
                image, original_Fs, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace']:
            image = filter_data(
                image, original_Fs, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
        if sig == 'ecg':
            image = filter_data(
                image, original_Fs, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # 3. Resample data
        if new_Fs != original_Fs:
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2',
                       'abd', 'chest', 'airflow', 'ptaf', 'cflow', 'breathing_trace', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = np.repeat(image, new_Fs)
                image = image[::original_Fs]

        # 4. Insert in new DataFrame
        new_df.loc[:, sig] = image

    del signals
    return new_df

def clip_normalize_signals(signals, sample_rate, br_trace, split_loc, min_max_times_global_iqr=20):
    # run over all channels
    channels = signals.columns.tolist()
    stage = 'stage'
    st = [item for item in  channels if stage in item]
    resp = 'resp'
    re = [item for item in  channels if resp in item]
    arousal = 'arousal'
    ar = [item for item in  channels if arousal in item]
    skiped_columns = ar + st + re
    for chan in signals.columns:
        # import pdb; pdb.set_trace()
        # print(chan)
        # skip labels
        # skiped_columns = ['resp-h3_converted_0', 'resp-h4_expert_0','stage_expert_0', 'arousal-shifted_converted_0', 'arousal_expert_0']
        skipped_channels = skiped_columns + ['cpap_pressure', 'cpap_on']
        if np.any([t in chan for t in skipped_channels]):
            continue
        
        signal = signals.loc[:, chan].values
        # clips spo2 @60%
        if chan == 'spo2':
            signals.loc[:, chan] = np.clip(signal.round(), 60, 100)
            continue

        # for all EEG (&ECG) traces
        if chan in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'ecg']:
            # Compute global IQR
            iqr = np.subtract(*np.percentile(signal, [75, 25]))
            threshold = iqr * min_max_times_global_iqr

            # clip outliers
            signal_clipped = np.clip(signal, -threshold, threshold)

            # normalize channel
            sig = np.atleast_2d(signal_clipped).T
            transformer = RobustScaler().fit(sig)
            signal_normalized = np.squeeze(transformer.transform(sig).T)

        # for all breathing traces
        elif chan in ['abd', 'chest', 'airflow', 'ptaf', 'cflow']:
            # import pdb; pdb.set_trace()
            if np.all(signal == 0):
                continue
            region = np.arange(len(signal))
            # cut split-night recordings and do only local nomralization
            if split_loc is not None:
                replacement_signal = np.empty(len(signals)) * np.nan
                if chan == br_trace[0]:
                    region = region[:split_loc]
                elif chan == br_trace[1]:
                    region = region[split_loc:]
                replacement_signal[region] = signal[region]
                signal = replacement_signal
            # import pdb; pdb.set_trace()
            # normalize signal
            signal_clipped = np.clip(signal, np.nanpercentile(
                signal[region], 5), np.nanpercentile(signal[region], 95))
            signal_normalized = np.array(
                (signal - np.nanmean(signal_clipped)) / np.nanstd(signal_clipped))
            
            # clip extreme values
            thresh = np.mean((np.abs(np.nanquantile(signal_normalized[region], 0.0001)), np.nanquantile(
                signal_normalized[region], 0.9999)))
            # import pdb; pdb.set_trace()
            if region[0] == 0:
                signal_normalized[np.concatenate(
                    [signal_normalized[region] < -thresh, np.full(len(signal)-len(region), False)])] = -thresh
                signal_normalized[np.concatenate([signal_normalized[region] > thresh, np.full(
                    len(signal)-len(region), False)])] = -thresh
            else:
                signal_normalized[np.concatenate([np.full(
                    len(signal)-len(region), False), signal_normalized[region] < -thresh])] = -thresh
                signal_normalized[np.concatenate([np.full(
                    len(signal)-len(region), False), signal_normalized[region] > thresh])] = thresh
            # import pdb; pdb.set_trace()
        # replace original signal
        signals.loc[:, chan] = signal_normalized
        # import pdb; pdb.set_trace()

    return signals

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

def select_signals_cohort(path):
    
    original_Fs = 200
    new_Fs = 100
    Fs = 100
    # 
    period_length_sec = 0.5 # sec
    
    data, params = load_prepared_data(path)
    data = data.astype(np.float64)
    channels = data.columns.tolist()
    common_sigs = ['e1-m2', 'chin1-chin2', 'abd', 'chest', 'ecg']
    eeg = ['c3-m2', 'c4-m1']
    # import pdb; pdb.set_trace()
    select_channels = eeg + common_sigs
    signals = data[select_channels]
    signals = do_initial_preprocessing(signals, new_Fs, original_Fs)
    length_to_match_30 = int(signals.shape[0]/(Fs * period_length_sec))
    length_to_match_30 = int(length_to_match_30 * Fs * period_length_sec)
    signals = signals.iloc[0:length_to_match_30,:]
    
    sigs = signals[eeg + common_sigs].values.T 
    sigs, chan_inds = clip_noisy_values(sigs.T, new_Fs, period_length_sec)
    transformer = RobustScaler().fit(sigs) 
   
    sigs = transformer.transform(sigs)
    sigs = sigs.T
    
    channles_selected = eeg + common_sigs
    return sigs, new_Fs, channles_selected, len(data)


def segment_data_unseen(signals):
    
    # create 30-sec epochs
    Fs = 100
    epoch_time = 30  
    epoch_size = int(round(epoch_time*Fs))
    epoch_inds = np.arange(0, signals.shape[1]-epoch_size+1, epoch_size)
    seg_ids = list(map(lambda x:np.arange(x,x+epoch_size), epoch_inds))
    # segment EEG plus data into 30-sec epochs
    segs = signals[:, seg_ids].transpose(1,0,2) # shape = (#epoch, #channel, 30*Fs)
    return segs