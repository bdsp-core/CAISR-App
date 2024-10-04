import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import numpy as np
from mne.filter import filter_data, notch_filter
from mne.time_frequency import psd_array_multitaper


def eeg_filter(eeg, Fs, notch_freq=60., bandpass_low=0.02, bandpass_high=60):
    """
    eeg filter
    """
    
    notch_freq = notch_freq  # [Hz]
    bandpass_freq = [bandpass_low, bandpass_high]  # [Hz]
    
    # filter EEG
    if notch_freq is not None:
        eeg = notch_filter(eeg, Fs, notch_freq, verbose=False)
    if bandpass_freq is not None:
        eeg = filter_data(eeg, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)

    return eeg


def filter_routine(signal, Fs, notch_freq=60., bandpass_low=0.02, bandpass_high=60):
    """
    filter routine, notch and bandpass
    """
    notch_freq = notch_freq  # [Hz]
    bandpass_freq = [bandpass_low, bandpass_high]  # [Hz]
    
    # filter EEG
    if notch_freq is not None:
        signal = notch_filter(signal, Fs, notch_freq, verbose=False)
    if (bandpass_freq is not None) & ((bandpass_low is not None) | (bandpass_high is not None)):
        signal = filter_data(signal, Fs, bandpass_freq[0], bandpass_freq[1], verbose=False)

    return signal



def spectrogram(signal, Fs, signaltype=None, epoch_time=30, epoch_step_time=30, decibel=True, fmin=0.02, fmax=60, bandwidth=None, adaptive=True, n_jobs=1):
    """
    Inputs:
    signal: 1d numpy array of signal (time domain)
    Fs: sampling frequency
    signaltype: keywords/shortcuts (see code below, selects bandwith based on keyword)
    epoch_time: window-length in seconds
    epoch_step_time: stepsize in seconds
    decibel: boolean, if result shall be return in decibel (default True)
    fmin: minimum frequency of interest
    fmax: maximum frequency of interest
    bandwidth: multi-taper bandwidth parameter
    adaptive: (see MNE description. True=more accurate but slow)
    n_jobs: parallel jobs.
    Returns:
    # specs.shape = (#epoch, #channel, #freq)
    # freq.shape = (#freq,)
    """
    
    if n_jobs == 'max':
        import multiprocessing
        n_jobs = max(multiprocessing.cpu_count() - 1, 1)

    # segment
    epoch_size = int(round(epoch_time*Fs))
    epoch_step = int(round(epoch_step_time*Fs))
    start_ids = np.arange(0, signal.shape[1]-epoch_size+epoch_step, epoch_step)
    seg_ids = list(map(lambda x: np.arange(x, x+epoch_size), start_ids))
    signal_segs = signal[:, seg_ids].transpose(1, 0, 2)  # signal_segs.shape=(#epoch, #channel, Tepoch)
    if 0:
        print(f'signal.shape = {signal.shape}')
        print(f'start_ids = {start_ids.shape}')
        print(f'signal_segs.shape = {signal_segs.shape}')
        print('start_ids', start_ids[:2], start_ids[-2:])
        print('seg_ids', seg_ids[:2], seg_ids[-2:])
        
    # compute spectrogram

    if bandwidth is None:
        if signaltype == 'eeg':       
            NW = 10.
            bandwidth = NW*2./epoch_time
        elif signaltype == 'resp_effort':
            NW = 1
            bandwidth = NW/epoch_time
        else:
            raise ValueError("Unexpected signaltype! ")

    # this is how half nbw is computed in code:
    n_times = signal_segs.shape[-1]
    half_nbw = float(bandwidth) * n_times / (2. * Fs)
    n_tapers_max = int(2 * half_nbw)
    # print(half_nbw)
    # print(n_tapers_max)
    specs, freq = psd_array_multitaper(signal_segs, Fs, fmin=fmin, fmax=fmax, adaptive=adaptive, low_bias=True, verbose='ERROR', bandwidth=bandwidth, normalization='full', n_jobs=n_jobs);

    specs[np.isnan(specs)] = 0
    
    if decibel:
        specs[specs == 0] = 1e-25 # avoid log(0)
        specs = 10*np.log10(specs)
    
    # if specs has nan, raise error:
    if np.any(np.isnan(specs)):
        raise ValueError('Spectrogram contains NaN')
    
    return specs, freq, signal_segs
    

