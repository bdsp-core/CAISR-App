import os
import numpy as np
import h5py
from DE_PSD import *


def graph_feat_extraction(segs, label, sig_tags, Fs, window, out_paths):    
    # the parameters to extract DE and PSD

    stft_para_eeg = {
        'stftn' : window*Fs,
        'fStart': [0.5, 2, 4,  6,  8, 11, 14, 22, 31],  # you can define different bands for your task
        'fEnd'  : [4,   6, 8, 11, 14, 22, 31, 40, 50],
        'fs'    : Fs,
        'window': window
    }

    stft_para_resp = {
        'stftn' : window*Fs,
        'fStart': [0,    0.05, 0.15, 0.3 ,  0.45, 0.6 , 0.75, 0.9, 1.2],
        'fEnd'  : [0.05, 0.15, 0.3 , 0.45,  0.6 , 0.75, 0.9 , 1.2, 1.6],
        'fs'    : Fs,
        'window': window
    }

    # compute PSD\DE
    n_bands = len(stft_para_eeg['fStart'])
    MYpsd = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)
    MYde  = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)

    # run over each segement
    for i in range(segs.shape[0]):
        # run over each channel
        for s, sig in enumerate(sig_tags):
            data = segs[i, s, :][np.newaxis]
            
            # EEG
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
                stft_para = stft_para_eeg
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)
            
            # RESPIRATION
            elif sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'ecg']:
                stft_para = stft_para_resp
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)


    mean = MYpsd.mean(axis=0)
    std = MYpsd.std(axis=0)
    MYpsd -= mean
    MYpsd /= std

    mean = MYde.mean(axis=0)
    std = MYde.std(axis=0)
    MYde -= mean
    MYde /= std

    return MYpsd, MYde , label

def graph_feat_extraction_unseen_docker(segs, sig_tags, Fs, window):    
    # the parameters to extract DE and PSD

    stft_para_eeg = {
        'stftn' : window*Fs,
        'fStart': [0.5, 2, 4,  6,  8, 11, 14, 22, 31],  # you can define different bands for your task
        'fEnd'  : [4,   6, 8, 11, 14, 22, 31, 40, 50],
        'fs'    : Fs,
        'window': window
    }

    stft_para_resp = {
        'stftn' : window*Fs,
        'fStart': [0,    0.05, 0.15, 0.3 ,  0.45, 0.6 , 0.75, 0.9, 1.2],
        'fEnd'  : [0.05, 0.15, 0.3 , 0.45,  0.6 , 0.75, 0.9 , 1.2, 1.6],
        'fs'    : Fs,
        'window': window
    }

    # compute PSD\DE
    n_bands = len(stft_para_eeg['fStart'])
    MYpsd = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)
    MYde  = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)

    # run over each segement
    for i in range(segs.shape[0]):
        # run over each channel
        for s, sig in enumerate(sig_tags):
            data = segs[i, s, :][np.newaxis]
            
            # EEG
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
                stft_para = stft_para_eeg
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)
            
            # RESPIRATION
            elif sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'ecg']:
                stft_para = stft_para_resp
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)


    mean = MYpsd.mean(axis=0)
    std = MYpsd.std(axis=0)
    MYpsd -= mean
    MYpsd /= std

    mean = MYde.mean(axis=0)
    std = MYde.std(axis=0)
    MYde -= mean
    MYde /= std

    return MYpsd, MYde

def graph_feat_extraction_mgh(segs, sig_tags, Fs, window, out_paths):    
    # the parameters to extract DE and PSD

    stft_para_eeg = {
        'stftn' : window*Fs,
        'fStart': [0.5, 2, 4,  6,  8, 11, 14, 22, 31],  # you can define different bands for your task
        'fEnd'  : [4,   6, 8, 11, 14, 22, 31, 40, 50],
        'fs'    : Fs,
        'window': window
    }

    stft_para_resp = {
        'stftn' : window*Fs,
        'fStart': [0,    0.05, 0.15, 0.3 ,  0.45, 0.6 , 0.75, 0.9, 1.2],
        'fEnd'  : [0.05, 0.15, 0.3 , 0.45,  0.6 , 0.75, 0.9 , 1.2, 1.6],
        'fs'    : Fs,
        'window': window
    }

    # compute PSD\DE
    n_bands = len(stft_para_eeg['fStart'])
    MYpsd = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)
    MYde  = np.zeros([segs.shape[0], segs.shape[1], n_bands], dtype=float)

    # run over each segement
    for i in range(segs.shape[0]):
        # run over each channel
        for s, sig in enumerate(sig_tags):
            data = segs[i, s, :][np.newaxis]
            
            # EEG
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2']:
                stft_para = stft_para_eeg
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)
            
            # RESPIRATION
            elif sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow', 'ecg']:
                stft_para = stft_para_resp
                MYpsd[i, s, :], MYde[i, s, :] = DE_PSD(data, stft_para)


    mean = MYpsd.mean(axis=0)
    std = MYpsd.std(axis=0)
    MYpsd -= mean
    MYpsd /= std

    mean = MYde.mean(axis=0)
    std = MYde.std(axis=0)
    MYde -= mean
    MYde /= std

    return MYpsd, MYde
