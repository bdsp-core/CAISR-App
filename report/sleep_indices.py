import pandas as pd
import numpy as np

# COLLECTION OF CODE TO COMPUTE COMMON SLEEP INDICES.
# All of them assume common "mgh" format, as code computes in mgh_sleeplab.py, i.e. vectorized annotations.

def compute_ahi(resp, hours_sleep=None, stage=None, fs=None):
    """
    Input: 
    respiratory annotation vector
    Either hours_of_sleep integer or stage array with fs
    Output:
    AHI, RDI, obstructive apneas/hour of sleep, central a/hour,
    mixed a/hour, hypopnea/hour, rera/hour
    """
    
    assert (hours_sleep is not None) | (stage is not None)
    
    if hours_sleep is not None:
            
        resp_binary = np.isin(resp, [1, 2, 3, 4]).astype(int)
        ahi = np.sum(np.diff(resp_binary) == 1) / hours_sleep
    
        rdi_binary = np.isin(resp, [1, 2, 3, 4, 5]).astype(int)
        rdi = np.sum(np.diff(rdi_binary) == 1) / hours_sleep
        
        oa_binary = (resp == 1).astype(int)
        oai = np.sum(np.diff(oa_binary) == 1) / hours_sleep
        
        ca_binary = (resp == 2).astype(int)
        cai = np.sum(np.diff(ca_binary) == 1) / hours_sleep
        
        ma_binary = (resp == 3).astype(int)
        mai = np.sum(np.diff(ma_binary) == 1) / hours_sleep
        
        hy_binary = (resp == 4).astype(int)
        hyi = np.sum(np.diff(hy_binary) == 1) / hours_sleep
        
        rera_binary = (resp == 5).astype(int)
        rerai = np.sum(np.diff(rera_binary) == 1) / hours_sleep
        
        ahi_nrem, ahi_rem = np.nan, np.nan
        
    elif stage is not None:
        hours_sleep = np.sum(np.isin(stage, [1, 2, 3, 4])) / fs / 3600
        if hours_sleep == 0:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
        sleep = np.isin(stage, [1, 2, 3, 4])
        resp_sleep = resp.copy()
        resp_sleep[~sleep] = 0
        
        resp_binary = np.isin(resp_sleep, [1, 2, 3, 4]).astype(int)
        events = np.sum(np.diff(resp_binary) == 1)
        ahi = np.sum(np.diff(resp_binary) == 1) / hours_sleep
    
        rdi_binary = np.isin(resp_sleep, [1, 2, 3, 4, 5]).astype(int)
        rdi = np.sum(np.diff(rdi_binary) == 1) / hours_sleep
        
        oa_binary = (resp_sleep == 1).astype(int)
        oai = np.sum(np.diff(oa_binary) == 1) / hours_sleep
        
        ca_binary = (resp_sleep == 2).astype(int)
        cai = np.sum(np.diff(ca_binary) == 1) / hours_sleep
        
        ma_binary = (resp_sleep == 3).astype(int)
        mai = np.sum(np.diff(ma_binary) == 1) / hours_sleep
        
        hy_binary = (resp_sleep == 4).astype(int)
        hyi = np.sum(np.diff(hy_binary) == 1) / hours_sleep
        
        rera_binary = (resp_sleep == 5).astype(int)
        rerai = np.sum(np.diff(rera_binary) == 1) / hours_sleep
        
        sleep_nrem = np.isin(stage, [1, 2, 3])
        hours_nrem = np.sum(sleep_nrem) / fs / 3600
        if hours_nrem > 0:
            resp_nrem = resp.copy()
            resp_nrem[~sleep_nrem] = 0
            resp_nrem_binary = np.isin(resp_nrem, [1, 2, 3, 4]).astype(int)
            events_nrem = np.sum(np.diff(resp_nrem_binary) == 1)
            ahi_nrem = events_nrem / hours_nrem
        else:
            ahi_nrem = np.nan
        
        sleep_rem = np.isin(stage, [4])
        hours_rem = np.sum(sleep_rem) / fs / 3600
        if hours_rem > 0:
            resp_rem = resp.copy()
            resp_rem[~sleep_rem] = 0
            events_rem = np.sum(np.diff(np.isin(resp_rem, [1, 2, 3, 4]) == 1))
            ahi_rem = events_rem / hours_rem
        else:
            ahi_rem = np.nan

    ahi = np.round(ahi, 2)
    rdi = np.round(rdi, 2)
    oai = np.round(oai, 2)
    cai = np.round(cai, 2)
    mai = np.round(mai, 2)
    hyi = np.round(hyi, 2)
    rerai = np.round(rerai, 2)
    ahi_nrem = np.round(ahi_nrem, 2)
    ahi_rem = np.round(ahi_rem, 2)
    
    return ahi, rdi, oai, cai, mai, hyi, rerai, ahi_nrem, ahi_rem
        

def compute_plmi(limb, hours_sleep=None, stage=None, fs=None):
    """
    Input: 
    limb movements annotation vector
    Either hours_of_sleep integer or stage array with fs
    Output:
    PLMI, LMI
    """
    
    assert (hours_sleep is not None) | (stage is not None)
    if hours_sleep is None:
        hours_sleep = np.sum(np.isin(stage, [0, 1, 2, 3, 4])) / fs / 3600
     
    lm_binary = np.isin(limb, [1, 2, 4]).astype(int) # both isolated and Periodic
    lmi = np.sum(np.diff(lm_binary) == 1) / hours_sleep
 
    plm_binary = np.isin(limb, [2]).astype(int) # only periodic
    plmi = np.sum(np.diff(plm_binary) == 1) / hours_sleep
 
    return lmi, plmi


def compute_arousal_index(arousal, hours_sleep=None, stage=None, fs=None):
    """
    Input: 
    arousal annotation vector
    Either hours_of_sleep integer or stage array with fs
    Output:
    Arousal index = number of arousals per hour of sleep
    """
    
    assert (hours_sleep is not None) | (stage is not None)
    if hours_sleep is None:
        hours_sleep = np.sum(np.isin(stage, [0, 1, 2, 3, 4])) / fs / 3600
     
    arousal_index = np.sum(np.diff(arousal) == 1) / hours_sleep
  
    return arousal_index


def compute_sfi(stage, fs):
    """
    Input: 
    sleep stages annotation vector
    fs: sampling frequency
    Ouput: sleep fragmentation index (# of shifts to N1,W from N2, N3, R per hour of sleep)
    """
    
    hours_sleep = np.sum(np.isin(stage, [0, 1, 2, 3, 4])) / fs / 3600
        
    w_or_n1 = np.isin(stage, [5, 3]).astype(int)
    deep_sleep = np.isin(stage, [1, 2, 4]).astype(int)
    fragmentation_shift = np.logical_and(w_or_n1[1:], deep_sleep[:-1]).astype(int)
    fragmentation_pos = np.where(fragmentation_shift)[0]
    sfi = np.round(len(fragmentation_pos) / hours_sleep, 1)
    
    return sfi


def sleep_indices_stages(stage, fs):
    """
    Input: 
    sleep stages annotation vector
    fs: sampling frequency
    Ouput: list of sleep indices: [hours_sleep, hours_psg, sleep_efficiency, perc_r, perc_n1, perc_n2, perc_n3, waso, sleep_latency, r_latency]
    """
    # compute indices   
    hours_sleep = np.sum(np.isin(stage, [1, 2, 3, 4])) / fs / 3600
    hours_psg = np.sum(np.isin(stage, [1, 2, 3, 4, 5])) / fs / 3600
    sleep_efficiency = np.round(hours_sleep / hours_psg, 3)

    perc_r = np.round(np.sum(np.isin(stage, [4])) / fs / 3600 / hours_sleep, 3)
    perc_n1 = np.round(np.sum(np.isin(stage, [3])) / fs / 3600 / hours_sleep, 3)
    perc_n2 = np.round(np.sum(np.isin(stage, [2])) / fs / 3600 / hours_sleep, 3)
    perc_n3 = np.round(np.sum(np.isin(stage, [1])) / fs / 3600 / hours_sleep, 3)

    first_sleep = np.where(np.isin(stage, [1, 2, 3, 4]))[0]
    if len(first_sleep) > 0:
        first_sleep = first_sleep[0]
        stage_aso = stage[first_sleep:]
        waso = np.round(np.sum(np.isin(stage_aso, [5])) / fs / 60, 1) # WASO in minutes
    else:
        waso = np.nan
    stage_psg_only = stage[np.where(np.isin(stage, [1, 2, 3, 4, 5]))[0]]
    first_sleep = np.where(np.isin(stage_psg_only, [1, 2, 3, 4]))[0]
    if len(first_sleep) > 0:
        first_sleep = first_sleep[0]
        sleep_latency = np.round(first_sleep / fs / 60, 1) # in minutes
    else:
        sleep_latency = np.nan
    first_rem = np.where(np.isin(stage_psg_only, [4]))[0]
    if len(first_rem) > 0:
        first_rem = first_rem[0]
        r_latency = np.round(first_rem / fs / 60, 1)
    else:
        r_latency = np.nan

    return [hours_sleep, hours_psg, sleep_efficiency, perc_r, perc_n1, perc_n2, perc_n3, waso, sleep_latency, r_latency]