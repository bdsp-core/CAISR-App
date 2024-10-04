import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple

# set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import utils
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Bad_signal_check import bad_signal_check, remove_error_events
from Event_array_modifiers import *
# import additional algorithm functions
sys.path.insert(0, f'{RB_folder}/rule_based_functions/')
from Ventilation_envelope import *
from Ventilation_spectral_analysis import *
from Post_processing import *


### MAIN ###

def find_flow_reductions(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    This function identifies flow reductions from respiratory data traces.
    
    Args:
    - data: A pandas DataFrame containing respiratory data signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    
    Returns:
    - Updated DataFrame with identified flow reduction events across different traces.
    """

    # run over the two channels <breathing_trace> & <ABD+CHEST>
    for i, trace in enumerate(['br', 'eff', 'air']):
        # set ptaf/cflow
        if i == 0:
            drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.41, 0.80, 8, 7.5, 0.70, False
            if hdr['test_type'] == 'titration': drop_h = 0.38; drop_a = 0.80; dur_hyp = 9
            data['Ventilation_combined'] = data['breathing_trace']
            data['immidiate_remove_br'] = 0
            flat_std = 0.05
        # set ABD+CHEST
        elif i == 1:
            drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.48, 0.85, 7, 7, 0.65, True
            if hdr['test_type'] == 'titration': drop_h = 0.45; dur_hyp = 10
            abd = data['ABD'].rolling(int(0.5*hdr['newFs']), center=True).median().fillna(0)
            chest = data['CHEST'].rolling(int(0.5*hdr['newFs']), center=True).median().fillna(0) 
            data['ABD_CHEST'] = abd+chest
            data['Ventilation_combined'] = abd+chest
            data['immidiate_remove_eff'] = 0
            flat_std = 0.05
        # set airflow, if possible
        elif i == 2:
            drop_h, drop_a, dur_apnea, dur_hyp, quant, extra_smooth = 0.40, 0.85, 8, 7.5, 0.65, False
            # skip if no valid Airflow signals present (or used as primary br. trace)
            check_same = (sum(data['airflow']==data['breathing_trace'])==len(data))
            # check for bad airflow signal
            data['Ventilation_combined'] = data['airflow']
            flat_std = 0.07
            data = bad_signal_check(data, hdr['newFs'], flat_std)
            check_flat = sum(data[['bad_signal', 'flat_signal']].sum(1)>0) > 0.9*len(data)
            if ('airflow' not in data.columns) or all(data['airflow']==0) or check_same or check_flat: 
                data['flow_reductions_air'] = 0
                data['soft_ventilation_drop_apnea_air'] = 0
                data['ventilation_drop_soft_hyp_air'] = 0
                data['bad_signal_air'] = 1
                data['flat_signal_air'] = 0
                data['flat_airflow'] = 0
                continue

        # compute envelope and baseline on breathing trace
        data = compute_ventilation_envelopes(data, hdr['newFs'])

        # perform bad signal check, to be masked for event detection
        data = bad_signal_check(data, hdr['newFs'], flat_std)
        data[f'bad_signal_{trace}'] = data.bad_signal.values
        data[f'flat_signal_{trace}'] = data.flat_signal.values
        if trace == 'air' and hdr['test_type'] == 'titration':
            data['bad_signal_air'] = 1
        elif trace == 'air' and hdr['test_type'] == 'split-night':
            data.loc[hdr['cpap_start']:, 'bad_signal_air'] = 1
            
        # compute flow reductions
        if trace == 'br': plot = False
        if trace == 'air': plot = False
        if trace == 'eff': plot = False
        data = assess_ventilation(data, hdr, drop_hyp=drop_h, drop_apnea=drop_a, dur_apnea=dur_apnea, 
                                    dur_hyp=dur_hyp, quant=quant, extra_smooth=extra_smooth, plot=plot)
        
        # remove effort flow reductions from sum, if not found in either of the individual traces
        if trace == 'eff':
            data = remove_non_individual_effort_drops(data, hdr, drop_hyp=drop_h, drop_apnea=0.5, 
                                                dur_apnea=dur_apnea, dur_hyp=dur_hyp, quant=quant)
            # save backwards effort drops for RERA detection
            data['RERA_effort_drops'] = np.array(data['Ventilation_drop_apnea'] > 0).astype(int)
                                
        # remove found flow reductions, with no clear forward drop
        data = remove_non_forward_drops(data, hdr, drop_hyp=0.3, drop_apnea=0.5, dur_apnea=dur_apnea,
                                            dur_hyp=dur_hyp, quant=quant, extra_smooth=extra_smooth)

        # create flow reduction array
        data = combine_flow_reductions(data, hdr)

        # remove flat signal segments
        cols = ['flow_reductions', 'soft_ventilation_drop_apnea', 'Ventilation_drop_soft_hypopnea']
        data = remove_error_events(data, trace, columns=cols)

        if trace == 'air':
            data = spectral_assess_thermistor_hypopnea(data, hdr)

        # save flow reductions for this trace
        data[f'flow_reductions_{trace}'] = data.flow_reductions.values
        data[f'soft_ventilation_drop_apnea_{trace}'] = data.soft_ventilation_drop_apnea.values
        data[f'ventilation_drop_soft_hyp_{trace}'] = data.Ventilation_drop_soft_hypopnea.values

        # find central components in effort signal
        if trace == 'eff':
            data = find_central_effort_componenets(data, hdr, quant, plot=False)

        # find 'flat' (low std) airflow segments
        if trace == 'air':
            data = find_flat_thermistor_componenets(data, hdr)
            
    ########################################
    # # plot      breathing + Airflow + effort
    # data['patient_asleep'] = np.logical_and(data.Stage < 5, data.Stage > 0)
    # plt.plot(data.breathing_trace.mask(data.patient_asleep==0)+10, 'k', alpha=0.5)
    # plt.plot(data.breathing_trace.mask(data.patient_asleep==1)+10, 'k', alpha=0.1)
    # plt.plot(data.flat_signal_br.mask(data.flat_signal_br==0)*10, 'r')
    # plt.plot(data.bad_signal_br.mask(data.bad_signal_br==0)*9, 'y')
    # plt.plot(data.airflow.mask(data.patient_asleep==0)-2, 'b', alpha=0.4)
    # plt.plot(data.airflow.mask(data.patient_asleep==1)-2, 'b', alpha=0.1)
    # plt.plot(data.flat_signal_air.mask(data.flat_signal_air==0)-2, 'r')
    # plt.plot(data.bad_signal_air.mask(data.bad_signal_air==0)-3, 'y')
    # plt.plot(data.flat_airflow.mask(data.flat_airflow==0)-3.5, 'r') ##########
    # plt.plot(data.ABD_CHEST.mask(data.patient_asleep==0)-10, 'g', alpha=0.5)
    # plt.plot(data.ABD_CHEST.mask(data.patient_asleep==1)-10, 'g', alpha=0.1)
    # plt.plot(data.flat_signal_eff.mask(data.flat_signal_eff==0)-10, 'r')
    # plt.plot(data.bad_signal_eff.mask(data.bad_signal_eff==0)-11, 'y')
    # # plot      original labels
    # raters = ['resp-h3_expert_0', 'resp-h3_expert_1', 'resp-h3_expert_2'] if hdr['dataset'] in ['jedi', 'stanford'] else ['Apnea']
    # if 'resp-h3_platinum' in data.columns: raters = ['resp-h3_platinum']
    # for r, resp in enumerate(raters):
    #     plt.plot(data[resp].mask(data[resp]!=4) + 8 + 0.2*r, 'm*', lw=4)
    #     plt.plot(data[resp].mask(data[resp]!=1) + 11 + 0.2*r, 'k*', lw=4)
    #     plt.plot(data[resp].mask(data[resp]!=2) + 10 + 0.2*r, 'k*', lw=4)
    #     plt.plot(data[resp].mask(data[resp]!=3) + 9 + 0.2*r, 'k*', lw=4)
    #     plt.plot(data[resp].mask(data[resp]!=5) + 7 + 0.2*r, 'r*', lw=4)
    # # plot      br reductions
    # plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=1)+6, 'k', lw=2)
    # plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=2)+5, 'm', lw=2)
    # plt.plot(data.ventilation_drop_soft_hyp_br.mask(data.ventilation_drop_soft_hyp_br!=1)+5.5, 'c', lw=2)
    # # plt.plot(data.soft_ventilation_drop_apnea_br.mask(data.soft_ventilation_drop_apnea_br!=1)+5, 'c', lw=2)
    # # plot      air reductions
    # plt.plot(data.flow_reductions_air.mask(data.flow_reductions_air!=1)-4, 'k', lw=2)
    # plt.plot(data.flow_reductions_air.mask(data.flow_reductions_air!=2)-5, 'm', lw=2)
    # plt.plot(data.ventilation_drop_soft_hyp_air.mask(data.ventilation_drop_soft_hyp_air!=1)-5.5, 'c', lw=2)
    # # plt.plot(data.soft_ventilation_drop_apnea_air.mask(data.soft_ventilation_drop_apnea_air!=1)-5, 'c', lw=2)
    # # plot      effort reductions
    # plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=1)-12, 'k', lw=2)
    # plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=2)-13, 'm', lw=2)
    # plt.plot(data.ventilation_drop_soft_hyp_eff.mask(data.ventilation_drop_soft_hyp_eff!=1)-13.5, 'c', lw=2)
    # # plt.plot(data.soft_ventilation_drop_apnea_eff.mask(data.soft_ventilation_drop_apnea_eff!=1)-13, 'c', lw=2)
    # data['patient_asleep'] = 1
    ########################################
    
    ### do some post-processing ###
    # 1) split breathing hypopneas if multiple Effort events fit inside
    for trace in ['eff']:
        if not f'flow_reductions_{trace}' in data.columns: continue
        ref_reductions = find_events(data[f'flow_reductions_{trace}'] > 0)
        breathing_reductions = find_events(data['flow_reductions_br'] == 2)
        for b, (st, end) in enumerate(ref_reductions[:-1]):
            next_st = ref_reductions[b+1][0]
            next_end = ref_reductions[b+1][1]
            remain = [ev for ev in breathing_reductions if st>ev[0]-2*hdr['newFs'] and next_end<ev[1]+2*hdr['newFs']]
            if len(remain) > 0:
                e_type = data.loc[remain[0][0], 'flow_reductions_br']
                data.loc[remain[0][0]:remain[-1][-1], 'flow_reductions_br'] = 0
                data.loc[st:end, 'flow_reductions_br'] = e_type
                data.loc[next_st:next_end, 'flow_reductions_br'] = e_type

    # 2) convert apnea and hypopnea breathing reductions if..
    if True:
        # for apneas
        for st, end in find_events(data['flow_reductions_br'].values==1):
            # skip Titration phase 
            if hdr['test_type'] == 'titration' or (hdr['test_type'] == 'split-night' and st > hdr['cpap_start']): continue
            
            # NO Airflow hypopnea NOR Effort hypopnea drop --> remove apnea event; continue
            check1a = np.any(data.loc[st:end, 'flow_reductions_air'] > 0)
            check1b = np.any(data.loc[st:end, 'flow_reductions_eff'] > 0)
            flat_cols = ['flat_signal_air', 'flat_signal_eff', 'bad_signal_air', 'bad_signal_eff']
            notFlat = np.all(data.loc[st:end, flat_cols] == 0)
            if not check1a and not check1b and notFlat: 
                # if desat, set to hypopnea, else, remove
                desat = any(data.loc[st:end+10*hdr['newFs'], 'saturation_drop']==1)
                arousal = any(data.loc[st:end+10*hdr['newFs'], 'EEG_arousals']==1)
                if desat or arousal:
                    data.loc[st:end, 'flow_reductions_br'] = 2
                else:
                    data.loc[st:end, 'flow_reductions_br'] = 0
                continue
            
            # No Airflow apnea NOR Effort apnea drop --> convert apnea to hypopnea
            check2a = np.any(data.loc[st:end, 'flow_reductions_air'] == 1)
            check2b = np.any(data.loc[st:end, 'flow_reductions_eff'] == 1)
            if not check2a and not check2b: data.loc[st:end, 'flow_reductions_br'] = 2
        # for hypopneas
        for st, end in find_events(data['flow_reductions_br'].values==2):
            # Only convert Hypopneas into Apneas, if indicated by Airflow OR Effort; and duration > %
            check2a = sum(data.loc[st:end, 'flow_reductions_air'] == 1) > 0.1*(end-st)
            check2b = sum(data.loc[st:end, 'flow_reductions_air'] > 0) > 0.25*(end-st)
            check2ab = sum(data.loc[st:end, 'flat_airflow'] > 0) > 0.25*(end-st)
            check3a = sum(data.loc[st:end, 'flow_reductions_eff'] == 1) > 0.25*(end-st)
            check3b = sum(data.loc[st:end, 'flow_reductions_eff'] > 0) > 0.25*(end-st)

            # Disregard Airflow for Titration phase
            if hdr['test_type'] == 'titration' or (hdr['test_type'] == 'split-night' and st > hdr['cpap_start']):
                check2b = True

            if (check2a and check3b) or (check3a and check2b) or check2ab: 
                data.loc[st:end, 'flow_reductions_br'] = 1

    # 3) remove overlapping effort/airflow reductions
    for trace in ['eff', 'air']:
        for col in ['flow_reductions_', 'soft_ventilation_drop_apnea_']:
            if trace == 'eff':
                # ... by running over over all effort events
                for st, end in find_events(data[col+'eff'].values>0):
                    # remove effort flow reductions if already found in breathing trace
                    if np.any(data.loc[st:end, col+'br'] > 0): 
                        data.loc[st:end, col+'eff'] = 0
                    else:
                        # remove if right immediately after br apnea
                        check1 = np.any(data.loc[st-5*hdr['newFs']:st, 'flow_reductions_br'] == 1)
                        check2 = data.loc[st+(end-st)//2, 'flow_reductions_br'] == 0
                        if check1 and check2:
                            data.loc[st:end, col+'eff'] = 0
                            data.loc[st:end, 'immidiate_remove_eff'] = 1
                            continue
                        # convert effort flow reductions to <apnea> if indicated by Airflow
                        if np.any(data.loc[st:end, col+'air'] == 1): 
                            data.loc[st:end, col+'br'] = 1
                            data.loc[st:end, col+'eff'] = 0
                        else: 
                            data.loc[st:end, col+'br'] = 2
                            data.loc[st:end, col+'eff'] = 0        
            if trace == 'air':
                # run over all airflow reductions
                for st, end in find_events(data[col+'air'].values>0):
                    # remove air flow reductions if already found in breathing trace
                    if np.any(data.loc[st:end, col+'br'] > 0) or np.any(data.loc[st:end, col+'eff'] > 0): 
                        data.loc[st:end, col+'air'] = 0
                    # if event found in thermistor but nothing else while valid signals, remove event
                    flat_cols = ['flat_signal_br', 'flat_signal_eff', 'bad_signal_br', 'bad_signal_eff']
                    notFlat = np.all(data.loc[st:end, flat_cols] == 0)
                    if not np.any(data.loc[st:end, [col+'br', col+'eff']] > 0) and notFlat: 
                        data.loc[st:end, col+'air'] = 0
                    # if event found in thermistor but no hypopnea in breathing, remove event
                    flat_cols = ['flat_signal_br', 'bad_signal_br']
                    notFlat = np.all(data.loc[st:end, flat_cols] == 0)
                    if not np.any(data.loc[st:end, [col+'br', col+'eff']] > 1) and notFlat: 
                        data.loc[st:end, col+'air'] = 0
    
    # 4) add soft hypopneas that result in desaturation
    data['soft_desat_hyps'] = 0
    for drop in find_events(data['saturation_drop']>0):
        mid = int(np.mean(drop))
        begin = max(0, min(drop[0], mid-60*hdr['newFs']))
        region = list(range(begin, mid))
        # check in effort and airflow  whether a hypopnea was found
        for col in ['flow_reductions_eff', 'flow_reductions_air', 'ventilation_drop_soft_hyp_br']: #, 'ventilation_drop_soft_hyp_air']:
            if col not in data.columns: continue # in case some signals are not present
            found = begin + np.where(data.loc[region, col] > 0)[0]
            # if an event is found
            if len(found) > 0:
                # set region to -5min < drop < +1min
                mini = max(0, begin-5*60*hdr['newFs'])
                maxi = min(len(data), drop[0]+60*hdr['newFs'])
                for st, end in find_events(data.loc[list(range(mini, maxi)), col]>0):
                    reg = list(range(mini+st, mini+end))
                    check1 = found[-1] >= mini+st and found[-1] <= mini+end # 
                    check2 = np.all(data.loc[reg, 'flow_reductions_br']==0)
                    check3 = np.all(data.loc[reg, 'immidiate_remove_eff'] == 0)
                    if check1 and check2 and check3: 
                        data.loc[reg, col] = 0
                        data.loc[reg, 'flow_reductions_br'] = 2
                        data.loc[reg, 'soft_desat_hyps'] = 2
                        break
                break
                    
    # remove tration hypopneas only found in effort
    if hdr['test_type'] == 'split-night':
        eff_hyps = [ev for ev in find_events(data['flow_reductions_eff'].values==2) if ev[1]>hdr['cpap_start']]
        for st, end in eff_hyps:
            data.loc[st:end, 'flow_reductions_eff'] = 0
            data.loc[st:end, 'soft_ventilation_drop_apnea_eff'] = 0
    elif hdr['test_type'] == 'titration':
        data.loc[np.where(data['flow_reductions_eff']==2)[0], 'flow_reductions_eff'] = 0
        data.loc[:, 'soft_ventilation_drop_apnea_eff'] = 0
   
    # remove br hypopneas immediately after br apnea
    for st, end in find_events(data['flow_reductions_br'].values==2):
        check1 = np.any(data.loc[st-2*hdr['newFs']:st, 'flow_reductions_br'] == 1)
        check2 = np.any(data.loc[end:end+2*hdr['newFs'], 'flow_reductions_br'] == 1)
        if check1 or check2:
            data.loc[st:end, 'flow_reductions_br'] = 0
            data.loc[st:end, 'immidiate_remove_br'] = 1
            continue

    # add flow reductions together
    data['final_flow_reductions'] = data.flow_reductions_br + data.flow_reductions_eff
    if 'flow_reductions_air' in data.columns: 
        data['final_flow_reductions'] += data.flow_reductions_air 
    data['final_flow_reductions'] = data['final_flow_reductions'].astype(int)
    data['soft_ventilation_drop_apnea'] = data.soft_ventilation_drop_apnea_br + data.soft_ventilation_drop_apnea_eff 
    if 'soft_ventilation_drop_air' in data.columns: 
        data['final_flow_reductions'] += data.soft_ventilation_drop_apnea_air
    data['soft_ventilation_drop_apnea'] = data['soft_ventilation_drop_apnea'].astype(int)
    
    # correct for doulbe hypopnea
    for st, end in find_events(data['final_flow_reductions']>2):
        if data.loc[st-1, 'final_flow_reductions'] == 2 and data.loc[end+1, 'final_flow_reductions'] == 2:
            data.loc[st:end, 'final_flow_reductions'] = 2

    assert np.all(data['final_flow_reductions'] <= 2), 'Addition (+) error for combining <flow_reductions>'
    assert np.all(data['soft_ventilation_drop_apnea'] <= 2), 'Addition (+) error for combining <soft_ventilation_drop_apnea>'
    
    # reset DF colums to default breathing_trace --> baselines/envelopes/excursions
    data['Ventilation_combined'] = data['breathing_trace'].values
    data = compute_ventilation_envelopes(data, hdr['newFs'])
    data = assess_ventilation(data, hdr, drop_hyp=0.2, drop_apnea=0.8, dur_apnea=7, dur_hyp=7, quant=0.75)
    data = combine_flow_reductions(data, hdr)
    
    # save combined signals
    data['bad_signal'] = np.logical_and(data.bad_signal_br, data.bad_signal)
    data['Ventilation_drop_apnea'] = np.array(data['final_flow_reductions']==1).astype(int)
    data['Ventilation_drop_hypopnea'] = np.array(data['final_flow_reductions']==2).astype(int)
    
    ########################################
    # # final flow reductions
    # plt.plot(data.flow_reductions_br.mask(data.immidiate_remove_br!=1)+7.25, 'y', lw=2) ####
    # plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=2)+12, 'm', lw=4)
    # plt.plot(data.flow_reductions_br.mask(data.soft_desat_hyps!=2)+13, 'y', lw=4)
    # plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=1)+13, 'k', lw=4)
    # # plt.plot(data.soft_ventilation_drop_apnea_br.mask(data.soft_ventilation_drop_apnea_br!=1)+13.5, 'c', lw=4)
    # # final efffort reductions
    # plt.plot(data.flow_reductions_eff.mask(data.immidiate_remove_eff!=1)-11.25, 'y', lw=2) ####
    # plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=2)+12, 'm', lw=2)
    # plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=1)+13, 'k', lw=2)
    # # plt.plot(data.soft_ventilation_drop_apnea_eff.mask(data.soft_ventilation_drop_apnea_eff!=1)+13.5, 'c', lw=2)
    # # final AIR reductions
    # plt.plot(data.flow_reductions_air.mask(data.flow_reductions_air!=2)+12, 'm', lw=2)
    # plt.plot(data.flow_reductions_air.mask(data.flow_reductions_air!=1)+13, 'k', lw=2)
    # # plt.plot(data.soft_ventilation_drop_apnea_air.mask(data.soft_ventilation_drop_apnea_air!=1)+13.5, 'c', lw=2)
    # # create CMT
    # # sys.path.insert(0, os.getcwd()); from run_rule_based_algo import convert_samples_to_event_arrays
    # # from sklearn.metrics import confusion_matrix
    # # resp = data['resp-h3_expert_1'] if hdr['dataset'] == 'jedi' else data['Apnea']
    # # data['y'], data['yp'], data['sleep_stages'] = resp, data.final_flow_reductions, data.Stage
    # # yt, yp = convert_samples_to_event_arrays(data, hdr['newFs'])
    # # x, y = np.array(yt), np.array(yp)
    # # x[x==2] = 1; x[x==3] = 1; x[x==4] = 2; x[x==5] = 2
    # # cmt = confusion_matrix(x, y, labels=range(3))
    # # cmt_n = np.round(cmt.astype('float') / cmt.sum(axis=1)[:, np.newaxis], 2)
    # # print(cmt)
    # # plot all
    # tag, tt = hdr['patient_tag'], hdr['test_type']
    # plt.title(f'{tag} - {tt}')
    # plt.show()
    # import pdb; pdb.set_trace()
    #######################################

    # remove a bunch of columns
    del_cols = ['final_flow_reductions', 'flat_signal']
    del_cols += [c for c in data.columns if '_br' in c or '_eff' in c or '_air' in c]
    # data = data.drop(columns=del_cols)

    return data

def find_central_effort_componenets(data: pd.DataFrame, hdr: dict, quant: float, plot: bool = False) -> pd.DataFrame:
    """
    Identifies central effort components in respiratory effort signals.

    Args:
    - data: A pandas DataFrame containing the respiratory effort signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    - quant: Quantile threshold for defining excursions.
    - plot: A boolean flag for whether to plot the results.

    Returns:
    - Updated DataFrame with central effort component classifications.
    """

    Fs = hdr['newFs']; window = 300*Fs
    sizes = ['tiny', 'xsmall', 'small']
    thresholds = [0.10, 0.20, 0.30] # [0.10, 0.20, 0.30]
    
    for j, trace in enumerate(['ABD_CHEST', 'ABD', 'CHEST']):
        # compute envelope and baseline on breathing trace
        data['Ventilation_combined'] = data[trace]
        data = compute_ventilation_envelopes(data, Fs)
        pos_excursion, neg_excursion = data.Ventilation_pos_envelope, data.Ventilation_neg_envelope

        # compute dynamic excursion threshold, both for apnea and hypopneas.
        excursion_duration = 300     # the larger the interval, the less dynamic a baseline gets computed.
        excursion_q = 0.8           # ventilation envelope quantile: 0.8

        # use lagging moving windonw, events are found based on future eupnea / recovery breaths
        pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
        pos_excursion[:-excursion_duration*Fs*2] = pos_excursion[excursion_duration*Fs*2:]
        pos_excursion[-excursion_duration*Fs*2:] = np.nan
        neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values
        neg_excursion[:-excursion_duration*Fs*2] = neg_excursion[excursion_duration*Fs*2:]
        neg_excursion[-excursion_duration*Fs*2:] = np.nan

        # add additional envelope smoothing
        minutes = 20
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values # 0.4
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values # 0.6
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  

        exc = pos_excursion - neg_excursion
        exc_max = exc
        data[trace+'_exc'] = exc_max
        for i, (col, thresh) in enumerate(zip(sizes, thresholds)):
            data[f'{trace}_{col}_exc'] = (data.Ventilation_pos_envelope - data.Ventilation_neg_envelope) < thresh*exc_max
            data.loc[data['flow_reductions_br']==0, f'{trace}_{col}_exc'] = 0

            # remove bad signal events
            data = bad_signal_check(data, hdr['newFs'])
            data[f'bad_signal_{trace}'] = data.bad_signal.values
            data[f'flat_signal_{trace}'] = data.flat_signal.values
            data = remove_error_events(data, trace, columns=[f'{trace}_{col}_exc'])
            if plot:
                if i==0: c = 'r'
                elif i==1: c = 'b'
                else: c = 'k'
                plt.plot(data[f'{trace}_{col}_exc'].mask(data[f'{trace}_{col}_exc']==0)+1+(0.2*i)-5*(j+1), c)

    # flat central components
    data['central_component'] = 0
    data['soft_central_component'] = 0
    data['mixed_component'] = 0
    data['central_component_flow'] = 0
    for st, end in find_events(data['flow_reductions_eff']>0):
        # skip small segments
        if (end-st) < 8*Fs: continue
        # spectral assess effort limitations; if peak >0.5Hz, likely not respiration
        not_breathing = spectral_assess_effort_limitation(data, st, end, Fs, 'CHEST')
        if not_breathing[0]:
            data.loc[st:end, 'soft_central_component'] = 1

    for st, end in find_events(data['flow_reductions_br']>0):
        event_len = end-st
        # if 332600>st and 332600<end: import pdb; pdb.set_trace()
        # spectral assess effort limitations; if peak >0.5Hz, likely not respiration
        not_breathing = spectral_assess_effort_limitation(data, st, end, Fs, 'CHEST')
        if not_breathing[0]:
            data.loc[st:end, 'soft_central_component'] = 1
        elif not_breathing[1]:
            data.loc[st:end, 'soft_central_component'] = 1
            data.loc[st:end, 'mixed_component'] = 1
        
        # 2 channels show 10% or 3sec red (tiny)
        check1a = sum(data.loc[st:end, 'ABD_tiny_exc'] == 1) > 0.1*event_len
        check1b = sum(data.loc[st:end, 'CHEST_tiny_exc'] == 1) > 0.1*event_len
        check2a = sum(data.loc[st:end, 'ABD_tiny_exc'] == 1) > 3*Fs
        check2b = sum(data.loc[st:end, 'CHEST_tiny_exc'] == 1) > 3*Fs
        # if (check1a and check1b) or (check2a and check2b):
        if (check1a and check1b) and (check2a and check2b):
            data.loc[st:end, 'central_component'] = 1
            continue

        # 1/2 channels shows 25% red (tiny) + ABD & CHEST show blue (xsmall)
        check2a = sum(data.loc[st:end, 'ABD_tiny_exc'] == 1) > 0.25*event_len
        check2b = sum(data.loc[st:end, 'ABD_tiny_exc'] == 1) > 4*Fs
        check3a = sum(data.loc[st:end, 'CHEST_tiny_exc'] == 1) > 0.25*event_len
        check3b = sum(data.loc[st:end, 'CHEST_tiny_exc'] == 1) > 4*Fs
        # if check2a or check2b or check3a or check3b:
        if (check2a and check2b) or (check3a and check3b):
            cols = ['ABD_xsmall_exc', 'CHEST_xsmall_exc', 'ABD_CHEST_xsmall_exc']
            if sum(np.any(data.loc[st:end, cols] == 1, 0)) == 3:
                data.loc[st:end, 'central_component'] = 1
                continue

        # if >70% or 7 sec blue (xsmall)
        check5 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 0.7*event_len
        check6 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 7*Fs
        check7 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 0.7*event_len
        check8 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 7*Fs
        check9 = sum(data.loc[st:end, 'ABD_CHEST_tiny_exc'] == 1) > 0.5*event_len
        check10 = sum(data.loc[st:end, 'ABD_CHEST_tiny_exc'] == 1) > 10*Fs
        if (check5 and check6 and (check9 or check10)) or (check7 and check8 and (check9 or check10)):
        # if (check5 and check6 and check7 and check8 and (check9 or check10)):
            data.loc[st:end, 'central_component'] = 1
            continue

        # soft central components + 50% blue
        check5 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 0.4*event_len
        check6 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 4*Fs
        check7 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 0.4*event_len
        check8 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 4*Fs
        check9 = np.any(data.loc[st:end, 'soft_central_component']>0)
        if (check5 or check7) and (check6 or check8) and check9:
            data.loc[st:end, 'central_component'] = 1
            continue

        # if nearly all breathing flow reduction is red/blue, also detect as central
        reg = np.where(data.loc[st:end, 'flow_reductions_br']==1)[0]+st
        if len(reg)<6*Fs: continue
        check10 = sum(data.loc[st:end, 'ABD_tiny_exc'] == 1) > 0.2*len(reg)
        check11 = sum(data.loc[st:end, 'CHEST_tiny_exc'] == 1) > 0.2*len(reg)
        check12 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 0.8*len(reg)
        check13 = sum(data.loc[st:end, 'ABD_xsmall_exc'] == 1) > 6*Fs
        check14 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 0.8*len(reg)
        check15 = sum(data.loc[st:end, 'CHEST_xsmall_exc'] == 1) > 6*Fs
        check16 = sum(data.loc[st:end, 'ABD_CHEST_tiny_exc'] == 1) > 0.5*len(reg)
        if (check10 and check11) or (check12 and check13 and check16) or (check14 and check15 and check16):
            data.loc[st:end, 'central_component'] = 1
            data.loc[reg, 'central_component_flow'] = 1
            continue

    if plot:
        # plot excursion
        # plt.plot(data.Ventilation_pos_envelope - data.Ventilation_neg_envelope, 'r', alpha=0.5)
        # plt.plot(exc_max, 'g', alpha=0.5)

        # plot      breathing + effort
        data['patient_asleep'] = np.logical_and(data.Stage < 5, data.Stage > 0)
        plt.plot(data.breathing_trace.mask(data.patient_asleep==0)+10, 'k', alpha=0.5)
        plt.plot(data.breathing_trace.mask(data.patient_asleep==1)+10, 'k', alpha=0.1)
        plt.plot(data.flat_signal_br.mask(data.flat_signal_br==0)*10, 'r')
        plt.plot(data.bad_signal_br.mask(data.bad_signal_br==0)*9.5, 'y')
        # combined effort trace
        plt.plot(data.ABD_CHEST.mask(data.patient_asleep==0)-5, 'c', alpha=0.5)
        plt.plot(data.ABD_CHEST.mask(data.patient_asleep==1)-5, 'c', alpha=0.1)
        plt.plot(data.Ventilation_pos_envelope, 'k', alpha=0.3)
        plt.plot(data.Ventilation_neg_envelope, 'k', alpha=0.3)
        plt.plot(data.flat_signal_eff.mask(data.flat_signal_eff==0)-5, 'r')
        plt.plot(data.bad_signal_eff.mask(data.bad_signal_eff==0)-5.5, 'y')
        # individual effort traces
        plt.plot(data.ABD.mask(data.patient_asleep==0)-10, 'b', alpha=0.5)
        plt.plot(data.ABD.mask(data.patient_asleep==1)-10, 'b', alpha=0.1)
        plt.plot(data.CHEST.mask(data.patient_asleep==0)-15, 'g', alpha=0.5)
        plt.plot(data.CHEST.mask(data.patient_asleep==1)-15, 'g', alpha=0.1)
        
        # plot      original labels 
        resp = 'resp-h3_expert_1' if hdr['dataset'] == 'jedi' else 'Apnea'
        if 'resp-h3_platinum' in data.columns: resp = 'resp-h3_platinum'
        plt.plot(data[resp].mask(data[resp]!=1) + 11, 'b', lw=4)
        plt.plot(data[resp].mask(data[resp]!=2) + 11, 'g', lw=4)
        plt.plot(data[resp].mask(data[resp]!=3) + 11, 'c', lw=4)
        plt.plot(data[resp].mask(data[resp]!=4) + 8, 'm', lw=4)
        # plot      br reductions
        plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=1)+6, 'k', lw=2)
        plt.plot(data.flow_reductions_br.mask(data.flow_reductions_br!=2)+5, 'm', lw=2)
        # plot      effort reductions
        plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=2)-3, 'm', lw=2) 
        plt.plot(data.flow_reductions_eff.mask(data.flow_reductions_eff!=1)-2, 'k', lw=2) 
        plt.plot(data.central_component.mask(data.central_component==0)-2.2, 'g', lw=2)
        plt.plot(data.soft_central_component.mask(data.soft_central_component==0)-1.6, 'c', lw=2)      
        plt.plot(data.central_component_flow.mask(data.central_component_flow==0)-2.4, 'b', lw=2)
        tag, tt = hdr['patient_tag'], hdr['test_type']
        plt.title(f'{tag} - {tt}')
        plt.show()
        import pdb; pdb.set_trace()
        data['patient_asleep'] = 1

    return data

def find_flat_thermistor_componenets(data: pd.DataFrame, hdr: dict, plot: bool = False) -> pd.DataFrame:
    """
    Identifies flat segments in thermistor airflow signal.

    Args:
    - data: A pandas DataFrame containing respiratory airflow data.
    - hdr: A dictionary containing metadata and header information for the recording.
    - plot: A boolean flag for whether to plot the results.

    Returns:
    - Updated DataFrame with flat airflow segments identified.
    """

    # compute 20min envelope to normalize std threshold
    Fs = hdr['newFs']
    minutes = 20
    win = int(Fs*60*minutes)
    hour_envelope = (data.Ventilation_pos_envelope - data.Ventilation_neg_envelope) / 2
    hour_envelope = hour_envelope.rolling(win, center=True, min_periods=60*Fs).median().values
    thresh1 = 0.05*hour_envelope
    thresh2 = 0.08*hour_envelope

    # low std
    breathing_std = data['airflow'].rolling(Fs, center=True).std().fillna(-1)
    low_std = np.logical_and(breathing_std<thresh1, breathing_std>-thresh1)
    # low diff
    breathing_diff = data['airflow'].diff(periods=Fs).rolling(Fs, center=True).std().fillna(-1)
    low_diff = np.logical_and(breathing_diff<thresh2, breathing_diff>-thresh2)
    # low std + low diff
    flat = np.logical_or(low_std, low_diff)
    flat = flat.rolling(int(0.75*Fs), center=True).max().fillna(1)

    # connect very close events
    connected_events, _ = connect_events(find_events(flat>0), 2, hdr['newFs'], max_dur=20)
    connected_flat = events_to_array(connected_events, len(data))

    # keep only segments 5sec > seg > 30sec 
    data['flat_airflow'] = 0
    for st, end in find_events(connected_flat>0):
        flat_cols = ['flat_signal_air', 'bad_signal_air']
        notFlat = np.all(data.loc[st:end, flat_cols] == 0)
        if (end-st) > 6*Fs and (end-st) < 30*Fs and notFlat:
            data.loc[st:end, 'flat_airflow'] = 1

    if plot: 
        plt.plot(data.Ventilation_combined, 'y')
        plt.plot(data.flat_airflow.mask(data.flat_airflow!=1), 'k')
        plt.plot(low_std.mask(low_std!=1)-0.5, 'r')
        plt.plot(low_diff.mask(low_diff!=1)-0.75, 'b')
        plt.show()
        import pdb; pdb.set_trace()

    return data



# helper functions
def combine_flow_reductions(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Combines apnea and hypopnea flow reductions into a single flow reduction array.

    Args:
    - data: A pandas DataFrame containing respiratory data.
    - hdr: A dictionary containing metadata and header information for the recording.

    Returns:
    - Updated DataFrame with combined flow reductions.
    """

    # set data arrays
    Fs = hdr['newFs']
    apneas = data.Ventilation_drop_apnea
    hypopneas = data.Ventilation_drop_hypopnea
    grouped_hypopneas = find_events(hypopneas>0)
    data['flow_reductions'] = apneas

    # add hypopneas to apnea array
    for st, end in grouped_hypopneas:
        region = list(range(st, end))
        # insert when no apnea is found in that region
        if np.all(apneas[region] == 0):
            data.loc[region, 'flow_reductions'] = 2
        else:
            reg = region[2*Fs:-2*Fs]
            if len(reg)<10*Fs: continue
            if np.all(apneas[reg] == 0):
                data.loc[reg, 'flow_reductions'] = 2
    
    return data

def assess_ventilation(data: pd.DataFrame, hdr: dict, drop_hyp: float, drop_apnea: float, dur_apnea: float, dur_hyp: float, quant: float, extra_smooth: bool = False, plot: bool = False) -> pd.DataFrame:
    """
    Assesses ventilation by detecting apnea and hypopnea events in the respiratory signal.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    - drop_hyp: Hypopnea drop threshold.
    - drop_apnea: Apnea drop threshold.
    - dur_apnea: Minimum apnea duration.
    - dur_hyp: Minimum hypopnea duration.
    - quant: Quantile threshold for defining excursions.
    - extra_smooth: Whether to apply additional smoothing to the signal.
    - plot: A boolean flag for whether to plot the results.

    Returns:
    - Updated DataFrame with detected ventilation drops and events.
    """

    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    pos_excursion[:-excursion_duration*Fs*2] = pos_excursion[excursion_duration*Fs*2:]
    pos_excursion[-excursion_duration*Fs*2:] = np.nan
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values
    neg_excursion[:-excursion_duration*Fs*2] = neg_excursion[excursion_duration*Fs*2:]
    neg_excursion[-excursion_duration*Fs*2:] = np.nan

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 20
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  
        # compute smoothed uncalibrated apnea excursion
        middle = np.mean([pos_excursion, neg_excursion], 0)
        pos_distance_to_baseline = np.abs(pos_excursion-middle)
        neg_distance_to_baseline = np.abs(neg_excursion-middle)
        data['pos_excursion_apnea1'] = middle + (pos_distance_to_baseline * (1-drop_apnea))
        data['neg_excursion_apnea1'] = middle - (neg_distance_to_baseline * (1-drop_apnea)) 

    # Relative pos/neg excursion (Hypopnneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))
    ##########################
    ###
    data['pos_excursion_apnea1'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
    data['neg_excursion_apnea1'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
    ###

    # Average pos/neg excursion (Apneas)
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_default_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_default_baseline'])
    dist_to_baseline = np.mean([pos_distance_to_baseline, neg_distance_to_baseline], 0)
    data['pos_excursion_apnea'] = data['Ventilation_default_baseline'] + (dist_to_baseline * (1-drop_apnea))
    data['neg_excursion_apnea'] = data['Ventilation_default_baseline'] - (dist_to_baseline * (1-drop_apnea))
    ###
    data['pos_excursion_hyp1'] = data['Ventilation_default_baseline'] + (dist_to_baseline * (1-drop_hyp))
    data['neg_excursion_hyp1'] = data['Ventilation_default_baseline'] - (dist_to_baseline * (1-drop_hyp))
    ####
    # data['pos_excursion_hyp'] = data[['pos_excursion_hyp1', 'pos_excursion_hyp2']].mean(1)
    # data['neg_excursion_hyp'] = data[['neg_excursion_hyp1', 'neg_excursion_hyp2']].mean(1)

    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    # combine positive and negative excursion flow limitations
    data = pos_neg_excursion_combinations(data, hdr)

    if plot:
        plt.figure(figsize=(9.5,6)) 

        # signal and baseline
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==0),'y', lw=0.5, alpha=0.5)
        plt.plot(data.Ventilation_combined.mask(data.patient_asleep==1),'r', lw=0.5, alpha=0.5)
        plt.plot(data.Ventilation_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'k', lw=0.5)
        plt.plot(data.Ventilation_default_baseline.mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),'k--', lw=0.5)
        # envelopes
        plt.plot(pos_excursion,'b--')
        plt.plot(neg_excursion,'b--')
        
        # Main apnea / hypopnea threshold lines
        for col, c, lw in zip(['excursion_apnea', 'excursion_hyp', 'excursion_soft_hyp'], ['r', 'm'], [0.8, 0.5]):
            for pn in ['pos_', 'neg_']:
                plt.plot(data[pn+col].mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)), c, lw=lw)
        # Secondary version
        for col, c in zip(['excursion_apnea1', 'excursion_hyp1'], ['r', 'm']):
            for pn in ['pos_', 'neg_']:
                if not pn+col in data.columns: continue
                plt.plot(data[pn+col].mask(np.isnan(data.Ventilation_combined) | (data.patient_asleep==0)),c+'--', lw=0.5)

        # original labels
        d = data['resp-h3_expert_1'] if hdr['dataset'] == 'jedi' else data['Apnea']
        plt.plot(d.mask((data.patient_asleep==0) | (d!=1)), 'b', lw=2)
        plt.plot(d.mask((data.patient_asleep==0) | (d!=2)), 'g', lw=2)
        plt.plot(d.mask((data.patient_asleep==0) | (d!=3)), 'c', lw=2)
        plt.plot(d.mask((data.patient_asleep==0) | (d!=4)), 'm', lw=2)

        #### 
        plt.plot(-data.Ventilation_drop_apnea.mask(data.Ventilation_drop_apnea!=1), 'g', lw=2)
        # plt.plot(-data.soft_ventilation_drop_apnea.mask(data.soft_ventilation_drop_apnea!=1)*1.25, 'c', lw=2)

        plt.plot(-4*data.Ventilation_drop_hypopnea.mask(data.Ventilation_drop_hypopnea!=1), 'm', lw=2)
        plt.plot(-3.5*data.pos_Ventilation_drop_hypopnea.mask(data.pos_Ventilation_drop_hypopnea!=1), 'k')
        plt.plot(-4.5*data.neg_Ventilation_drop_hypopnea.mask(data.neg_Ventilation_drop_hypopnea!=1), 'k')

        # plt.plot(-6.5*data['just_pos_hypes'].mask(data['just_pos_hypes'] != 1), 'm')
        # plt.plot(-7*data['test_either_hypes'].mask(data['test_either_hypes'] != 1), 'c')

        plt.show()
        import pdb; pdb.set_trace()

    return data

def remove_non_forward_drops(data: pd.DataFrame, hdr: dict, drop_hyp: float, drop_apnea: float, dur_apnea: float, dur_hyp: float, quant: float, extra_smooth: bool = False) -> pd.DataFrame:
    """
    Removes ventilation events without forward drops from the respiratory signal.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    - drop_hyp: Hypopnea drop threshold.
    - drop_apnea: Apnea drop threshold.
    - dur_apnea: Minimum apnea duration.
    - dur_hyp: Minimum hypopnea duration.
    - quant: Quantile threshold for defining excursions.
    - extra_smooth: Whether to apply additional smoothing to the signal.

    Returns:
    - Updated DataFrame with non-forward drops removed.
    """
    
    # compute dynamic excursion threshold, both for apnea and hypopneas.
    Fs = hdr['newFs']
    excursion_duration = 60     # the larger the interval, the less dynamic a baseline gets computed.
    excursion_q = quant         # ventilation envelope quantile

    # use lagging moving windonw, events are found based on future eupnea / recovery breaths
    pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
    neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values

    # add additional envelope smoothing
    if extra_smooth:
        minutes = 5
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg)  

    # set excursion into DF
    pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
    neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
    data['pos_excursion_apnea'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
    data['pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
    data['neg_excursion_apnea'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
    data['neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    ### add soft hypopneas ###
    data['pos_excursion_soft_hyp'] = data['Ventilation_default_baseline'] + (pos_distance_to_baseline * (1-drop_hyp/1.02))
    data['neg_excursion_soft_hyp'] = data['Ventilation_default_baseline'] - (neg_distance_to_baseline * (1-drop_hyp/1.02))
    ##########################
   
    # find drops in ventilation signal for apneas and hypopneas
    data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)

    apnea_cols = [c for c in data.columns if '_Ventilation_drop_apnea' in c]
    hyp_cols = [c for c in data.columns if '_Ventilation_drop_hypopnea' in c]
    soft_hyp_cols = [c for c in data.columns if '_Ventilation_drop_soft_hypopnea' in c]
    
    # run over all found apneas
    for st, end in find_events(data['Ventilation_drop_apnea']>0):
        # if no apnea is found in either trace, remove apnea
        if np.all(data.loc[st:end, apnea_cols]==0):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 0
            # if hypopnea is found, replace apnea by hypopnea
            if np.any(data.loc[st:end, hyp_cols]>0):
                data.loc[st:end, 'Ventilation_drop_hypopnea'] = 1

    # run over all found hypopneas
    for st, end in find_events(data['Ventilation_drop_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_hypopnea'] = 0

    # run over all found soft hypopneas
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, soft_hyp_cols+hyp_cols]==0):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def remove_non_individual_effort_drops(data: pd.DataFrame, hdr: dict, drop_hyp: float, drop_apnea: float, dur_apnea: float, dur_hyp: float, quant: float, plot: bool = False) -> pd.DataFrame:
    """
    Removes effort flow reduction events not found in individual respiratory traces.

    Args:
    - data: A pandas DataFrame containing respiratory effort signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    - drop_hyp: Hypopnea drop threshold.
    - drop_apnea: Apnea drop threshold.
    - dur_apnea: Minimum apnea duration.
    - dur_hyp: Minimum hypopnea duration.
    - quant: Quantile threshold for defining excursions.
    - plot: A boolean flag for whether to plot the results.

    Returns:
    - Updated DataFrame with non-individual effort drops removed.
    """

    for eff in ['abd', 'chest']:
        data['Ventilation_combined'] = data[eff.upper()]
        # compute envelope and baseline on breathing trace
        data = compute_ventilation_envelopes(data, hdr['newFs'])
    
        # compute dynamic excursion threshold, both for apnea and hypopneas.
        Fs = hdr['newFs']
        excursion_duration = 40     # the larger the interval, the less dynamic a baseline gets computed.
        excursion_q = quant         # ventilation envelope quantile

        # use lagging moving windonw, events are found based on future eupnea / recovery breaths
        pos_excursion = data.Ventilation_pos_envelope.rolling(excursion_duration*Fs*2).quantile(excursion_q, interpolation='lower').values
        neg_excursion = data.Ventilation_neg_envelope.rolling(excursion_duration*Fs*2).quantile(1-excursion_q, interpolation='lower').values

        # smooth envelope
        minutes = 5
        win = int(Fs*60*minutes)
        pos = pd.DataFrame(data=pos_excursion).rolling(win, center=True, min_periods=1).quantile(0.4).values
        neg = pd.DataFrame(data=neg_excursion).rolling(win, center=True, min_periods=1).quantile(0.6).values
        pos_excursion = np.squeeze(pos)
        neg_excursion = np.squeeze(neg) 

        # set excursion into DF
        pos_distance_to_baseline = np.abs(pos_excursion - data['Ventilation_baseline'])
        neg_distance_to_baseline = np.abs(neg_excursion - data['Ventilation_baseline'])
        data[f'pos_excursion_apnea'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_apnea))
        data[f'pos_excursion_hyp'] = data['Ventilation_baseline'] + (pos_distance_to_baseline * (1-drop_hyp))
        data[f'neg_excursion_apnea'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_apnea))
        data[f'neg_excursion_hyp'] = data['Ventilation_baseline'] - (neg_distance_to_baseline * (1-drop_hyp))
    
        # find drops in ventilation signal for apneas and hypopneas
        data = locate_ventilation_drops(data, hdr, dur_apnea, dur_hyp)
        data = data.rename(columns={"pos_Ventilation_drop_hypopnea": f"pos_Ventilation_drop_hypopnea_{eff}"})
        data = data.rename(columns={"neg_Ventilation_drop_hypopnea": f"neg_Ventilation_drop_hypopnea_{eff}"})
        data = data.rename(columns={"pos_Ventilation_drop_apnea": f"pos_Ventilation_drop_apnea_{eff}"})
        data = data.rename(columns={"neg_Ventilation_drop_apnea": f"neg_Ventilation_drop_apnea_{eff}"})

    hyp_abd_cols = [c for c in data.columns if '_Ventilation_drop_hypopnea_abd' in c]
    hyp_chest_cols = [c for c in data.columns if '_Ventilation_drop_hypopnea_chest' in c]
    apnea_abd_cols = [c for c in data.columns if '_Ventilation_drop_apnea_abd' in c]
    apnea_chest_cols = [c for c in data.columns if '_Ventilation_drop_apnea_chest' in c]
    # run over all found sum apneas
    for st, end in find_events(data['Ventilation_drop_apnea']>0):
        # if no apnea is found in either trace, remove apnea
        if np.all(data.loc[st:end, apnea_abd_cols]==0) and np.all(data.loc[st:end, apnea_chest_cols]==0):
            data.loc[st:end, 'Ventilation_drop_apnea'] = 0
            # if hypopnea is found, replace apnea by hypopnea
            if np.any(data.loc[st:end, hyp_abd_cols]>0) or np.any(data.loc[st:end, hyp_chest_cols]>0):
                data.loc[st:end, 'Ventilation_drop_hypopnea'] = 1

    # run over all found hypopneas
    for st, end in find_events(data['Ventilation_drop_hypopnea']>0):
        # remove when no forward hypopnea is found
        if np.all(data.loc[st:end, hyp_abd_cols]==0) or np.all(data.loc[st:end, hyp_chest_cols]==0):
            data.loc[st:end, 'Ventilation_drop_hypopnea'] = 0

    return data

def locate_ventilation_drops(data: pd.DataFrame, hdr: dict, dur_apnea: float, dur_hyp: float) -> pd.DataFrame:
    """
    Locates drops in ventilation signals indicating apnea or hypopnea events.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - hdr: A dictionary containing metadata and header information for the recording.
    - dur_apnea: Minimum apnea duration.
    - dur_hyp: Minimum hypopnea duration.

    Returns:
    - Updated DataFrame with located ventilation drops.
    """

    # remove NaN values only for selected channels -->
    selected_columns = ['pos_excursion_apnea', 'pos_excursion_hyp', 'pos_excursion_soft_hyp',
                        'neg_excursion_apnea', 'neg_excursion_hyp', 'neg_excursion_soft_hyp',
                        'Ventilation_baseline']
    new_df = data[selected_columns + ['Ventilation_combined']].copy()

    # put selected columns in original dataframe
    for col in selected_columns:
        data[col] = new_df[col]

    # *add smoothed exursion thresholds*
    win = int(hdr['newFs']*60*10)
    data['pos_excursion_apnea_smooth'] = data['pos_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_apnea_smooth'] = data['neg_excursion_apnea'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_hyp_smooth'] = data['pos_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_hyp_smooth'] = data['neg_excursion_hyp'].rolling(win, center=True, min_periods=1).median()
    data['pos_excursion_soft_hyp_smooth'] = data['pos_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()
    data['neg_excursion_soft_hyp_smooth'] = data['neg_excursion_soft_hyp'].rolling(win, center=True, min_periods=1).median()

    # find areas with potential apnea / hypopnea flow limitations
    sig = data.Ventilation_combined
    data['pos_Ventilation_drop_apnea'] = np.logical_or(sig<data.pos_excursion_apnea, sig<data.pos_excursion_apnea_smooth)
    data['neg_Ventilation_drop_apnea'] = np.logical_or(sig>data.neg_excursion_apnea, sig>data.neg_excursion_apnea_smooth)
    data['pos_Ventilation_drop_hypopnea'] = np.logical_or(sig<data.pos_excursion_hyp, sig<data.pos_excursion_hyp_smooth)
    data['neg_Ventilation_drop_hypopnea'] = np.logical_or(sig>data.neg_excursion_hyp, sig>data.neg_excursion_hyp_smooth)
    data['pos_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig<data.pos_excursion_soft_hyp, sig<data.pos_excursion_soft_hyp_smooth)
    data['neg_Ventilation_drop_soft_hypopnea'] = np.logical_or(sig>data.neg_excursion_soft_hyp, sig>data.neg_excursion_soft_hyp_smooth)

    # run over the various ventilation drop options, and find flow limitations
    data['either_hypes'] = 0  
    tag_window = [dur_apnea*0.8, dur_apnea, dur_hyp, dur_hyp]
    for ex in ['pos', 'neg']:
        tag_list = [f'{ex}_soft_ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_apnea', 
                    f'{ex}_Ventilation_drop_hypopnea',
                    f'{ex}_Ventilation_drop_soft_hypopnea']
        data = find_flow_limitations(data, tag_list, tag_window, hdr['newFs'])

    return data

def find_flow_limitations(data: pd.DataFrame, tag_list: list, tag_window: list, Fs: int) -> pd.DataFrame:
    """
    Identifies flow limitations in respiratory signals based on windowed excursions.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - tag_list: List of tags to identify flow limitations.
    - tag_window: List of window sizes for each tag.
    - Fs: Sampling frequency.

    Returns:
    - Updated DataFrame with identified flow limitations.
    """

    for t, tag in enumerate(tag_list):
        win = int(tag_window[t]*Fs)
        # find events with duration <win>
        if t==0:
            col = [c for c in tag_list if 'Ventilation_drop_apnea' in c]
            data[tag] = data[col].rolling(win, center=True).mean() > 0.75 # allow for small peaks exceeding threshold
        else:
            data[tag] = data[tag].rolling(win, center=True).mean() == 1 # <win> should stay below threshold

        # apply window correction
        data[tag] = np.array(data[tag].fillna(0))
        cut = Fs//2 if 'hypopnea' in tag else 0 # slightly shorten event for Hypopneas
        data[tag] = window_correction(data[tag], window_size=win-cut)

        # remove all apnea events with a duration > .. sec
        max_dur = 150 if 'hypopnea' in tag else 120
        data = remove_long_events(data, tag, Fs, max_duration=max_dur)

    return data

def pos_neg_excursion_combinations(data: pd.DataFrame, hdr: dict) -> pd.DataFrame:
    """
    Combines positive and negative excursions to identify apneas and hypopneas.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - hdr: A dictionary containing metadata and header information for the recording.

    Returns:
    - Updated DataFrame with combined excursions.
    """

    # run over apnea options
    apnea_cols = ['Ventilation_drop_apnea', 'soft_ventilation_drop_apnea']
    hypopnea_cols = ['Ventilation_drop_hypopnea', 'Ventilation_drop_soft_hypopnea']
    for col in apnea_cols + hypopnea_cols:
        # for soft apneas, pos and neg flow limitation has to occur simultaniously
        if 'soft_ventilation' in col:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
        # for hypopneas, either positive or negative flow limitation is saved (prioiritize pos) 
        elif 'hypopnea' in col:
            data = hyp_flow_limitations(data, col)
        # for apneas, pos and neg flow limitation has to occur simultaniously
        else:
            data[col] = (data['pos_%s'%col] * data['neg_%s'%col]) > 0
            # connect apneas if pos or neg criteria continues
            data[col] = connect_apneas(data, col, 10, hdr['newFs'], max_dur=120)
            
        # connect events, within 5 sec (only if total event < 20sec)
        events = find_events(data[col].fillna(0)>0)
        if len(events) == 0: continue
        events, _ = connect_events(events, 3, hdr['newFs'], max_dur=20)
        data[col] = events_to_array(events, len(data))
        
        # remove events < 4 sec
        data[col] = remove_short_events(data[col], 4*hdr['newFs'])

    # remove soft ventilation drop apnea, if apnea found
    for st, end in find_events(data['soft_ventilation_drop_apnea']>0):
        if np.any(data.loc[st:end, 'Ventilation_drop_apnea']==1):
            data.loc[st:end, 'soft_ventilation_drop_apnea'] = 0
    # remove soft ventilation drop hypopnea, if apnea/hypopnea found
    for st, end in find_events(data['Ventilation_drop_soft_hypopnea']>0):
        if np.any(data.loc[st:end, ['Ventilation_drop_apnea', 'Ventilation_drop_hypopnea']]==1):
            data.loc[st:end, 'Ventilation_drop_soft_hypopnea'] = 0

    return data

def hyp_flow_limitations(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Identifies hypopnea flow limitations based on positive and negative excursions.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - col: Column name for the hypopnea flow limitation to assess.

    Returns:
    - Updated DataFrame with identified hypopnea flow limitations.
    """

    data[col] = data[f'pos_{col}'].values
    neg = find_events(data[f'neg_{col}'].fillna(0)>0)

    # run over flow limitation regions
    for st, end in neg:
        region = list(range(st, end))
        
        # skip if already saved in pos array
        if any(data.loc[region, f'pos_{col}']): continue

        # save in array
        data.loc[region, col] = 1

    return data

def connect_apneas(data: pd.DataFrame, col: str, win: int, Fs: int, max_dur: bool = False) -> np.ndarray:
    """
    Connects apnea events that are close in time based on a defined window.

    Args:
    - data: A pandas DataFrame containing respiratory signals.
    - col: Column name for the apnea flow limitation to assess.
    - win: Time window for connecting apneas.
    - Fs: Sampling frequency.
    - max_dur: Maximum duration for connecting events.

    Returns:
    - Updated array with connected apnea events.
    """

    # set events
    events = find_events(data[col].fillna(0)>0)
    if len(events) == 0: return data[col].values

    # connect apneas if pos or neg negative treshold remains
    new_events = []
    cnt = 0
    win = win*Fs
    while cnt < len(events)-1:
        st = events[cnt][0]
        end = events[cnt][1]
        dist1 = events[cnt+1][0] - end 
        condition1 = (dist1<win) if max_dur == False else (dist1<win) and ((events[cnt+1][1]-st) < max_dur*Fs)
        condition2 = any(np.all(data.loc[st:end+dist1, [f'pos_{col}', f'neg_{col}']] == 1, 0))
        if condition1 and condition2:      
            new_events.append((st, events[cnt+1][1]))
            cnt += 2
        else:
            new_events.append((st, end))
            cnt += 1  
    new_events.append((events[-1]))

    # convert back to array
    new_array = events_to_array(new_events, len(data))

    return new_array


