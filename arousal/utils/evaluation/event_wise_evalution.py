from ast import AsyncFunctionDef
import numpy as np
import pandas as pd
from glob import glob
from time import time
import matplotlib.pyplot as plt
from arousal.utils.load_write.ids2label import *
from os.path import join as opj
from statistics import mode

from arousal.utils.loaders.preparation_data_for_loader import data_for_loader

##################
# functions
##################

def find_consecutive_numbers(nums):
    """
    returns nx2 array with starting and ending points of consecutive numbers
    """
    nums = sorted(set(nums))
    gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
    edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
    consecutive = np.array(list(zip(edges, edges))) 
    if len(consecutive) != 0:
        consecutive[:,1] +=1
    return consecutive

def find_events(y_true_samplewise,positive_label_array):
    """
    pre-allocate array
    """
    events = np.array([0,0])
    events_labels = 0
    #loop over positive classes and add to arrays
    for lab in positive_label_array:
        #find positive label
        events_temp = find_consecutive_numbers(np.where(y_true_samplewise==lab)[0])
        #add to array
        if events_temp.shape[0]!=0:
            events = np.vstack((events,events_temp))
            events_labels = np.append(events_labels,np.ones(len(events_temp))*lab)
       
    #remove 1st index since it was used for pre-allocation,
    if len(events.shape)>1:
        events = events[1:,:]
        events_labels = events_labels[1:]
    else:
        events = []
        events_labels = []

    return events,events_labels

def find_prob_events(events_idx,y_prob):
    """
    returns mean probabilities for given events
    """
    event_prob = []
    for s,e in events_idx:
        event_prob.append(np.max(np.mean(y_prob[s:e,:],axis=0)))
    return event_prob

def get_biggest_overlap(exp_event,pred_events,poss_event_idx):
    """
    If multiple events, select event with biggest overlap. ignore other evetns
    """
    #crop prediction events to min(pred_events)==min(exp_event) and max(pred_events)==max(exp_event)
    pred_events[pred_events>exp_event[1]]=exp_event[1]
    pred_events[pred_events<exp_event[0]]=exp_event[0]

    #find biggeest event (due to cropping biggest event==biggest overlap)
    loc = np.where(pred_events[1]-pred_events[0]==np.max(pred_events[1]-pred_events[0]))[0]
    
    return poss_event_idx[loc],poss_event_idx[~loc]

def get_misclassified_info(dict,labels,Hz=128):
    FP = []
    FP_before = []
    FP_after = []
    FN = []
    FN_before = []
    FN_after = []
    FP_ct_W = []
    FN_ct_W = []
    
    try:
        fp_idx = dict['pred_events'][dict['FP_event_idx_pred']]
        fn_idx = dict['exp_events'][dict['FN_event_idx_exp']]
    except:
        pass
    
    if 'fp_idx' in locals():
        for i,(s,e) in enumerate(fp_idx):
            lab = np.unique(labels[s:e])
            FP.append(lab)
            lab_b = labels[np.max((s-30*Hz,0)):s]
            FP_before.append(np.unique(lab_b))
            lab_a = labels[e:e+30*Hz]
            FP_after.append(np.unique(lab_a))
            if len(np.where(lab_a == 5)[0])>0 or len(np.where(lab_b == 5)[0])>0:
                FP_ct_W.append(i)

    if 'fn_idx' in locals():
        for i,(s,e) in enumerate(fn_idx):
            lab = np.unique(labels[s:e])
            FN.append(lab)
            lab_b = labels[s-60*Hz:s]
            FN_before.append(np.unique(lab_b))
            lab_a = labels[e:e+60*Hz]
            FN_after.append(np.unique(lab_a))
            if len(np.where(lab_a == 5)[0])>0 or len(np.where(lab_b == 5)[0])>0:
                FN_ct_W.append(i)


    R_count_FN = [i for i, val in enumerate(FN) if len(np.where(val==4)[0])>0]
    N1_count_FN = [i for i, val in enumerate(FN) if len(np.where(val==3)[0])>0]
    N2_count_FN = [i for i, val in enumerate(FN) if len(np.where(val==2)[0])>0]
    N3_count_FN = [i for i, val in enumerate(FN) if len(np.where(val==1)[0])>0]
    R_count_FP = [i for i, val in enumerate(FP) if len(np.where(val==4)[0])>0]
    N1_count_FP = [i for i, val in enumerate(FP) if len(np.where(val==3)[0])>0]
    N2_count_FP = [i for i, val in enumerate(FP) if len(np.where(val==2)[0])>0]
    N3_count_FP = [i for i, val in enumerate(FP) if len(np.where(val==1)[0])>0]
    R_len = len(np.where(labels==4)[0])/Hz/60/60
    W_len = len(np.where(labels==5)[0])/Hz/60/60
    N1_len = len(np.where(labels==3)[0])/Hz/60/60
    N2_len = len(np.where(labels==2)[0])/Hz/60/60
    N3_len = len(np.where(labels==1)[0])/Hz/60/60
    
    return_dict = {'FP':FP,
                'FP_before':FP_before,
                'FP_after':FP_after,
                'FN':FN,
                'FN_before':FN_before,
                'FN_after':FN_after,
                'FP_ct_W':FP_ct_W,
                'FN_ct_W':FN_ct_W,
                'R_count_FN':len(R_count_FN),
                'N1_count_FN':len(N1_count_FN),
                'N2_count_FN':len(N2_count_FN),
                'N3_count_FN':len(N3_count_FN),
                'R_count_FP':len(R_count_FP),
                'N1_count_FP':len(N1_count_FP),
                'N2_count_FP':len(N2_count_FP),
                'N3_count_FP':len(N3_count_FP),
                'R_rate_FN':len(R_count_FN)/(R_len+0.00001),
                'N1_rate_FN':len(N1_count_FN)/(N1_len+0.00001),
                'N2_rate_FN':len(N2_count_FN)/(N2_len+0.00001),
                'N3_rate_FN':len(N3_count_FN)/(N3_len+0.00001),
                'R_rate_FP':len(R_count_FP)/(R_len+0.00001),
                'N1_rate_FP':len(N1_count_FP)/(N1_len+0.00001),
                'N2_rate_FP':len(N2_count_FP)/(N2_len+0.00001),
                'N3_rate_FP':len(N3_count_FP)/(N3_len+0.00001),
                'R_len':R_len,
                'N1_len':N1_len,
                'N2_len':N2_len,
                'N3_len':N3_len,
                'R_len':R_len,
                'N1_len':N1_len,
                'N2_len':N2_len,
                'N3_len':N3_len
                }
    return return_dict

##################
# Binary
##################

def compute_TP_FN_event_IDX(ytrue, ypred, y_prob, masking = [2,2],Hz=128):

    #pre-allocate
    TP_IDX_pred = []
    IGN_IDX_pred = []
    FN_IDX_exp = []
    TP_IDX_exp = []
    FN_IDX_exp = []
    FN_prob = []
    TP_prob = []

    # segment all events in tech array
    exp_events,events_labels = find_events(ytrue,positive_label_array=[1])
    pred_events,pred_labels = find_events(ypred,positive_label_array=[1])
    
    # run over each expert label
    for exp_event_idx,[s,e] in enumerate(exp_events):
        if s==e: 
            print(f"{s, e}. Unexpected, exp event length 1?")
            continue
        
        #add buffer to start and end idx
        S = s-masking[0]*Hz
        E = e+masking[1]*Hz

        #if pred_events - (s or e) crosses zero within one event, there is overlap for that event with the expert event. 
        # OR if pred_events[:,0]-start >0 and pred_events[:,1]-e>0 detection is smaller than expert event but there is overlap
        # if not, event is prior to or after an expert event
        overlap_idx = np.where(((pred_events[:,0]-S<0) & (pred_events[:,1]-S>0)) | ((pred_events[:,0]-E<0) & (pred_events[:,1]-E>0)) | ((pred_events[:,0]-S>=0) & (pred_events[:,1]-E<=0)))[0]

        #FALSE NEGATIVE
        if len(overlap_idx)==0:
            FN_IDX_exp.append(exp_event_idx)
            FN_prob.append(np.nanmean(y_prob[S:E,1]))

        #TRUE POSITIVE
        if len(overlap_idx)==1:
            TP_IDX_exp.append(exp_event_idx)
            TP_IDX_pred.append(overlap_idx[0])
            TP_prob.append(np.mean(y_prob[pred_events[overlap_idx][0],1]))

        #TRUE POSITIVE + ignored events
        if len(overlap_idx)>1:
            idx_used,idx_ign = get_biggest_overlap([s,e],pred_events[overlap_idx],overlap_idx)
            TP_IDX_exp.append(exp_event_idx)
            TP_IDX_pred.append(idx_used[0])
            IGN_IDX_pred.append(idx_ign[0])
            prob = [np.mean(y_prob[pred_events[x][0]:pred_events[x][1],1]) for x in overlap_idx]
            TP_prob.append(np.max(prob))

    outcome_dict =  {
        'exp_events':exp_events,
        'pred_events':pred_events,
        'TP_event_idx_exp':TP_IDX_exp,
        'FN_event_idx_exp':FN_IDX_exp,
        'TP_event_idx_pred':TP_IDX_pred+IGN_IDX_pred,
        'FP_event_idx_pred':[],
        'linked_TP_event_idx_pred':TP_IDX_pred,
        'linked_IGN_IDX_pred':IGN_IDX_pred,
        'count_TP':len(TP_IDX_exp),
        'count_FP':[],
        'count_FN':len(FN_IDX_exp),
        'count_TN':[],
        'prob_TP':TP_prob,
        'prob_FP':[],
        'prob_TN':[],
        'prob_FN':FN_prob}
    return outcome_dict

def compute_FP_event_IDX(dict,y_prob):
    #FP events --> find all predictin events that are not TP
    FP_idx_pred = [x for x in range(len(dict['pred_events'])) if x not in np.array(dict['TP_event_idx_pred'])]
    #
    #find probability per FP event
    prob = [np.mean(y_prob[dict['pred_events'][x][0]:dict['pred_events'][x][1],1]) for x in FP_idx_pred]
    
    #save
    dict['prob_FP'] = prob
    dict['FP_event_idx_pred']=FP_idx_pred
    dict['count_FP']=len(FP_idx_pred)
    return dict

def compute_TN_events(dict,y_true, y_pred,y_prob,event_length=3,Hz=128):
    """
    constraint X: samples are selected if sample wise event label == 0 
                && if sample wise prediction label == 0 
                && if sample wise prediction label is not altered by post processing"""

    #find sample locations using constraint X
    loc_TN_samp = np.where((y_true==0) & (y_pred==0) & (np.isnan(y_prob[:,1])==False))[0]
    #find probability per TN event
    prob = [np.mean(y_prob[loc_TN_samp[x:x+event_length*Hz],1]) for x in range(0,len(loc_TN_samp),int(event_length*Hz))]
    #find amount per TN event
    TN_events = int(np.floor(len(loc_TN_samp)/(event_length*Hz)))

    #save
    dict['count_TN']= TN_events
    dict['prob_TN']= prob[:int(np.floor(TN_events))] # remove last sample if total samples is not exactly a multiple of 3sec*Hz
    return dict

def eventwise_evaluation_linked(y_true_samplewise, y_pred_samplewise, y_pred_prob_sampleswise,Hz):#, y_pred_sampleswise, y_pred_prob_sampleswise, overlap="any"):
    """
    inputs:
    actual_array_samplewise: N samples x 1 dimension. Treated as ground truth.
    predicted_array_sampleswise: N samples x 1 dimension. Treated as predicted array
    probability_array_sampleswise: N samples x C (classes) dimension. Probability vector associated to predicted_array_samplewise.
    positive_label_array: give in array for all classes that are considered as an event. 
    
    output:
    Dictionary with the following arrays:
    NAME:                       Dimention:                      Explanation:
    exp_events       :          K x 2                           Returns the start and end idx of each Expert event.
    pred_events      :          N x 2                           Returns the start and end idx of each Predicted event.
    TP_event_idx_exp :          'Amount of TP in exp' x 1       Returns an array with the idx of the True positive events based on the Expert event trace. 
                                                                exp_events[TP_event_idx_exp] returns the start and end idx of each expert event that overlaps with an predicted event.
    FN_event_idx_exp :          'Amount of FN in exp' x 1       Returns an array with the idx of the False negative events based on the Expert event trace.
                                                                exp_events[FN_event_idx_exp] returns the start and end idx of each expert event that does NOT overlap with an predicted event.
    TP_event_idx_pred:          'Amount of TP in pred' x 1      Returns an array with the idx of the True positive events based on the prediction event trace. 
                                                                pred_events[TP_event_idx_pred] returns the start and end idx of each predicted event that overlaps with an expert event.
    FP_event_idx_pred:          'Amount of TP in pred' x 1      Returns an array with the idx of the False positive events based on the prediction event trace.
                                                                pred_events[FP_event_idx_pred] returns the start and end idx of each predicted event that does NOT overlap with an expert event.
    linked_TP_event_idx_pred:   'Amount of TP in exp' x 1       returns TP_event_idx_pred BUT, only one prediction event can be returned per expert event. if multiple predictin events occur during 1 expert event
                                                                The prediction event with the biggest overlap will be chosen. All other events that overlap with the same expert event will be ignored. 
    linked_IGN_IDX_pred:        'Amount of ingored events x 1'  Ignored events, see explanation linked_TP_event_idx_pred. 
    count_TP:                   '1'                             Amount of True positives (prediction events that do overlap with expert events)
    count_FP:                   '1'                             Amount of False positives (prediction events that do NOT overlap with expert events)
    count_FN:                   '1'                             Amount of False negatives (expert events that do NOT overlap with prediction events)
    count_TN:                   '1'                             Amount of True negatives, this is calculated by the number of 'samples found with constraint X'/(3seconds*Hz) 
                                                                                            constraint X: samples are selected if sample wise event label == 0 
                                                                                            && if sample wise prediction label == 0 
                                                                                            && if sample wise prediction label is not altered by post processing
    prob_TP:                    '1'                             Probability of each TP event (prediction events that do overlap with expert events, if multiple events overlap with 1 expert event, Max prob is taken)
    prob_FP:                    '1'                             Probability of each FP event (prediction events that do NOT overlap with expert events) 
    prob_FN:                    '1'                             Probability of each FN event (expert events that do NOT overlap with prediction events)
    prob_TN:                    '1'                             Probability of each TN event (mean probability of 3seconds*HZ non overlapping windows from sample locations found with constraint X)                                                     
    
    """

    outcome_dict = compute_TP_FN_event_IDX(y_true_samplewise, y_pred_samplewise, y_pred_prob_sampleswise,Hz=Hz)
    outcome_dict = compute_FP_event_IDX(outcome_dict,y_pred_prob_sampleswise)
    outcome_dict = compute_TN_events(outcome_dict,y_true_samplewise, y_pred_samplewise,y_pred_prob_sampleswise,Hz=Hz)

    return outcome_dict


##################
# MULTI CLASS
##################

def compute_TP_FN_event_IDX_multiclass(outcome_dict,CM,positive_label_array,ytrue, ypred, y_prob, negative_label=0,masking = [0,0],Hz=2):

    #pre-allocate variables
    TP_IDX_pred = []
    IGN_IDX_pred = []
    FN_IDX_exp = []
    TP_IDX_exp = []
    FN_IDX_exp = []
    FN_prob = []
    TP_prob = []
    FN_true_lab = []

    #find positive (predictin/expert) events using positive_label_array
    exp_events,exp_event_labels = find_events(ytrue,positive_label_array=positive_label_array)
    pred_events,pred_event_labels = find_events(ypred,positive_label_array=positive_label_array)
    
    # go over each expert label
    for exp_event_idx,[[s,e],lab] in enumerate(zip(exp_events,exp_event_labels)):

        #if start and end are similar, no event
        if s==e:
            print(f"{s}, {e}. Unexpected. Expert label length 1?")
            continue
        
        #if no predictions are found, all expterts events are false negatives
        if len(pred_events) ==0:
            #False Negative
            FN_IDX_exp.append(exp_event_idx)
            # import pdb; pdb.set_trace()
            FN_prob.append(np.nanmean(y_prob[s:e,:],axis=0)) #TODO change row accordingly
            pred_lab = mode(ypred[s:e])
            FN_true_lab.append(int(lab))
            CM[int(lab),int(pred_lab)]+=1

        
        else:
            #When events are present

            #add buffer to start and end idx
            S = np.max((0,s-masking[0]*Hz))
            E = np.min((len(ypred),e+masking[1]*Hz))

            #if pred_events - (start or end of expert event) crosses zero within one event, there is overlap for that event with the expert event. 
            # OR if pred_events[:,0]-start >0 and pred_events[:,1]-e>0 detection is smaller than expert event but there is overlap
            # if not, event is prior to or after an expert event
            correct_pred_idx = np.where(pred_event_labels==lab)[0]
            overlap_idx = np.where(((pred_events[correct_pred_idx,0]-S<=0) & (pred_events[correct_pred_idx,1]-S>=0)) | ((pred_events[correct_pred_idx,0]-E<=0) & (pred_events[correct_pred_idx,1]-E>=0)) | ((pred_events[correct_pred_idx,0]-S>=0) & (pred_events[correct_pred_idx,1]-E<=0)))[0]

            #FALSE NEGATIVE

            if len(overlap_idx)==0:
                pred_lab = np.max(ypred[s:e])
                #add all False negatives, since they can not be added by FP later on.
                if pred_lab == negative_label:
                    #save
                    FN_IDX_exp.append(exp_event_idx)
                    FN_prob.append(np.nanmean(y_prob[S:E,:],axis=0))
                    FN_true_lab.append(int(lab))
                    CM[int(lab),int(pred_lab)]+=1

            #TRUE POSITIVE
            if len(overlap_idx)==1:
                #if only one expert events overlaps 
                #finds overlap between prediction and expert events for probability
                overlap = [i for i in range(S,E+1) if i in range(pred_events[correct_pred_idx[overlap_idx[0]]][0],pred_events[correct_pred_idx[overlap_idx[0]]][1]+1)]
                if len(overlap) == 0: 
                    print("overlap length is zero, not expected")
                #save
                TP_IDX_exp.append(exp_event_idx)
                TP_IDX_pred.append(correct_pred_idx[overlap_idx[0]])
                try:
                    TP_prob.append(np.nanmean(y_prob[overlap,:],axis=0))#if the recording ends with a positive prediction, this will raise an error,but is needed for 1 idx overlap
                except:
                    TP_prob.append(np.nanmean(y_prob[overlap[:-1],:],axis=0))
                CM[int(lab),int(lab)]+=1

            #TRUE POSITIVE + ignored events
            if len(overlap_idx)>1:
                #if multiple overlaps are present, find the best predition for linking to the expert event.
                idx_used,idx_ign = get_biggest_overlap([s,e],pred_events[overlap_idx],overlap_idx)
                #save
                TP_IDX_exp.append(exp_event_idx)
                TP_IDX_pred.append(correct_pred_idx[idx_used[0]]) #linked event
                IGN_IDX_pred.append(correct_pred_idx[idx_ign[0]]) #ignored event
                #find overlap for all segments
                overlap = []
                for o_idx in overlap_idx:
                    overlap_ = [i for i in range(S,E+1) if i in range(pred_events[correct_pred_idx[o_idx]][0],pred_events[correct_pred_idx[o_idx]][1]+1)]
                    if len(overlap_)>0:
                        overlap += overlap_
                    else:
                        print('a')
                #overlap = np.array(overlap).reshape(-1)
                #save probability using the mean of all segments overlap
                try:
                    TP_prob.append(np.nanmean(y_prob[overlap,:],axis=0))#if the recording ends with a positive prediction, this will raise an error,but is needed for 1 idx overlap
                except:
                    TP_prob.append(np.nanmean(y_prob[overlap[:-1],:],axis=0))
                CM[int(lab),int(lab)]+=1

    #save everything in a dictionary 
    outcome_dict =  {
        'exp_events':exp_events,
        'exp_event_labels':exp_event_labels,
        'pred_events':pred_events,
        'pred_event_labels':pred_event_labels,
        'TP_event_idx_exp':TP_IDX_exp,
        'FN_event_idx_exp':FN_IDX_exp,
        'TP_event_idx_pred':TP_IDX_pred+IGN_IDX_pred,
        'FP_event_idx_pred':[],
        'FP_true_label':[],
        'FN_true_label':FN_true_lab,
        'linked_TP_event_idx_pred':TP_IDX_pred,
        'linked_IGN_IDX_pred':IGN_IDX_pred,
        'count_TP':len(TP_IDX_exp),
        'count_FP':[],
        'count_FN':len(FN_IDX_exp),
        'count_TN':[],
        'prob_TP':TP_prob,
        'prob_FP':[],
        'prob_TN':[],
        'prob_FN':FN_prob}
    return outcome_dict, CM

def compute_FP_event_IDX_multiclass(dict,CM,y_true,y_prob,y_pred):
    #FP events --> find all prediction events that are incorrect AND do have overlap with an expert event
    # incorrect predicted expert events that are predected as no event are conciderd FN!!
    FP_prob = []
    FP_idx = []
    FP_true_lab = []
    #pre-allocate
    exp_events = dict['exp_events']

    if len(exp_events) == 0:
        exp_events = np.array([[np.nan, np.nan]])

    #find FP events based on expert event
    for i,[[s,e],pred_lab] in enumerate(zip(dict['pred_events'],dict['pred_event_labels'])):
        
        if i in dict['TP_event_idx_pred']:
            continue
        
        #finds overlab between expert event and current prediction event
        overlap_idx = np.where(((exp_events[:,0]-s<=0) & (exp_events[:,1]-s>=0)) | ((exp_events[:,0]-e<=0) & (exp_events[:,1]-e>=0)) | ((exp_events[:,0]-s>=0) & (exp_events[:,1]-e<=0)))[0]

        #if there is no expert event overlap and pred event is a TP skip
        #if there is overlap, and the event is a TP, it can also be a FP. (very long predictions, covering multiple experts events)
        if len(overlap_idx)==0 and i in dict['TP_event_idx_pred']: 
            continue

        #if there is overlap
        elif len(overlap_idx)>0:
            for s_o,e_o in exp_events[overlap_idx]:
                if s_o==e_o: 
                    print(f"Unexpected, {s_o}, {e_o}. If expert events are longer than 1 sample, s_o is expected to be unequal to e_o. Expert event length 1 in the data?")
                #for each overlapping event find the expert label
                exp_lab = mode(y_true[s_o : e_o])
                #if expert label is predicted label, it is an TP
                if exp_lab != pred_lab:
                    #save
                    FP_true_lab.append(exp_lab)
                    CM[int(exp_lab),int(pred_lab)]+=1
                    overlap = [i for i in range(s,e) if i in range(s_o,e_o)]
                    FP_prob.append(np.nanmean(y_prob[overlap,:],axis=0))
                    FP_idx.append(i)

        else:
            #no overlap and also no TP. SO --> FP
            if s==e: print(f"Unexpected, {s}, {e}. If predicted events are longer than 1 sample, s is expected to be unequal to e. Predicted event length 1 in the data?")
            FP_prob.append(np.nanmean(y_prob[s:e,:],axis=0))
            FP_idx.append(i)
            FP_true_lab.append(0)
            CM[0,int(pred_lab)]+=1

    #save
    dict['prob_FP'] = FP_prob
    dict['FP_event_idx_pred']=FP_idx
    dict['count_FP']=len(FP_prob)
    dict['FP_true_label']=FP_true_lab
    
    return dict, CM

def compute_TN_events_multiclass(dict,CM,y_true, y_pred,y_prob,event_length=3,Hz=128,neg_lab=0):
    """
    constraint X: samples are selected if expert label == 0 
                && if prediction label == 0 
                && if prediction label is not altered by post processing"""

    #find sample locations using constraint X
    loc_TN_samp = np.where((y_true==0) & (y_pred==0) & (np.isnan(y_prob[:,1])==False))[0] #y_prob[:,1] is for calculation purposes
    #find probability per TN event
    prob = [np.mean(y_prob[loc_TN_samp[x:x+event_length*Hz],:],axis=0) for x in range(0,len(loc_TN_samp),int(event_length*Hz))]
    #find amount per TN event
    TN_events = int(np.floor(len(loc_TN_samp)/(event_length*Hz)))

    #save
    CM[neg_lab,neg_lab]+=TN_events
    dict['count_TN']= TN_events
    dict['prob_TN'] = prob[:int(np.floor(TN_events))] # remove last sample if total samples is not exactly a multiple of 3sec*Hz
    return dict, CM

def eventwise_evaluation_multiclass(label_dict, y_true_samplewise, y_pred_samplewise, y_pred_prob_sampleswise, Hz,mask=[0,0]):
    """
    inputs:
    label_dictionary: ['pos_labels']= list with event labels, ['neg_labels']=list with negative label (mostly zero)
    actual_array: N samples x 1 dimension. Treated as ground truth.
    predicted_array: N samples x 1 dimension. Treated as predicted array
    probability_array: N samples x C (classes) dimension. Probability vector associated to predicted_array_samplewise.
    positive_label_array: give in array for all classes that are considered as an event. 
    Hz: the sample frequency
    
    output:
    Dictionary with the following arrays:
    NAME:                       Dimention:                              Explanation:
    exp_events       :          K x 2                                   Returns the start and end idx of each Expert event.
    exp_event_labels :          K x 1                                   Returns the labels for each expert event.
    pred_events      :          N x 2                                   Returns the start and end idx of each Predicted event.
    pred_event_labels :         N x 1                                   Returns the labels for each expert event.
    TP_event_idx_exp :          'Amount of TP in exp' x 1               Returns an array with the idx of the True positive events based on the Expert event trace. 
                                                                        exp_events[TP_event_idx_exp] returns the start and end idx of each expert event that overlaps with an predicted event.
    FN_event_idx_exp :          'Amount of FN in exp' x 1               Returns an array with the idx of the False negative events based on the Expert event trace. FN's are only FN if event is missed by predictions.
                                                                        exp_events[FN_event_idx_exp] returns the start and end idx of each expert event that does NOT overlap with an predicted event.
                                                                        NOTE: If an event is incorrectly predicted and the incorrect prediction is an event. the event is conciterd a FP!
    TP_event_idx_pred:          'Amount of TP in pred' x 1              Returns an array with the idx of the True positive events based on the prediction event trace. 
                                                                        pred_events[TP_event_idx_pred] returns the start and end idx of each predicted event that overlaps with an expert event.
    FP_event_idx_pred:          'Amount of TP in pred' x 1              Returns an array with the idx of the False positive events based on the prediction event trace.
                                                                        pred_events[FP_event_idx_pred] returns the start and end idx of each predicted event that does NOT overlap with an expert event.
    linked_TP_event_idx_pred:   'Amount of TP in exp' x 1               returns TP_event_idx_pred BUT, only one prediction event can be returned per expert event. if multiple predictin events occur during 1 expert event
                                                                        The prediction event with the biggest overlap will be chosen. All other events that overlap with the same expert event will be ignored. 
    linked_IGN_IDX_pred:        'Amount of ingored events x 1'          Ignored events, see explanation linked_TP_event_idx_pred. 
    count_TP:                   '1'                                     Amount of True positives (prediction events that do overlap with expert events)
    count_FP:                   '1'                                     Amount of False positives (prediction events that do NOT overlap with expert events)
    count_FN:                   '1'                                     Amount of False negatives (expert events that do NOT overlap with prediction events)
    count_TN:                   '1'                                     Amount of True negatives, this is calculated by the number of 'samples found with constraint X'/(3seconds*Hz) 
                                                                                            constraint X: samples are selected if sample wise event label == 0 
                                                                                            && if sample wise prediction label == 0 
                                                                                            && if sample wise prediction label is not altered by post processing
    prob_TP:                    'len(count_TP) x Camount of classes)    Probability of each TP event (prediction events that do overlap with expert events, if multiple events overlap with 1 expert event, Max prob is taken)
    prob_FP:                    'L x C'                                 Probability of each FP event (prediction events that do NOT overlap with expert events) 
    prob_FN:                    'O x C'                                 Probability of each FN event (expert events that do NOT overlap with prediction events)
    prob_TN:                    'P x C'                                 Probability of each TN event (mean probability of 3seconds*HZ non overlapping windows from sample locations found with constraint X)                                                     
    
    """
    
    num_labels = len(label_dict['pos_labels'])+len(label_dict['neg_labels'])
    pos_labels = label_dict['pos_labels']
    neg_labels = label_dict['neg_labels'][0]

    Cm = np.zeros((num_labels,num_labels))
   
    outcome_dict,Cm = compute_TP_FN_event_IDX_multiclass(label_dict,Cm,pos_labels,y_true_samplewise, y_pred_samplewise, y_pred_prob_sampleswise,negative_label=neg_labels, masking = mask,Hz=Hz)
    outcome_dict,Cm = compute_FP_event_IDX_multiclass(outcome_dict,Cm,y_true_samplewise,y_pred_prob_sampleswise,y_pred_samplewise)
    outcome_dict,Cm = compute_TN_events_multiclass(outcome_dict,Cm,y_true_samplewise, y_pred_samplewise,y_pred_prob_sampleswise,Hz=Hz)

    return outcome_dict, Cm