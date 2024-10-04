from arousal.utils.post_processing.helper_functions import line_len
from arousal.utils.post_processing.smoothing_arousal import movav
from arousal.utils.evaluation.event_wise_evalution import find_events

import numpy as np


def rem_post_processing(emg,predictions,hypnogram,output_pp_trace=False):

    #########################
    # Line length calc
    ########################
    ll = np.abs(np.diff(emg))
    ll = line_len(ll,1*128)
    ll2 = line_len(ll,3*128)

    #mov av for thres 
    ll2_mov = movav(ll2,window_width=128*20,center=True)
    #increase by 15% to get rid of small increases
    ll2_mov = ll2_mov*1.15

    #threshold
    ll2_thes = ll2-ll2_mov
    ll2_thes[ll2_thes<0]=0
    ll2_thes[ll2_thes>0]=1

    #########################
    # find events
    #########################
    #find increased EMG activity events
    events = find_events(ll2_thes,[1])[0]
    
    #predictions
    events_pred = find_events(predictions,[1])[0]

    if len(events_pred)>0:
        #find overlap
        idx_to_hold = []
        for exp_event_idx,[s,e] in enumerate(events_pred):
            if s==e: 
                print(f"{s, e}. Unexpected, exp event length 1?")
                continue
            
            #add buffer to start and end idx
            S = s-(0*128)
            E = e+(0*128)

            overlap_idx = np.where(((events[:,0]-S<0) & (events[:,1]-S>0)) | ((events[:,0]-E<0) & (events[:,1]-E>0)) | ((events[:,0]-S>0) & (events[:,1]-E<0)))[0]

            if len(overlap_idx)>0:
                idx_to_hold.append(exp_event_idx)

        #create new label
        events_p_trace = np.zeros(len(predictions))
        for s,e in events_pred[idx_to_hold,:]:
            if np.max(ll2[s:e])>=0.07:
                events_p_trace[s:e]=1
    else:
        events_p_trace = np.zeros(len(predictions))

    #allow all other stages exept wakefullness
    nrem_loc = np.where(hypnogram!=4)[0]
    events_p_trace[nrem_loc]=1
    wake_loc = np.where(hypnogram==5)[0]
    events_p_trace[wake_loc]=0

    #mask prediction
    pred = events_p_trace*predictions
    
    if output_pp_trace:
        return events_p_trace
    else:
        return pred