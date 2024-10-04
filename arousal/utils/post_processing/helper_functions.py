import numpy as np
from arousal.utils.evaluation.event_wise_evalution import find_consecutive_numbers, find_events

def line_len(array,idx1):

    line_len = np.append(0,np.cumsum(np.abs(array)))
    array_b = np.zeros(len(line_len))
    array_b[:-idx1] = line_len[idx1:]
    array_b[-idx1:] = array_b[-idx1-1]
    return array_b-line_len

def flatline_detection_removal(eeg,prediction):

    #pre-allocate arrays
    flat_line_detection = np.zeros((len(eeg[0,:]))) #binary
    flat_line_detection_all = np.zeros((len(eeg[0,:]))) #+1 per channel 

    for c in range(eeg.shape[0]):

        #find flatlines
        line_len_seg = np.abs(np.diff(eeg[c,:]))
        line_len_seg2 = np.abs(np.diff(line_len_seg)) 

        #line len in 1st percentile that is not 0 == flat
        pers_value = np.percentile(line_len_seg2,[0.5,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]) #due to varience in data multiple thresholds are chosen 
        pers_loc = np.where(pers_value)[0]
        thres_val = pers_value[pers_loc[0]]
        loc = np.where(line_len_seg2<thres_val)[0]

        #find length of flatelines
        locs = find_consecutive_numbers(loc)
        len_locs = locs[:,1]-locs[:,0]

        #find eeg values
        for s,e in locs[len_locs>5]:
            #make sure the flatelines are not clipped
            value_list= np.array((np.min(eeg[c,:]),0,np.max(eeg[c,:])))
            max_check = np.where( np.abs(value_list-np.mean(np.squeeze(eeg[c,s:e]))) == min(np.abs(value_list-np.mean(np.squeeze(eeg[c,s:e])))) )[0][0]
            
            #if not clipped
            if max_check==1:
                flat_line_detection[s:e]=1
                flat_line_detection_all[s:e]+=1


    flat_line_detection_all[flat_line_detection_all<4]=0
    flat_line_detection_all[flat_line_detection_all>0]=1

    f_events = find_events(flat_line_detection_all,[1])[0]
    events = find_events(prediction,[1])[0]

    if len(f_events) >0:
        idx_to_rem = []
        for exp_event_idx,[s,e] in enumerate(events):
            if s==e: 
                print(f"{s, e}. Unexpected, exp event length 1?")
                continue
            
            #add buffer to start and end idx
            S = s-(0*128)
            E = e+(0*128)

            overlap_idx = np.where(((f_events[:,0]-S<0) & (f_events[:,1]-S>0)) | ((f_events[:,0]-E<0) & (f_events[:,1]-E>0)) | ((f_events[:,0]-S>0) & (f_events[:,1]-E<0)))[0]

            if len(overlap_idx)>0:
                idx_to_rem.append(exp_event_idx)

        for s,e in events[idx_to_rem,:]:
            prediction[s-1:e+1] = 0


    return prediction
