from arousal.utils.evaluation.event_wise_evalution import find_consecutive_numbers,find_events
import numpy as np

def crop_label(data,k):
    return data[:len(data)//k*k]
    
def moving_average(a, n=3,max=1,min=0,center=False):
    b = a.copy()
    b[b>=max]=max
    b[b<=min]=min
    ret = np.cumsum(b, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    if center:
        s = [np.mean(b[:x]) for x in range(1,int(n/2))]
        e = [np.mean(b[n-x:]) for x in  range(int(n/2),n)]
        return np.append(np.append(np.array(s),ret),np.array(e))
    else:
        s = [np.mean(b[:x]) for x in range(1,n)]
        return np.append(np.array(s),ret)
    

def movav(data_array,window_width,max_val=None,min_val=None,center=True):
    
    #make sure we dont influence the original data
    data = data_array.copy()

    #clip data
    if not max_val == None:
        data[data>max_val]=max_val
    if not min_val == None:
        data[data<min_val]=min_val

    #pre-allocate
    moving_average_vector = np.zeros(len(data))
    #get vector
    cumsum_vec = np.cumsum(np.insert(data, 0, 0))
    #mov_av raw.
    mov_av_count =  (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) 
    mov_av_half_count =  (cumsum_vec[int(window_width/2):] - cumsum_vec[:-int(window_width/2)]) 
    mov_av = mov_av_count / window_width

    #get vector flipLR
    cumsum_vec_lr = np.cumsum(np.insert(np.flip(data), 0, 0))
    mov_av_half_flip_count =  (cumsum_vec_lr[int(window_width/2):] - cumsum_vec_lr[:-int(window_width/2)]) 
    
    if center:
        #get lengths
        mov_av_time_trace_start = np.arange(int(window_width/2))+1
        
        #get changing mov_av start
        mov_av_start_count = cumsum_vec[:int(window_width/2)]
        mov_av_start_count2 = mov_av_half_flip_count[-int(window_width/2):]
        mov_av_start = (mov_av_start_count+mov_av_start_count2)/(mov_av_time_trace_start+int(window_width/2))

        #get changing mov_av end
        mov_av_end_count = cumsum_vec_lr[:int(window_width/2)]
        mov_av_end_count2 = mov_av_half_count[-int(window_width/2):]
        mov_av_end = (mov_av_end_count+np.flip(mov_av_end_count2))/(mov_av_time_trace_start+int(window_width/2))
        
        #build final vector
        moving_average_vector[:len(mov_av_start)]=mov_av_start
        moving_average_vector[len(mov_av_start):len(mov_av_start)+len(mov_av)]=mov_av
        moving_average_vector[-len(mov_av_end):]=np.flip(mov_av_end)

        return moving_average_vector
    
    else:
        #get lengths
        mov_av_time_trace_start = np.arange(int(window_width))
        
        #get changing mov_av
        mov_av_start = cumsum_vec[:int(window_width)]/mov_av_time_trace_start
        
        #build final vector
        moving_average_vector[:len(mov_av_start)]=mov_av_start
        moving_average_vector[len(mov_av_start):]=mov_av
        #remove 0's in first and last place
        moving_average_vector[0]=moving_average_vector[1]
        moving_average_vector[-1]=moving_average_vector[-2]
        
        return moving_average_vector
    

    

def smooth_pred_prob(pred,N=16,round=True,threshold_1=0.88):
    pred = pred[:,1]
    pred_ = moving_average(pred,n=N)
    if round:
        pred_ =np.round(pred_-threshold_1+0.5)
    return pred_


def smooth_pred(pred,N_movav=128,N_merge_dist=1280,N_remove_length=383,threshold_1=0.5,threshold_2=0.88,Hz=128):

    pred_smooth = smooth_pred_prob(pred,N=N_movav,threshold_1=threshold_1)
    #find events
    events_temp = find_consecutive_numbers(np.where(pred_smooth==1)[0])
    #check len
    events_temp_len_gap = events_temp[1:,0]-events_temp[:-1,1]
    #if closer than 1.5 sec merge
    events_temp_len_gap[events_temp_len_gap>N_merge_dist]=0
    events_temp_len_gap= np.append(events_temp_len_gap,0)
    events_temp_ = events_temp.copy()
    events_temp_[:,1]+=events_temp_len_gap

    #create new predicion array
    pred_ = np.zeros(len(pred_smooth))
    for s,e in events_temp_:
        pred_[s:e+1]=1
    
    #same as above but remove if shorter than 2 seconds
    events_temp_ = find_consecutive_numbers(np.where(pred_==1)[0])
    events_temp_len = events_temp_[:,1]-events_temp_[:,0]
    idx = np.where(events_temp_len>N_remove_length)[0]
    events_temp_1= events_temp_[idx,:]

    #create new predicion array
    pred_2 = np.zeros(len(pred_))
    for s,e in events_temp_1:
        
        if np.mean(pred_smooth[s:e])>=threshold_2:
            if e-s >15*Hz:
                #len / 15 sec
                cuts = (e-s)//(15*Hz)
                #len / cuts+1 (cuts+1 = segments)
                cut_len = np.floor((e-s)/(cuts+1))
                pred_2[s:e]=1
                cut = s+cut_len
                while cut<e-cuts:
                    pred_2[int(cut):int(cut+2)]=0
                    cut+=cut_len
            else:
                pred_2[s:e+1]=1       

    return pred_2, smooth_pred_prob(pred,N=N_movav,round=False)

def binarize_prediction(pred,N_movav=10,N_merge_dist=1,N_remove_length=1,Hz=128):

    #binarize_prediction
    mov_av_thres = moving_average(pred,n=Hz*N_movav,max=0.4,min=0.2)
    mov_av_thres = mov_av_thres*1.15
    prediction_binary = np.round(pred-mov_av_thres+0.4)

    #find events
    events_temp = find_consecutive_numbers(np.where(prediction_binary==1)[0])
    #check len
    events_temp_len_gap = events_temp[1:,0]-events_temp[:-1,1]
    #if closer than 1.5 sec merge
    events_temp_len_gap[events_temp_len_gap>int(N_merge_dist*Hz)]=0
    events_temp_len_gap= np.append(events_temp_len_gap,0)
    events_temp_ = events_temp.copy()
    events_temp_[:,1]+=events_temp_len_gap

    #create new predicion array
    pred_ = np.zeros(len(prediction_binary))
    for s,e in events_temp_:
        pred_[s:e+1]=1
    
    #same as above but remove if shorter than 2 seconds
    events_temp_ = find_consecutive_numbers(np.where(pred_==1)[0])
    events_temp_len = events_temp_[:,1]-events_temp_[:,0]
    idx = np.where(events_temp_len>int(N_remove_length*Hz))[0]
    events_temp_1= events_temp_[idx,:]

    #create new predicion array
    pred_2 = np.zeros(len(pred_))
    for s,e in events_temp_1:
        pred_2[s:e]=1       

    return pred_2

def post_process_after_smoothing(pred,positive_lab_array,N_remove_length=2.5,N_max_len=15,Hz=2,N_merge_dist=int(3)):
    N_merge_dist = N_merge_dist*Hz
    #remove long predictions
    events_temp_, labels = find_events(pred,positive_lab_array)
    if len(events_temp_) == 0:
        return pred
    
    #check len
    events_temp_len_gap = events_temp_[1:,0]-events_temp_[:-1,1]
    #if closer than 3 sec merge
    events_temp_len_gap[events_temp_len_gap>N_merge_dist]=0
    events_temp_len_gap= np.append(events_temp_len_gap,0)
    events_temp_ = events_temp_.copy()
    events_temp_[:,1]+=events_temp_len_gap

    #create new predicion array
    pred_ = np.zeros(len(pred))
    for s,e in events_temp_:
        pred_[s:e]=1

    events_temp_, labels = find_events(pred_,positive_lab_array)
    pred_1 = np.zeros(len(pred_))
    for [s,e],lab in zip(events_temp_,labels):
        
        if e-s >(N_max_len*Hz):
            #len / 15 sec
            cuts = (e-s)//(N_max_len*Hz)
            #len / cuts+1 (cuts+1 = segments)
            cut_len = np.floor((e-s)/(cuts+1))
            pred_1[s:e]=lab
            cut = s+cut_len
            while cut<e-cuts:
                pred_1[int(cut-1):int(cut+1)]=0
                cut+=cut_len
        else:
            pred_1[s:e]=lab   

    #same as above but remove if shorter than N seconds
    events_temp_, labels = find_events(pred_1,positive_lab_array)
    events_temp_len = events_temp_[:,1]-events_temp_[:,0]
    idx = np.where(events_temp_len>int(N_remove_length*Hz))[0]
    events_temp_ = events_temp_[idx,:]
    labels = labels[idx]

    #create new predicion array
    pred_2 = np.zeros(len(pred))
    for [s,e],lab in zip(events_temp_,labels):
        pred_2[s:e]=lab   

    return pred_2

def remove_small_predictions(pred,positive_lab_array,N_remove_length=383):

    #same as above but remove if shorter than N seconds
    events_temp_, labels = find_events(pred,positive_lab_array)
    if len(events_temp_) == 0:
        return pred
    events_temp_len = events_temp_[:,1]-events_temp_[:,0]
    idx = np.where(events_temp_len>N_remove_length)[0]
    events_temp_ = events_temp_[idx,:]
    labels = labels[idx]

    #create new predicion array
    pred_2 = np.zeros(len(pred))
    for [s,e],lab in zip(events_temp_,labels):
        pred_2[s:e+1]=lab       

    return pred_2

def exclude_prob(arousal_pred,enhanced_label_arousal,arousal_pred_prob):
    loc = np.where((arousal_pred==1) & (enhanced_label_arousal==0))[0]
    arousal_pred_prob_nan = arousal_pred_prob.copy()
    arousal_pred_prob_nan[loc,:]=np.nan
    return arousal_pred_prob_nan

def exclude_prob_1d(arousal_pred,enhanced_label_arousal,arousal_pred_prob,set_to=np.nan):
    loc = np.where((arousal_pred==1) & (enhanced_label_arousal==0))[0]
    loc = np.unique([loc,loc+1,loc-1])
    arousal_pred_prob_nan = arousal_pred_prob.copy()
    arousal_pred_prob_nan[loc]=set_to
   
    return arousal_pred_prob_nan

def expert2prob(data):
    data = np.vstack((data,data)).T
    return data