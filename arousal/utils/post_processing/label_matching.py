
import numpy as np
import itertools
from scipy.stats import rankdata
# from arousal.utils.evaluation.event_wise_evalution import *
# from arousal.utils.load_write.label2ids import *

def find_events_in_time_window(primary_label,target_label,arousal_pred_prob,time,Hz):
    
    overlap_events = []
    overlap_events_multi = []
    non_overlap_events = []
    for s,e in primary_label:
        S = s-time*Hz
        E = e+time*Hz

        #find overlap
        overlap_idx = np.where(((target_label[:,0]-S<0) & (target_label[:,1]-S>0)) | ((target_label[:,0]-E<0) & (target_label[:,1]-E>0)) | ((target_label[:,0]-S>0) & (target_label[:,1]-E<0)))[0]
        
        if len(overlap_idx) ==1:
            for o_idx in overlap_idx:
                overlap_events.append([target_label[o_idx,0],target_label[o_idx,1]])
        elif len(overlap_idx) >1:
            #overlap_events_multi.append([s,e])
            for o_idx in overlap_idx:
                overlap_events_multi.append([target_label[o_idx,0],target_label[o_idx,1]])
        else: 
            non_overlap_events.append([s,e])

    if len(np.array(overlap_events_multi))>0:
        events_to_add = find_single_event_for_multi_overlap(np.array(overlap_events_multi),target_label,arousal_pred_prob,time=10,Hz=128)
    else:
        events_to_add=[]


    #return  np.array(overlap_events),np.array(events_to_add),np.array(non_overlap_events)
    
    if overlap_events == []:
        overlap_events = [np.nan,np.nan]

    if events_to_add == []:
        events_to_add = [np.nan,np.nan]
    
    if non_overlap_events == []:
        non_overlap_events = [np.nan,np.nan]
    return np.vstack((np.array(overlap_events),np.array(events_to_add),np.array(non_overlap_events)))

def find_events_a_in_b(a,b,time,Hz):

    try:
        a.shape[1]
    except:
        a=[a]


    # pad with time constraint
    overlap_idx = []
    for s,e in a:
        S = s-time*Hz
        E = e+time*Hz

        #find overlap
        overlap_idx.append(np.where( ((b[:,0]-S<0) & (b[:,1]-S>0)) | ((b[:,0]-E<0) & (b[:,1]-E>0)) | ((b[:,0]-S>0) & (b[:,1]-E<0)) | ((b[:,0]-S==0) & (b[:,1]-E==0)))[0].astype(int))


    return overlap_idx

def find_highest_prob(events,prob,type='mean'):
    prob_ = []
    for s,e in events:
        if type == 'max':
            prob_.append(np.max(prob[s:e]))
        elif type == 'mean':
            prob_.append(np.nanmean(prob[s:e]))
        else:
            print('error')
    
    return events[np.where(prob_==np.max(prob_))[0]][0]

def convert_num2event(event_combi,pred_locations,event_to_use):
    locs = np.array(event_combi)[:,1].astype(int)
    for loc in locs:
        event_to_use.append(pred_locations[loc])
    
    return event_to_use
    print('a')

def find_single_event_for_multi_overlap(multi_overlap_label_event_locations,pred_locations,prob,time,Hz):

    #multiple multi-events
    if len(multi_overlap_label_event_locations)>1:
        event_to_use = []
        double_events = []
        for i, multi_event in enumerate(multi_overlap_label_event_locations):
            #find events corresponding to label
            overlap_idx1 = find_events_a_in_b(multi_event,pred_locations,time,Hz)
            overlap_idx2 = find_events_a_in_b(pred_locations[overlap_idx1[0]],multi_overlap_label_event_locations,time,Hz)
            
            #unique overlap
            if np.max([len(x) for x in overlap_idx2])==1:
                event_to_use.append(find_highest_prob(pred_locations[tuple(overlap_idx1)],prob))

            #non unique overlap                
            else:
                for j in range(len(overlap_idx1[0])):
                    for k in range(len(overlap_idx2[j])):
                        pred = np.mean(prob[pred_locations[overlap_idx1[0][j],0]:pred_locations[overlap_idx1[0][j],1]])
                        double_events.append([overlap_idx2[j][k],overlap_idx1[0][j],pred])
            
        #filter out double rows
        k = sorted(double_events)
        double_events = list(k for k, _ in itertools.groupby(k))
        double_events_ = double_events.copy()

        try:
            rank = rankdata(np.array(double_events)[:,2],method='dense')
            double_events = np.hstack((np.array(double_events),np.array([rank]).T))
            
            event_combi = []
            target = len(np.unique(double_events[:,-1]))-len(np.unique(double_events[:,0]))
            
            while double_events.shape[0]>target:

                    loc = np.where(double_events[:,-1]==np.max(double_events[:,-1]))[0]
                    if len(loc)==1:
                        event_combi.append(double_events[loc[0],:2])
                        loc = np.where(double_events[:,0]==double_events[loc[0],0])
                        double_events = np.delete(double_events,loc,axis=0)
                    else:
                        expert_events = double_events[tuple(loc),0]

                        #check which events overlap with the exp events
                        all_events_ = double_events[tuple(loc),0]
                        expert_events_loc = [x in expert_events for x in double_events[:,0]]
                        expert_events_loc_not = [not elem for elem in expert_events_loc]
                        #check if we need to expand the search
                        expert_events_loc = double_events[expert_events_loc]

                        #if num of events matches num of expert labels, add all
                        if len(np.unique(expert_events_loc[:,1]))==len(np.unique(expert_events_loc[:,0])):
                            temp = np.unique(expert_events_loc[:,1])
                            for j_ in temp:
                                event_combi.append([np.nan,j_])
                            double_events = double_events[expert_events_loc_not]
                            
                        else:
                          
                            if (len(loc)==2) and (len(np.unique(double_events[:,0]))==2):
                                #check means of events
                                mean_1 = np.mean(double_events[np.where(double_events[:,0]==np.unique(double_events[:,0])[0])[0],2])
                                mean_2 = np.mean(double_events[np.where(double_events[:,0]==np.unique(double_events[:,0])[1])[0],2])
                                #if lowest mean, choose max value
                                min = np.argmin((mean_1,mean_2))
                                loc = np.where(double_events[:,2]==np.max(double_events[:,2]))
                                event_combi.append([double_events[loc][min][0],double_events[loc][min][1]])
                                #delete row that are just used
                                loc = np.not_equal(double_events[:,0],double_events[loc][min][0])
                                double_events = double_events[loc,:]
                                #delete row with highes value
                                loc = np.where(double_events[:,2]==np.max(double_events[:,2]))
                                double_events = np.delete(double_events,loc,axis=0)
                                loc = np.where(double_events[:,2]==np.max(double_events[:,2]))
                                event_combi.append([double_events[loc,0],double_events[loc,1]])
                                double_events = np.delete(double_events,loc,axis=0)

                            else:
                                print('a')  
                                break
            
            event_to_use = convert_num2event(event_combi,pred_locations,event_to_use)                     
        except:
            pass

        #TODO add event_combi to event_to_use
    #single multi-event
    else:
        event_to_use = find_highest_prob(multi_overlap_label_event_locations,prob)
    return event_to_use
            
def add_events_to_arousal(label,pred_arousal,regions):

    #loop over regions
    for [s,e] in regions:
        #add prediction in regions (by adding, you dont overwrite the already existing labels)
        # region_pred = pred_arousal[int(s):int(e)]
        # label[int(s):int(e)] = label[int(s):int(e)]+region_pred

        #find overlap
        overlap_idx = np.where(((pred_arousal[:,0]-s<0) & (pred_arousal[:,1]-s>0)) | ((pred_arousal[:,0]-e<0) & (pred_arousal[:,1]-e>0)) | ((pred_arousal[:,0]-s>0) & (pred_arousal[:,1]-e<0)))[0]
            
        #add predictions
        if len(overlap_idx)>0:
            for i in overlap_idx:
                #max 10 sec
                if (pred_arousal[overlap_idx,1][0]-pred_arousal[overlap_idx,0][0])<2000:
                    #min 3 sec
                    if (pred_arousal[overlap_idx,1][0]-pred_arousal[overlap_idx,0][0])>600:
                        label[pred_arousal[overlap_idx,0][0]:pred_arousal[overlap_idx,1][0]] = 1
                    else:
                        center = int(np.mean((pred_arousal[overlap_idx,1][0],pred_arousal[overlap_idx,0][0])))
                        S = center-300
                        E = center+300
                        label[S:E] = 1
                else:
                    center = int(np.mean((pred_arousal[overlap_idx,1][0],pred_arousal[overlap_idx,0][0])))
                    S = center-1000
                    E = center+1000
                    label[S:E] = 1


    return label