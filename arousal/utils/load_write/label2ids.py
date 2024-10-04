#from tkinter import W
import numpy as np 
import h5py as h5 
from glob import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

def label_num2str(stage_num,label_type):
    if label_type == 'hypnogram':
        if stage_num == 5:
            stage_str = 'W'
        elif stage_num == 4:
            stage_str = 'REM'
        elif stage_num == 3:
            stage_str = 'N1'
        elif stage_num == 2:
            stage_str = 'N2'
        elif stage_num == 1:
            stage_str = 'N3'
        else:
            stage_str = 'UNKNOWN'
            
    if label_type == 'arousal' or label_type == 'arousal_enhanced' or label_type == 'arousal_enhanced_V2':
        if stage_num == 3:
            stage_str = 'arousal3'
        elif stage_num == 2:
            stage_str = 'arousal2'
        elif stage_num == 1:
            stage_str = 'arousal'
        elif stage_num == 0:
            stage_str = 'UNKNOWN'    
        else:
            stage_str = 'uncertain'
    
    if label_type == 'limb' or label_type == 'limb_enhanced_V2':
        if stage_num == 1:
            stage_str = 'movement'
        elif stage_num == 0:
            stage_str = 'UNKNOWN'    
        else:
            stage_str = 'uncertain'

    if label_type == 'resp' or label_type == 'resp_enhanced_V2':
  
        if stage_num == 5:
            stage_str = 'RERA'
        elif stage_num == 4:
            stage_str = 'hypopnia'
        elif stage_num == 3:
            stage_str = 'mixed'
        elif stage_num == 2:
            stage_str = 'central'
        elif stage_num == 1:
            stage_str = 'obstructive'
        elif stage_num == 0:
            stage_str = 'UNKNOWN'    
        else:
            stage_str = 'uncertain'

    if label_type == 'spindle':
        if stage_num == 1:
            stage_str = 'spindle'
        elif stage_num == 0:
            stage_str = 'UNKNOWN'    
        else:
            stage_str = 'uncertain'
        
    return stage_str

def to_ids(label,sample_rate,error='fake',patname='',type_='hypnogram'):
    if len(np.argwhere(np.isnan(label)))>0:
        label[np.argwhere(np.isnan(label))]=-9
    #find stage transitions
    a = np.where(np.diff(label)!=0)[0]

    #create ids file
    ids = []
    start = 0
    for i in a:
        stage_num = label[i-1]
        stage_str = label_num2str(stage_num,type_)
        if type_ == 'hypnogram':
            if int((i-start+1)/sample_rate)%30>0:
                pass
                #error.append(patname)
        ids.append(str(start/sample_rate) +','+str((i-start+1)/sample_rate)+','+stage_str)
        start = i+1
    #add last stage (no diff anymore so it is not detected by np.diff)
      
    stage_str = label_num2str(label[-1],type_)

    ids.append(str(start/sample_rate) +','+str( (len(label)/sample_rate) - (start/sample_rate))+','+stage_str)   

    return ids,error

        
def len_lines(lines):
    lst =  [pos for pos, char in enumerate(lines) if char == ',']

    return float(lines[:lst[0]])+ float(lines[lst[0]+1:lst[1]])