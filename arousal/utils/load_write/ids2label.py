#from tkinter import W
import numpy as np 
import h5py as h5 
from glob import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
from arousal.utils.load_write.label2ids import label_num2str

# def label_num2str(stage_num,label_type):
#     if label_type == 'hypnogram':
#         if stage_num == 5:
#             stage_str = 'W'
#         elif stage_num == 4:
#             stage_str = 'REM'
#         elif stage_num == 3:
#             stage_str = 'N1'
#         elif stage_num == 2:
#             stage_str = 'N2'
#         elif stage_num == 1:
#             stage_str = 'N3'
#         else:
#             stage_str = 'UNKNOWN'

#     if label_type == 'arousal' or label_type == 'arousal_enhanced' or label_type == 'arousal_enhanced_V2':
#         if stage_num == 1:
#             stage_str = 'arousal'
#         elif stage_num == 2:
#             stage_str = 'uncertain'
#         else:
#             stage_str = 'UNKNOWN'
        
#     return stage_str



def ids_to_hyp(ids,label):
    #pre-allocate
    lab = np.empty((len(label)))

    #create hypno
    for i in range(len(ids)):
        ids_ = ids[i]
        arr = ids_.split(',')
        if arr[2] == 'W':
            arr[2] = 5
        if arr[2] == 'REM':
            arr[2] = 4
        if arr[2] == 'N1':
            arr[2] = 3
        if arr[2] == 'N2':
            arr[2] = 2
        if arr[2] == 'N3':
            arr[2] = 1
        if arr[2] == 'arousal':
            arr[2] = 1
        if arr[2] == 'uncertain':
            arr[2] = 2
        if arr[2] == 'UNKNOWN':
            arr[2] = 0
        

        lab[int(float(arr[0])*200):int(float(arr[0])*200+float(arr[1])*200)] = arr[2]
    lab = lab[:int(float(arr[0])*200)+int(float(arr[1])*200)]
    

    plt.plot(lab)
    plt.plot(label)
    plt.show()
        
def len_lines(lines):
    lst =  [pos for pos, char in enumerate(lines) if char == ',']

    return float(lines[:lst[0]])+ float(lines[lst[0]+1:lst[1]])

def create_hypno(ids,fs=128):

    #length last line 
    lst =  [pos for pos, char in enumerate(ids[-1]) if char == ',']
    #add start + length of last line
    len_ids = float(ids[-1][:lst[0]])+ float(ids[-1][lst[0]+1:lst[1]])
    lab = np.empty((int(len_ids)*fs,))

    #create hypno
    for i in range(len(ids)):
        ids_ = ids[i]
        arr = ids_.split(',')
        if arr[2] == 'W' or arr[2] == 'W\n' or arr[2] == 'Sleep stage W\n' or arr[2] == 'Sleep stage W' or arr[2]=='RERA' or arr[2]=='RERA\n':
            arr[2] = 5
        if arr[2] == 'REM' or arr[2] == 'REM\n' or arr[2] == 'Sleep stage R\n' or arr[2] == 'Sleep stage R' or arr[2]=='hypopnia' or arr[2]=='hypopnia\n':
            arr[2] = 4
        if arr[2] == 'N1' or arr[2] == 'N1\n' or arr[2] == 'Sleep stage 1\n' or arr[2] == 'Sleep stage 1' or arr[2]=='mixed' or arr[2]=='mixed\n' or arr[2] == 'arousal3' or arr[2] == 'arousal3\n':
            arr[2] = 3
        if arr[2] == 'N2' or arr[2] == 'N2\n' or arr[2] == 'Sleep stage 2\n' or arr[2] == 'Sleep stage 2'or arr[2]=='central' or arr[2]=='central\n' or arr[2] == 'arousal2' or arr[2] == 'arousal2\n':
            arr[2] = 2
        if arr[2] == 'N3' or arr[2] == 'N3\n' or arr[2] == 'Sleep stage 3\n' or arr[2] == 'Sleep stage 4\n' or arr[2] == 'Sleep stage 3' or arr[2] == 'Sleep stage 4' or arr[2] == 'arousal' or arr[2] == 'arousal\n' or arr[2]=='obstructive' or arr[2]=='obstructive\n' or arr[2] == 'movement\n' or arr[2] == 'movement':
            arr[2] = 1
        if arr[2] == 'UNKNOWN' or arr[2] == 'UNKNOWN\n' or arr[2] == '?\n' or arr[2] == '?' or arr[2] == 'Movement time\n' or arr[2] == 'Sleep stage ?\n' :
            arr[2] = 0
        if arr[2] == 'uncertain' or arr[2] == 'uncertain\n':
            arr[2] = 2
            
        lab[int(float(arr[0])*fs):int(float(arr[0])*fs+float(arr[1])*fs)] = arr[2]
    return lab

def read_label(path,fs=128):
    with open(path,'r') as f:
        ids = f.readlines()
        hypno = create_hypno(ids,fs)
    return hypno