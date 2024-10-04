import keras
import numpy as np
import h5py as h5
from glob import glob
import os
import matplotlib.pyplot as plt

from arousal.utils.load_write.cvs2label import cvs2lab
from arousal.utils.load_write.ids2label import *
from scipy.signal import decimate
from os.path import join as opj
import random
import time
#import ray
from multiprocessing import current_process
from time import time


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    ret = ret[n - 1:] / n
    b = [np.mean(a[:x]) for x in range(1,n)]
    return np.append(np.array(b),ret)

def get_patients(data_list_per_set, chance_list, batch_size=64):

    if batch_size>len(data_list_per_set):
        pat_list_idx = np.random.choice(len(data_list_per_set),batch_size,replace=True,p=chance_list)
    else:
        pat_list_idx = np.random.choice(len(data_list_per_set),np.min((batch_size*4,len(data_list_per_set))),replace=False,p=chance_list)
    pat_list = [data_list_per_set[int(x)] for x in pat_list_idx]
    return pat_list


def load_h5_data(f,channels,start,end,fs=128,num_30s_epochs=35):
    EEG = np.zeros((int((end-start)/num_30s_epochs/30/fs),num_30s_epochs,fs*30,len(channels)))

    for i,chan in enumerate(channels):
        try:
            EEG[:,:,:,i] = np.array(f['channels'][chan][start:end]).reshape((-1,num_30s_epochs,fs*30))
  
        except:
            pass

    return EEG

def load_h5_data_idx(f,channels,idx,fs=128,num_30s_epochs=35):
    EEG = np.zeros((int(len(idx)/num_30s_epochs/30/fs),num_30s_epochs,fs*30,len(channels)))

    for i,chan in enumerate(channels):
        try:
            EEG[:,:,:,i] = np.array(f['channels'][chan][idx]).reshape((-1,num_30s_epochs,fs*30))
  
        except:
            pass

    return EEG
    
def find_start_idx_label(hypno,start_label,fs=128):

    hypno = np.mean(hypno[:len(hypno)//(30*fs)*(30*fs)].reshape(-1,(30*fs)),axis=1)
    loc = np.where(hypno==start_label)[0]
    loc = [n for n in loc if n < len(hypno)-35]
    np.random.shuffle(loc)
    
    try:
        return loc[0]*fs*30
    except:
        loc = np.arange(len(hypno)-36)
        np.random.shuffle(loc)
        return loc[0]*fs*30

def load_channels(all_channels,eeg_eog_list):
    all_eeg = [x for x in all_channels if x in eeg_eog_list['eeg']]
    all_eog = [x for x in all_channels if x in eeg_eog_list['eog']]
    eeg_to_load = np.random.choice(all_eeg,size=1,replace=False)
    eog_to_load = np.random.choice(all_eog,size=1,replace=False)
    return [eeg_to_load[0],eog_to_load[0]]

def load_channels_6(all_channels,eeg_eog_list,eeg_to_load=None,emg_to_load=None,ref=False):
    all_variable_eeg = [x for x in all_channels if x in eeg_eog_list['variable_eeg'][0]]
    if 'C4-M1' in all_variable_eeg:
        if 'C3-M2' in all_variable_eeg:
            all_variable_eeg.remove('C4-M1')

    
        

    all_variable_emg = [x for x in all_channels if x in eeg_eog_list['variable_emg']]
    all_fixed = [x for x in all_channels if x in eeg_eog_list['fixed']]
    if eeg_to_load==None:
        eeg_to_load = np.random.choice(len(all_variable_eeg),size=1,replace=False)[0]
    if emg_to_load==None:   
        emg_to_load = np.random.choice(len(all_variable_emg),size=1,replace=False)[0]

    if ref:
        return [eeg_eog_list['variable_eeg'][0][eeg_to_load]+'-ref',eeg_eog_list['variable_eeg'][1][eeg_to_load]+'-ref',eeg_eog_list['variable_emg'][emg_to_load]]+all_fixed

    else:
        return [eeg_eog_list['variable_eeg'][0][eeg_to_load],eeg_eog_list['variable_eeg'][1][eeg_to_load],eeg_eog_list['variable_emg'][emg_to_load]]+all_fixed

def load_channels_5(all_channels,eeg_eog_list,eeg_to_load=None,emg_to_load=None):
    all_variable_eeg = [x for x in all_channels if x in eeg_eog_list['variable_eeg'][0]]
    all_variable_emg = [x for x in all_channels if x in eeg_eog_list['variable_emg']]
    all_fixed = [x for x in all_channels if x in eeg_eog_list['fixed']]
    if eeg_to_load==None:
        eeg_to_load_tmp = np.random.choice(len(all_variable_eeg),size=1,replace=False)[0]
        eeg_to_load = eeg_eog_list['variable_eeg'][0].index(all_variable_eeg[eeg_to_load_tmp])
    if emg_to_load==None:   
        emg_to_load_tmp = np.random.choice(len(all_variable_emg),size=1,replace=False)[0]
        emg_to_load = eeg_eog_list['variable_emg'][0].index(all_variable_emg[emg_to_load_tmp])
    return [eeg_eog_list['variable_eeg'][0][eeg_to_load],eeg_eog_list['variable_eeg'][1][eeg_to_load],eeg_eog_list['variable_emg'][emg_to_load]]+all_fixed

def load_channels_breathing(all_channels,eeg_eog_list,double_to_load=None,single_to_load=None):
    all_variable_double = [x for x in all_channels if x in eeg_eog_list['variable_double'][0]]
    if len(eeg_eog_list['variable_single'])>0:
        all_variable_single = [x for x in all_channels if x in eeg_eog_list['variable_single'][0]]
    else:
        all_variable_single=[]
    all_fixed = [x for x in all_channels if x in eeg_eog_list['fixed']]
    if double_to_load==None:
        double_to_load = np.random.choice(len(all_variable_double),size=1,replace=False)[0]
    if single_to_load==None:   
        single_to_load = np.random.choice(len(all_variable_double),size=1,replace=False)[0]
    
    channels_to_return=[]
    try:
        channels_to_return.append(eeg_eog_list['variable_double'][0][double_to_load])
        channels_to_return.append(eeg_eog_list['variable_double'][1][double_to_load])
    except:
        pass

    try:
        channels_to_return.append(eeg_eog_list['variable_emg'][single_to_load])
    except:
        pass

    for item in all_fixed:
        channels_to_return.append(item)

    return channels_to_return


def load_patients(self,pat_list,start_label,batch_size=64):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,35,128*30,self.n_channels))
    LABEL = np.zeros((batch_size,30*35*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        pat =pat_list[i]

        try:
            #read hypno
            label = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            start_idx = []
            #find suiting 17,5 minutes of data
            if self.label_type == 'hypnogram':
                start_idx = find_start_idx_label(label,start_label[np.min((i,len(start_label)-1))])
            elif self.label_type == 'arousal' or 'arousal_enhanced' or 'arousal_enhanced_V2' or 'arousal_plus_masked':
                poss_idx = np.zeros(len(label))
                diff_arousal_idx = np.diff(label)
                diff_up = np.where(diff_arousal_idx==1)[0]

                for j in range(len(diff_up)):
                    poss_idx[np.max((diff_up[j]-int(15*self.label_hz),0)):np.min((diff_up[j],len(label)))]=1
                Poss_idx = np.where(poss_idx==1)[0]
                start_idx = np.random.choice(len(Poss_idx),size=1,replace=False)[0]
                start_idx = Poss_idx[start_idx]
            end_idx = start_idx+(35*30*self.label_hz)

            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels(channels,self.chan_type)
                
                #load data
                if len(f['channels'][channels_to_load[0]])/self.data_hz*self.label_hz == len(label):
                    Label = label[start_idx:end_idx]
                    eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))
                    EEG[success,:,:,:] = eeg
                elif end_idx < len(label):
                    Label = label[start_idx:end_idx]
                    eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))
                    EEG[success,:,:,:] = eeg
                LABEL[success,:]= Label#raises error when len label and eeg are not similar

                i+=1
                success+=1
        except:
            i+=1
            
    EEG=EEG[:success,:,:,:]
    LABEL=LABEL[:success,:]

        
    return EEG, LABEL 

# def load_patients_val(self,index,pat_list):
            
#         pat = pat_list[index]
#         try:
#             #read label
#             label = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
#             #find suiting 17,5 minutes of data
#             start_idx = 0
#             end_idx = (len(label)//(30*35*self.label_hz))*(30*35*self.label_hz)
            
#             with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
#                 channels = [x for x in f['channels'].keys()]
#                 channels_to_load = load_channels(channels,self.chan_type)
                
#                 #load data
#                 label = label[start_idx:end_idx]
#                 eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))
#         except:
#             eeg = np.zeros((64,35,30*self.data_hz,2))
#             label = np.zeros((64,35*30*self.data_hz,self.n_classes))

#         return eeg, label


def load_val_patients_arousal_NREM(self,index,pat_list): 

        pat = pat_list[index]
        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
            label = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

            #make sure labels are same length
            if len(label_hypno)!=int((len(label)*self.data_hz/self.label_hz)):
                if len(label_hypno)>int((len(label)*self.data_hz/self.label_hz)):
                    label_hypno = label_hypno[:int(len(label)*self.data_hz/self.label_hz)]
                else:
                    label = label[:len(label_hypno)]

            #find index corresponding to stage(s)
            idx_data = np.where(label_hypno !=4)[0]
            max_len = (len(idx_data)//(30*35*self.data_hz))*(30*35*self.data_hz)
            #if no 17,5 min are present in stage(s) load what we have and fill in extra with constant numbers (value of first index per row)
            if max_len == 0:
                if len(idx_data) < (30*35*self.data_hz):
                    idx_data_ = np.zeros((30*35*self.data_hz))
                    idx_data_[:len(idx_data)] = idx_data
                    idx_data = idx_data_.astype(int)
                    max_len=len(idx_data)

            #select a multiple of 17,5 minutes
            idx_data = idx_data[:max_len]
            #create label_idx based upon data idx
            idx_label = np.unique(np.floor(idx_data/self.data_hz*self.label_hz)).astype(int)

            #if label is shorter that 2100, same trick as above
            if len(idx_label)<2100:
                idx_label_ = np.zeros(2100)
                idx_label_[:len(idx_label)]=idx_label
                idx_label = idx_label_.astype(int)
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_6(channels,self.chan_type)
                
                #load data
                label = label[idx_label].reshape(-1,(30*35*self.label_hz))
                eeg = load_h5_data_idx(f,channels_to_load,idx_data)
        except:
            eeg = np.zeros((1,35,30*self.data_hz,5))
            label = np.zeros((1,35*30*self.data_hz,1))

        return eeg, label

def load_val_patients_arousal_REM(self,index,pat_list): 

        pat = pat_list[index]
        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
            label = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

            #make sure labels are same length
            if len(label_hypno)!=int((len(label)*self.data_hz/self.label_hz)):
                if len(label_hypno)>int((len(label)*self.data_hz/self.label_hz)):
                    label_hypno = label_hypno[:int(len(label)*self.data_hz/self.label_hz)]
                else:
                    label = label[:len(label_hypno)]

            #find index corresponding to stage(s)
            idx_data = np.where(label_hypno ==4)[0]
            max_len = (len(idx_data)//(30*35*self.data_hz))*(30*35*self.data_hz)
            #if no 17,5 min are present in stage(s) load what we have and fill in extra with constant numbers (value of first index per row)
            if max_len == 0:
                if len(idx_data) < (30*35*self.data_hz):
                    idx_data_ = np.zeros((30*35*self.data_hz))
                    idx_data_[:len(idx_data)] = idx_data
                    idx_data = idx_data_.astype(int)
                    max_len=len(idx_data)

            #select a multiple of 17,5 minutes
            idx_data = idx_data[:max_len]
            #create label_idx based upon data idx
            idx_label = np.unique(np.floor(idx_data/self.data_hz*self.label_hz)).astype(int)

            #if label is shorter that 2100, same trick as above
            if len(idx_label)<2100:
                idx_label_ = np.zeros(2100)
                idx_label_[:len(idx_label)]=idx_label
                idx_label = idx_label_.astype(int)
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_6(channels,self.chan_type)
                
                #load data
                label = label[idx_label].reshape(-1,(30*35*self.label_hz))
                eeg = load_h5_data_idx(f,channels_to_load,idx_data)
        except:
            eeg = np.zeros((1,35,30*self.data_hz,5))
            label = np.zeros((1,35*30*self.data_hz,1))

        return eeg, label

def load_patients_arousal(self,pat_list,start_label,batch_size=32):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,35,128*30,self.n_channels))
    LABEL_hypno = np.zeros((batch_size,int(30*35*self.hypno_hz)))
    LABEL_arousal = np.zeros((batch_size,30*35*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]

            
            try:
                #read hypno
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(35*30*self.data_hz)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_5(channels,self.chan_type)
                    
                    #load data
                    len_data = len(f['channels'][channels_to_load[0]])
                    eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx))

                    label_hypno_ = np.mean(label_hypno[int(start_idx):int(end_idx)].reshape(35,-1),axis=1)
                    
                    if len(np.unique(label_hypno_))>6:
                        a=b
                    label_hypno = label_hypno_
                    label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]
                    
                    EEG[success,:,:,:] = eeg
                    LABEL_hypno[success,:]= label_hypno
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #read hypno
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                diff_arousal_idx = np.diff(label_arousal)
                diff_up = np.where(diff_arousal_idx==1)[0]

                #choose random segment with arousal
                Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                start_idx = np.min((diff_up[int(Poss_idx1)],int(len(label_arousal)-(17.5*60*self.label_hz))))
                start_idx = np.max((start_idx,0))
                #match with hypno
                diff_hypno_idx = np.diff(label_hypno)
                diff = np.where(diff_hypno_idx)[0]
                start_idx_idx = np.argmin(abs(diff-start_idx))
                start_idx = diff[start_idx_idx]+1
                end_idx = start_idx+(35*30*self.label_hz)

                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                        
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_5(channels,self.chan_type)
                        
                        #load data
                        len_data = len(f['channels'][channels_to_load[0]])
                        eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))

                        label_hypno_ = np.mean(label_hypno[int(start_idx/self.label_hz*self.data_hz):int(end_idx/self.label_hz*self.data_hz)].reshape(35,-1),axis=1)
                        label_hypno = label_hypno_
                        label_arousal = label_arousal[int(start_idx):int(end_idx)]
                        
                        EEG[success,:,:,:] = eeg
                        LABEL_hypno[success,:]= label_hypno
                        LABEL_arousal[success,:]= label_arousal

                        i+=1
                        success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1


                
    EEG=EEG[:success,:,:,:]
    LABEL_hypno=LABEL_hypno[:success,:]
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, [LABEL_hypno,LABEL_arousal]

def load_patients_arousal_shallow(self,pat_list,start_label,batch_size=32,num_30s_seg=35,ref=False):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]

            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                except:
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    except:
                        hypno_name = self.label_type.replace('arousal','hypnogram')
                        hypno_name_old = hypno_name
                        if ('stanford' in pat) & ('_3' in hypno_name):
                            hypno_name = 'hypnogram_rater_0'
                        label_hypno = read_label(opj(pat,hypno_name+'.ids'),fs=self.data_hz)
                
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                if ('stanford_t' in pat):
                    if ('_3' not in hypno_name_old):
                        label_arousal[label_arousal<2]=0
                        label_arousal[label_arousal>1]=1
                    

                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*self.data_hz)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    
                    if 'mesa' in pat:
                        channels_to_load = ['CZ-OZ','C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mros' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'shhs' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mgh' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    if 'robert' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    if 'stanford' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    
                    #load data
                    eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)

                    label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    
                    EEG[success,:,:,:] = eeg

                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    except:
                        try:
                            label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                        except:
                            hypno_name = self.label_type.replace('arousal','hypnogram')
                            hypno_name_old = hypno_name
                            if ('stanford' in pat) & ('_3' in hypno_name):
                                hypno_name = 'hypnogram_rater_0'
                            label_hypno = read_label(opj(pat,hypno_name+'.ids'),fs=self.data_hz)
                    
                    label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                    
                    if ('stanford_t' in pat):
                        if ('_3' not in hypno_name_old):
                            label_arousal[label_arousal<2]=0
                            label_arousal[label_arousal>1]=1
                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    #find suiting 17,5 minutes of data
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['C4-M1'].shape[0]/self.data_hz*self.label_hz-(num_30s_seg*40*self.label_hz))))
                    start_idx = np.max((start_idx,0))
                    # #match with hypno
                    # diff_hypno_idx = np.diff(label_hypno)
                    # diff = int(np.where(diff_hypno_idx)[0][0]/self.data_hz*self.label_hz)//60
                    # if diff>0:
                    #     start_idx = start_idx-60+diff
                    end_idx = start_idx+(num_30s_seg*30*self.label_hz)

                
                    
                    if 'mesa' in pat:
                        channels_to_load = ['CZ-OZ','C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mros' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'shhs' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mgh' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    if 'robert' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    if 'stanford' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    
                    
                    #load data
                    if int(end_idx/self.label_hz*self.data_hz)-int(start_idx/self.label_hz*self.data_hz) == 134400:
                        eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz),num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')

                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    EEG[success,:,:,:] = np.squeeze(eeg)
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1

            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1

                
    EEG=EEG[:success,:,:,:]
    if self.scaled ==False:
        #eeg
        EEG[:,:,:,[0,1,3]] = EEG[:,:,:,[0,1,3]]*1000000
        EEG[:,:,:,[2]] = EEG[:,:,:,[2]]*1000000
        EEG[:,:,:,[4]] = EEG[:,:,:,[4]]*100000
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, LABEL_arousal


def load_patients_arousal_shallow_yield(hparams,pat_list,start_label,batch_size=32,num_30s_seg=35):
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,hparams['build']['n_channels']))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(hparams['data']['label_Hz'])))

    #loop though patients
    i = 0
    success = 0
    while success<hparams['fit']['batch_size']:
        
        while success< hparams['fit']['load_hypno']:
            pat =pat_list[i]

            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=hparams['data']['data_Hz'])
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=hparams['data']['data_Hz'])
                label_arousal = read_label(opj(pat,hparams['data']['label_type']+'.ids'),fs=hparams['data']['label_Hz'])
                
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*hparams['data']['data_Hz'])

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,hparams['channels'])
                    
                    #load data
                    eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)

                    label_arousal = label_arousal[int(start_idx/hparams['data']['data_Hz']*hparams['data']['label_Hz']):int(end_idx/hparams['data']['data_Hz']*hparams['data']['label_Hz'])]

                    
                    EEG[success,:,:,:] = eeg

                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= hparams['fit']['load_hypno'] and success <hparams['fit']['batch_size']:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=hparams['data']['data_Hz'])
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=hparams['data']['data_Hz'])
                    label_arousal = read_label(opj(pat,hparams['data']['label_type']+'.ids'),fs=hparams['data']['label_Hz'])

                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    #find suiting 17,5 minutes of data
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/hparams['data']['data_Hz']*hparams['data']['label_Hz']-(num_30s_seg*40*hparams['data']['label_Hz']))))
                    start_idx = np.max((start_idx,0))

                    end_idx = start_idx+(num_30s_seg*30*hparams['data']['label_Hz'])

                
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,hparams['channels'])
                    
                    #load data
                    if int(end_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz'])-int(start_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']) == 134400:
                        eeg = load_h5_data(f,channels_to_load,int(start_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']),int(end_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']),num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')

                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    

                    EEG[success,:,:,:] = np.squeeze(eeg)
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1

            except:

                i+=1

                
    EEG=EEG[:success,:,:,:]
    if hparams['data']['scaled']==False:
        #eeg
        EEG[:,:,:,[0,1,3]] = EEG[:,:,:,[0,1,3]]*1000000
        EEG[:,:,:,[2]] = EEG[:,:,:,[2]]*1000000
        EEG[:,:,:,[4]] = EEG[:,:,:,[4]]*100000
    LABEL_arousal=LABEL_arousal[:success,:]


    LABEL_arousal = keras.utils.all_utils.to_categorical(LABEL_arousal, num_classes=2)

    return [EEG, LABEL_arousal]
            
  

def load_patients_arousal_hypno_yield(hparams,i,pat,start_label,num_30s_seg=35):
    #pre-allocate data

    try:
        #read hypno
        try:
            label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=hparams['data']['data_Hz'])
        except:
            label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=hparams['data']['data_Hz'])
        label_arousal = read_label(opj(pat,hparams['data']['label_type']+'.ids'),fs=hparams['data']['label_Hz'])
        
        #if no arousals skip
        if max(label_arousal)==0:
            a=b

        #find suiting 17,5 minutes of data
        start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
        end_idx = start_idx+(num_30s_seg*30*hparams['data']['data_Hz'])

        with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
            
            channels = [x for x in f['channels'].keys()]
            channels_to_load = load_channels_6(channels,hparams['channels'])
            
            #load data
            eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)

            label_arousal = label_arousal[int(start_idx/hparams['data']['data_Hz']*hparams['data']['label_Hz']):int(end_idx/hparams['data']['data_Hz']*hparams['data']['label_Hz'])]
            label_arousal = np.expand_dims( keras.utils.all_utils.to_categorical(label_arousal, num_classes=2),axis=0)
    except:
        
        eeg = np.zeros((1,3840,35,5))
        label_arousal = np.zeros((1,2100,2))

    return [eeg, label_arousal]
       

def load_patients_arousal_arousal_yield(hparams,pat,num_30s_seg=35):
    pat 
    try:
        #load data
        with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
            

            label_arousal = read_label(opj(pat,hparams['data']['label_type']+'.ids'),fs=hparams['data']['label_Hz'])

            #if no arousals skip
            if max(label_arousal)==0:
                a=b

            #find suiting 17,5 minutes of data based upon arousals
            diff_arousal_idx = np.diff(label_arousal)
            diff_up = np.where(diff_arousal_idx==1)[0]

            #choose random segment with arousal
            Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
            start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/hparams['data']['data_Hz']*hparams['data']['label_Hz']-(num_30s_seg*40*hparams['data']['label_Hz']))))
            start_idx = np.max((start_idx,0))
            end_idx = start_idx+(num_30s_seg*30*hparams['data']['label_Hz'])

            channels = [x for x in f['channels'].keys()]
            channels_to_load = load_channels_6(channels,hparams['channels'])
            
            #load data
            if int(end_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz'])-int(start_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']) == 134400:
                eeg = load_h5_data(f,channels_to_load,int(start_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']),int(end_idx/hparams['data']['label_Hz']*hparams['data']['data_Hz']),num_30s_epochs=num_30s_seg)
            else:
                print('input length is incorrect for load_select_functions_for_loader.py line:360')

            label_arousal = label_arousal[int(start_idx):int(end_idx)]
            label_arousal = np.expand_dims( keras.utils.all_utils.to_categorical(label_arousal, num_classes=2),axis=0)
            eeg = np.squeeze(eeg)

    except:

        eeg = np.zeros((1,35,3840,5))
        label_arousal = np.zeros((1,2100,2))

    return [eeg, label_arousal]


def load_patients_arousal_shallow_aux(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_hypno = np.zeros((batch_size,int(30*num_30s_seg*self.hypno_hz)))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]

            
            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*self.data_hz)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)
                    label_hypno_ = np.mean(label_hypno[int(start_idx):int(end_idx)].reshape(num_30s_seg,-1),axis=1)
                    if len(np.unique(label_hypno_))>6:
                        a=b
                    label_hypno = label_hypno_
                    label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    
                    EEG[success,:,:,:] = eeg
                    LABEL_hypno[success,:]= label_hypno
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    #find suiting 17,5 minutes of data
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/self.data_hz*self.label_hz-(num_30s_seg*40*self.label_hz))))
                    start_idx = np.max((start_idx,0))
                    # #match with hypno
                    # diff_hypno_idx = np.diff(label_hypno)
                    # diff = int(np.where(diff_hypno_idx)[0][0]/self.data_hz*self.label_hz)//60
                    # if diff>0:
                    #     start_idx = start_idx-60+diff
                    end_idx = start_idx+(num_30s_seg*30*self.label_hz)

                
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    if int(end_idx/self.label_hz*self.data_hz)-int(start_idx/self.label_hz*self.data_hz) == 134400:
                        eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz),num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')
                    label_hypno_ = np.mean(label_hypno[int(start_idx/self.label_hz*self.data_hz):int(end_idx/self.label_hz*self.data_hz)].reshape(num_30s_seg,-1),axis=1)
                    label_hypno = label_hypno_
                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    EEG[success,:,:,:] = np.squeeze(eeg)
                    LABEL_hypno[success,:]= label_hypno
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1

            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1

                
    EEG=EEG[:success,:,:,:]
    if self.scaled ==False:
        #eeg
        EEG[:,:,:,[0,1,3]] = EEG[:,:,:,[0,1,3]]*1000000
        EEG[:,:,:,[2]] = EEG[:,:,:,[2]]*1000000
        EEG[:,:,:,[4]] = EEG[:,:,:,[4]]*100000
    LABEL_hypno=LABEL_hypno[:success,:]
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, [LABEL_hypno,LABEL_arousal]

def load_patients_arousal_shallow_hypno_plus(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_hypno_plus = np.zeros((batch_size,int(30*num_30s_seg*self.label_hz)))


    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]

            
            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*self.data_hz)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.label_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.label_hz)

                    label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]
                    label_hypno = label_hypno[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]
                    label_hypno[label_arousal==1]=6

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    
                    EEG[success,:,:,:] = eeg
                    LABEL_hypno_plus[success,:]= label_hypno

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    #find suiting 17,5 minutes of data
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/self.data_hz*self.label_hz-(num_30s_seg*40*self.label_hz))))
                    start_idx = np.max((start_idx,0))
                    # #match with hypno
                    # diff_hypno_idx = np.diff(label_hypno)
                    # diff = int(np.where(diff_hypno_idx)[0][0]/self.data_hz*self.label_hz)//60
                    # if diff>0:
                    #     start_idx = start_idx-60+diff
                    end_idx = start_idx+(num_30s_seg*30*self.label_hz)

                
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    if int(end_idx/self.label_hz*self.data_hz)-int(start_idx/self.label_hz*self.data_hz) == 134400:
                        eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz),num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.label_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.label_hz)

                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    label_hypno = label_hypno[int(start_idx):int(end_idx)]
                    label_hypno[label_arousal==1]=6
                    

                    #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    EEG[success,:,:,:] = np.squeeze(eeg)
                    if self.scaled ==False:
                        #eeg
                        EEG[:,:,:,[0,1,3]] = EEG[:,:,:,[0,1,3]]*1000000
                        EEG[:,:,:,[2]] = EEG[:,:,:,[2]]*1000000
                        EEG[:,:,:,[4]] = EEG[:,:,:,[4]]*100000
                    LABEL_hypno_plus[success,:]= label_hypno


                    i+=1
                    success+=1

            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1


                
    EEG=EEG[:success,:,:,:]
    LABEL_hypno_plus=LABEL_hypno_plus[:success,:]


        
    return EEG, LABEL_hypno_plus


def load_patients_val_arousal_hypno_plus(self,index,pat_list,num_30s_seg=35):
            
        pat = pat_list[index]
        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
            label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
            max_len = np.min((len(label_hypno)/self.data_hz,len(label_arousal)/self.label_hz))//(num_30s_seg*30)*(num_30s_seg*30)
            start_idx = 0
            max_hypno = int(max_len*self.data_hz)
            max_arousal = int(max_len*self.label_hz)
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_6(channels,self.chan_type)
                
                #load data
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.label_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.label_hz)
                label_arousal = label_arousal[start_idx:max_arousal]
                label_hypno = label_hypno[start_idx:max_arousal]
                label_hypno[label_arousal==1]=6
                label_hypno = label_hypno.reshape(-1,num_30s_seg*30*self.label_hz)
                eeg = load_h5_data(f,channels_to_load,0,max_hypno,num_30s_epochs=num_30s_seg)
                
        except:
            eeg = np.zeros((1,35,3840,5))
            label_hypno = np.zeros((1,35))

        # #test 
        # for xxxx in range(eeg.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,eeg[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')

        return eeg, label_hypno


def load_patients_arousal_NREM(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]
            
            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #make sure labels are same length
                if len(label_hypno)!=int((len(label_arousal)*self.data_hz/self.label_hz)):
                    if len(label_hypno)>int((len(label_arousal)*self.data_hz/self.label_hz)):
                        label_hypno = label_hypno[:int(len(label_arousal)*self.data_hz/self.label_hz)]
                    else:
                        label_arousal = label_arousal[:len(label_hypno)]

                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                NREM_loc = np.where(label_hypno!=4)[0]
                if len(NREM_loc)< (30*35*self.data_hz):
                    i+=1
                    print('skiped due to insufficient data')
                    continue
                
                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno[NREM_loc],start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*self.data_hz)
                idx_data = NREM_loc[start_idx:end_idx]
                #create label_idx based upon data idx
                idx_label = np.unique(np.floor(idx_data/self.data_hz*self.label_hz)).astype(int)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    eeg = load_h5_data_idx(f,channels_to_load,idx_data,num_30s_epochs=num_30s_seg)
                    label_hypno = np.mean(label_hypno[idx_data].reshape(num_30s_seg,-1),axis=1)
                    if len(np.unique(label_hypno))>6:
                        a=b
                    label_arousal = label_arousal[idx_label]
                    


                    EEG[success,:,:,:] = eeg
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

                    #make sure labels are same length
                    if len(label_hypno)!=int((len(label_arousal)*self.data_hz/self.label_hz)):
                        if len(label_hypno)>int((len(label_arousal)*self.data_hz/self.label_hz)):
                            label_hypno = label_hypno[:int(len(label_arousal)*self.data_hz/self.label_hz)]
                        else:
                            label_arousal = label_arousal[:len(label_hypno)]
                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    NREM_loc = np.where(label_hypno!=4)[0]
                    if len(NREM_loc)< (30*35*self.data_hz):
                        i+=1
                        print('skiped due to insufficient data')
                        continue
                    #create label_idx based upon onrem IDX
                    idx_label = np.unique(np.floor(NREM_loc/self.data_hz*self.label_hz)).astype(int)
                    label_arousal[idx_label]

                    #find arousals
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/self.data_hz*self.label_hz-(num_30s_seg*40*self.label_hz))))
                    start_idx = np.max((start_idx,0))
                    end_idx = start_idx+(num_30s_seg*30*self.label_hz)

                    #convert to hypno
                    start_idx_data= int(start_idx*self.data_hz/self.label_hz)
                    end_idx_data = start_idx_data+(num_30s_seg*30*self.data_hz)
                    idx_data = NREM_loc[start_idx_data:end_idx_data]
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    if int(end_idx/self.label_hz*self.data_hz)-int(start_idx/self.label_hz*self.data_hz) == 134400:
                        eeg = load_h5_data_idx(f,channels_to_load,idx_data,num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')

                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    
                    # #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')

                    EEG[success,:,:,:] = np.squeeze(eeg)
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1

            except:
                i+=1


                
    EEG=EEG[:success,:,:,:]
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, LABEL_arousal

def load_patients_arousal_REM(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
        while success< self.load_hypno:
            pat =pat_list[i]
            
            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #make sure labels are same length
                if len(label_hypno)!=int((len(label_arousal)*self.data_hz/self.label_hz)):
                    if len(label_hypno)>int((len(label_arousal)*self.data_hz/self.label_hz)):
                        label_hypno = label_hypno[:int(len(label_arousal)*self.data_hz/self.label_hz)]
                    else:
                        label_arousal = label_arousal[:len(label_hypno)]

                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                REM_loc = np.where(label_hypno==4)[0]
                if len(REM_loc)< (30*35*self.data_hz):
                    i+=1
                    print('skiped due to insufficient data')
                    continue
                
                #find suiting 17,5 minutes of data
                start_idx = find_start_idx_label(label_hypno[REM_loc],start_label[np.min((i,len(start_label)-1))])
                end_idx = start_idx+(num_30s_seg*30*self.data_hz)
                idx_data = REM_loc[start_idx:end_idx]
                #create label_idx based upon data idx
                idx_label = np.unique(np.floor(idx_data/self.data_hz*self.label_hz)).astype(int)

                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    eeg = load_h5_data_idx(f,channels_to_load,idx_data,num_30s_epochs=num_30s_seg)
                    label_hypno = np.mean(label_hypno[idx_data].reshape(num_30s_seg,-1),axis=1)
                    if len(np.unique(label_hypno))>6:
                        a=b
                    label_arousal = label_arousal[idx_label]
                    


                    EEG[success,:,:,:] = eeg
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        while success>= self.load_hypno and success <self.batch_size:
            pat =pat_list[i]
            try:
                #load data
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    #read hypno
                    try:
                        label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    except:
                        label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

                    #make sure labels are same length
                    if len(label_hypno)!=int((len(label_arousal)*self.data_hz/self.label_hz)):
                        if len(label_hypno)>int((len(label_arousal)*self.data_hz/self.label_hz)):
                            label_hypno = label_hypno[:int(len(label_arousal)*self.data_hz/self.label_hz)]
                        else:
                            label_arousal = label_arousal[:len(label_hypno)]
                    #if no arousals skip
                    if max(label_arousal)==0:
                        i+=1
                        continue

                    REM_loc = np.where(label_hypno==4)[0]
                    if len(REM_loc)< (30*35*self.data_hz):
                        i+=1
                        print('skiped due to insufficient data')
                        continue
                    #create label_idx based upon onrem IDX
                    idx_label = np.unique(np.floor(REM_loc/self.data_hz*self.label_hz)).astype(int)
                    label_arousal[idx_label]

                    #find arousals
                    diff_arousal_idx = np.diff(label_arousal)
                    diff_up = np.where(diff_arousal_idx==1)[0]

                    #choose random segment with arousal
                    Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
                    start_idx = np.min((diff_up[int(Poss_idx1)],int(f['channels']['F3-M2'].shape[0]/self.data_hz*self.label_hz-(num_30s_seg*40*self.label_hz))))
                    start_idx = np.max((start_idx,0))
                    end_idx = start_idx+(num_30s_seg*30*self.label_hz)

                    #convert to hypno
                    start_idx_data= int(start_idx*self.data_hz/self.label_hz)
                    end_idx_data = start_idx_data+(num_30s_seg*30*self.data_hz)
                    idx_data = REM_loc[start_idx_data:end_idx_data]
                    
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type)
                    
                    #load data
                    if int(end_idx/self.label_hz*self.data_hz)-int(start_idx/self.label_hz*self.data_hz) == 134400:
                        eeg = load_h5_data_idx(f,channels_to_load,idx_data,num_30s_epochs=num_30s_seg)
                    else:
                        print('input length is incorrect for load_select_functions_for_loader.py line:360')

                    label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    
                    # #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')

                    EEG[success,:,:,:] = np.squeeze(eeg)
                    LABEL_arousal[success,:]= label_arousal

                    i+=1
                    success+=1

            except:
                i+=1


                
    EEG=EEG[:success,:,:,:]
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, LABEL_arousal

# def load_patients_arousal_idx(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
#     #pre-allocate data
#     EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
#     LABEL_hypno = np.zeros((batch_size,int(30*num_30s_seg*self.hypno_hz)))
#     LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

#     #loop though patients
#     i = 0
#     success = 0
#     while success<self.batch_size:
        
#         pat =pat_list[i]
        
#         try:
#             #read hypno
#             label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#             label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
#             #if no arousals skip
#             if max(label_arousal)==0:
#                 i+=1
#                 continue

#             #find suiting 17,5 minutes of data
#             with open(opj(pat,'arousal_start.idx')) as f:
#                 start_idx = int(f.readlines()[0])
#             end_idx = start_idx+(4*num_30s_seg*30*self.data_hz)

#             with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                
#                 channels = [x for x in f['channels'].keys()]
#                 channels_to_load = load_channels_6(channels,self.chan_type)
                
#                 #load data
#                 eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)
#                 label_hypno_ = np.mean(label_hypno[int(start_idx):int(end_idx)].reshape(4,num_30s_seg,-1),axis=2)
#                 if len(np.unique(label_hypno_))>6:
#                     a=b
#                 label_hypno = label_hypno_
#                 label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)].reshape(4,-1)
                
#                 EEG[success:success+4,:,:,:] = eeg
#                 LABEL_hypno[success:success+4,:]= label_hypno
#                 LABEL_arousal[success:success+4,:]= label_arousal

#                 i+=1
#                 success+=4
#         except:
#             #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
#             i+=1
        

                
#     EEG=EEG[:success,:,:,:]
#     LABEL_hypno=LABEL_hypno[:success,:]
#     LABEL_arousal=LABEL_arousal[:success,:]

        
#     return EEG, [LABEL_hypno,LABEL_arousal]



# def load_patients_arousal_shallow_psd(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
#     #pre-allocate data
#     EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
#     LABEL_hypno = np.zeros((batch_size,int(30*num_30s_seg*self.hypno_hz)))
#     LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))
#     LABEL_PSD = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz),1,4))

#     #loop though patients
#     i = 0
#     success = 0
#     while success<self.batch_size:
        
#         while success< self.load_hypno:
#             pat =pat_list[i]

            
#             try:
#                 #read hypno
#                 label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#                 label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                    
#                 #if no arousals skip
#                 if max(label_arousal)==0:
#                     i+=1
#                     continue

#                 #find suiting 17,5 minutes of data
#                 start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
#                 end_idx = start_idx+(num_30s_seg*30*self.data_hz)

#                 with h5.File(glob(opj(pat,'*_psd.h5'))[0],'r') as f:
                    
#                     times = np.array(f['PSD']['freq']) #incorrectly saved!! so freq == times
#                     start_idx_psd = np.where(np.abs(times - start_idx/self.data_hz) == np.min(np.abs(times - start_idx/self.data_hz)))[0][0]#time in sec
#                     end_idx_psd = np.where(np.abs(times - end_idx/self.data_hz) == np.min(np.abs(times - end_idx/self.data_hz)))[0][0]#time in sec
#                     PSD_ = np.array(f['PSD']['psd'][:,start_idx_psd:end_idx_psd])
#                     time_start = times[start_idx_psd] 
#                     time_end = times[end_idx_psd] 

#                     freq = np.array(f['PSD']['times'])#incorrectly saved!! so times == freq
#                     Alpha_range = np.arange(128,192)
#                     Beta_range = np.arange(192,319)
#                     Theta_range = np.arange(64,128)
#                     Delta_range = np.arange(8,64)
#                     PSD = np.zeros((4,self.label_shape_b2[1]))
#                     len_psd = PSD_.shape[1]
#                     space = np.linspace(0,len_psd,2101)
#                     for k in np.arange(2100):
#                         PSD[0,k] = np.mean(np.mean(PSD_[Delta_range,int(space[k]):int(space[k+1])],axis=1))
#                         PSD[1,k] = np.mean(np.mean(PSD_[Theta_range,int(space[k]):int(space[k+1])],axis=1))
#                         PSD[2,k] = np.mean(np.mean(PSD_[Alpha_range,int(space[k]):int(space[k+1])],axis=1))
#                         PSD[3,k] = np.mean(np.mean(PSD_[Beta_range,int(space[k]):int(space[k+1])],axis=1))



#                 with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
#                     channels = [x for x in f['channels'].keys()]
#                     channels_to_load = load_channels_6(channels,self.chan_type)
                    
#                     #load data
#                     eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx),num_30s_epochs=num_30s_seg)

#                     # EEG_straight = eeg.reshape(-1,5)
#                     # f, (ax1, ax2) = plt.subplots(2, 1)
#                     # for i in range(5):
#                     #     ax1.plot(EEG_straight[60*128:120*128,i]-i*10,color='black',linewidth=0.5)
#                     # ax2.imshow(PSD[:,120:240],cmap='jet',aspect='auto',origin='lower')
#                     # plt.show()
#                     label_hypno_ = np.mean(label_hypno[int(start_idx):int(end_idx)].reshape(num_30s_seg,-1),axis=1)
                    
#                     if len(np.unique(label_hypno_))>6:
#                         a=b
#                     label_hypno = label_hypno_
#                     label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]
                    
#                     EEG[success,:,:,:] = eeg
#                     LABEL_hypno[success,:]= label_hypno
#                     LABEL_arousal[success,:]= label_arousal
#                     LABEL_PSD[success,:,0,:]= PSD.T

#                     i+=1
#                     success+=1
#             except:
#                 #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
#                 i+=1
#         # while success>= self.load_hypno and success <self.batch_size:
#         #     pat =pat_list[i]
#         #     try:
#         #         #read hypno
#         #         label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
#         #         #if no arousals skip
#         #         if max(label_arousal)==0:
#         #             i+=1
#         #             continue

#         #         label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)

#         #         #find suiting 17,5 minutes of data
#         #         diff_arousal_idx = np.diff(label_arousal)
#         #         diff_up = np.where(diff_arousal_idx==1)[0]

#         #         #choose random segment with arousal
#         #         Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
#         #         start_idx = np.min((diff_up[int(Poss_idx1)],int(len(label_arousal)-(num_30s_seg*30*self.label_hz))))
#         #         start_idx = np.max((start_idx,0))
                
#         #         #match with hypno
#         #         diff_hypno_idx = np.diff(label_hypno)
#         #         diff = np.where(diff_hypno_idx)[0]/self.data_hz*self.label_hz
#         #         start_idx_idx = np.argmin(abs(diff-start_idx))
#         #         start_idx = diff[start_idx_idx]+1
#         #         start_idx = np.min((start_idx,int(len(label_arousal)-(num_30s_seg*30*self.label_hz))))
#         #         end_idx = start_idx+(num_30s_seg*30*self.label_hz)

#         #         #load data
#         #         with h5.File(glob(opj(pat,'*_psd.h5'))[0],'r') as f:
                    
#         #             times = np.array(f['PSD']['freq']) #incorrectly saved!! so freq == times
#         #             start_idx_psd = np.where(np.abs(times - start_idx/self.label_hz) == np.min(np.abs(times - start_idx/self.label_hz)))[0][0]#time in sec
#         #             end_idx_psd = np.where(np.abs(times - end_idx/self.label_hz) == np.min(np.abs(times - end_idx/self.label_hz)))[0][0]#time in sec
#         #             PSD_ = np.array(f['PSD']['psd'][:,start_idx_psd:end_idx_psd])


#         #             freq = np.array(f['PSD']['times'])#incorrectly saved!! so times == freq
#         #             Alpha_range = np.arange(128,192)
#         #             Beta_range = np.arange(192,319)
#         #             Theta_range = np.arange(64,128)
#         #             Delta_range = np.arange(8,64)
#         #             PSD = np.zeros((4,self.label_shape_b2[1]))
#         #             len_psd = PSD_.shape[1]
#         #             space = np.linspace(0,len_psd,2101)
#         #             for k in np.arange(2100):
#         #                 PSD[0,k] = np.mean(np.mean(PSD_[Delta_range,int(space[k]):int(space[k+1])],axis=1))
#         #                 PSD[1,k] = np.mean(np.mean(PSD_[Theta_range,int(space[k]):int(space[k+1])],axis=1))
#         #                 PSD[2,k] = np.mean(np.mean(PSD_[Alpha_range,int(space[k]):int(space[k+1])],axis=1))
#         #                 PSD[3,k] = np.mean(np.mean(PSD_[Beta_range,int(space[k]):int(space[k+1])],axis=1))

#         #         with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                        
#         #             channels = [x for x in f['channels'].keys()]
#         #             channels_to_load = load_channels_6(channels,self.chan_type)
                    
#         #             #load data
#         #             len_data = len(f['channels'][channels_to_load[0]])
#         #             eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz),num_30s_epochs=num_30s_seg)

#         #             label_hypno_ = np.mean(label_hypno[int(start_idx/self.label_hz*self.data_hz):int(end_idx/self.label_hz*self.data_hz)].reshape(num_30s_seg,-1),axis=1)
#         #             label_hypno = label_hypno_
#         #             label_arousal = label_arousal[int(start_idx):int(end_idx)]
                    
#         #             EEG[success,:,:,:] = eeg
#         #             LABEL_hypno[success,:]= label_hypno
#         #             LABEL_arousal[success,:]= label_arousal
#         #             LABEL_PSD[success,:,0,:]= PSD

#         #             i+=1
#         #             success+=1
#         #     except:
#         #             #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
#         #             i+=1


                
#     EEG=EEG[:success,:,:,:]
#     LABEL_hypno=LABEL_hypno[:success,:]
#     LABEL_arousal=LABEL_arousal[:success,:]
#     LABEL_psd = LABEL_PSD[:success,:,:,:]
        
#     return [EEG,LABEL_psd], [LABEL_hypno,LABEL_arousal]


def load_patients_breathing(self,pat_list,start_label,batch_size=32,num_30s_seg=35):
    
    #pre-allocate data [breathing abdominal chest saturation 2 eeg ]]
    EEG = np.zeros((self.dim))
    LABEL_resp = np.zeros((self.label_shape_b1))
    LABEL_arousal = np.zeros((self.label_shape_b2))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        
       
        pat =pat_list[i]

        
        try:
            #read hypno
            label_b1 = read_label(opj(pat,'resp.ids'),fs=int(self.data_hz/self.data_per_prediction_b1))
            label_b2 = read_label(opj(pat,'arousal.ids'),fs=int(self.data_hz/self.data_per_prediction_b2))
    
            #find suiting 17,5 minutes of data
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_breathing(channels,self.chan_type)
                
                #load data
                start = int(random.randrange(0,int(len(label_b1)/int(self.data_hz/self.data_per_prediction_b1)/60)-75,1)*60*self.data_hz)
                len_data = (self.data_hz*30*35)*4
                einde = start+len_data

                start_b1 = int(start/self.data_per_prediction_b1)
                end_b1 = int(einde/self.data_per_prediction_b1)

                start_b2 = int(start/self.data_per_prediction_b2)
                end_b2 = int(einde/self.data_per_prediction_b2)

                eeg = load_h5_data(f,channels_to_load,start,einde,num_30s_epochs=35)
                if len(np.where(np.isnan(eeg))[0])>0:
                    a=b
                
                label_b1 = label_b1[start_b1:end_b1].reshape(eeg.shape[0],-1)
                label_b2 = label_b2[start_b2:end_b2].reshape(eeg.shape[0],-1)
                
                EEG[success:success+4,:,:,:] = eeg
                LABEL_resp[success:success+4,:]= label_b1
                LABEL_arousal[success:success+4,:]= label_b2

                i+=1
                success+=4
        except:
            #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
            i+=1
        


                
    EEG=EEG[:success,:,:,:]
    LABEL_resp=LABEL_resp[:success,:]
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, [LABEL_resp,LABEL_arousal]

def load_patients_val_breathing(self,index,pat_list,num_30s_seg=35):
    
    #loop though patients
    pat =pat_list[index]
    
    try:
        #read hypno
        label_b1 = read_label(opj(pat,'resp.ids'),fs=int(self.data_hz/self.data_per_prediction_b1))
        label_b2 = read_label(opj(pat,'arousal.ids'),fs=int(self.data_hz/self.data_per_prediction_b2))
        
        #find suiting 17,5 minutes of data

        with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
            
            channels = [x for x in f['channels'].keys()]
            channels_to_load = load_channels_breathing(channels,self.chan_type)
            
            #load data
            start = 0
            len_data = (len(f['channels'][channels[0]])//(30*35*self.data_hz))*(30*35*self.data_hz)
            einde = start+len_data

            start_b1 = int(start/self.data_per_prediction_b1)
            end_b1 = int(einde/self.data_per_prediction_b1)

            start_b2 = int(start/self.data_per_prediction_b2)
            end_b2 = int(einde/self.data_per_prediction_b2)

            eeg = load_h5_data(f,channels_to_load,start,einde,num_30s_epochs=35)
            if len(np.where(np.isnan(eeg))[0])>0:
                a=b
            
            label_b1 = label_b1[start_b1:end_b1].reshape(eeg.shape[0],-1)
            label_b2 = label_b2[start_b2:end_b2].reshape(eeg.shape[0],-1)
    except:
        pass       
                
        

        
    return eeg, [label_b1,label_b2]


# def load_patients_val(self,index,pat_list):
            
#         pat = pat_list[index]
#         try:
#             #read label
#             label = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
#             #find suiting 17,5 minutes of data
#             start_idx = 0
#             end_idx = (len(label)//(30*35*self.label_hz))*(30*35*self.label_hz)
            
#             with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
#                 channels = [x for x in f['channels'].keys()]
#                 channels_to_load = load_channels(channels,self.chan_type)
                
#                 #load data
#                 label = label[start_idx:end_idx]
#                 label
#                 eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))
#         except:
#             eeg = np.zeros((64,35,30*self.data_hz,2))
#             label = np.zeros((64,35*30*self.data_hz,self.n_classes))

#         return eeg, label

def load_patients_val_arousal(self,index,pat_list,num_30s_seg=35,ref=False):
            
        pat = pat_list[index]
        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
            except:
                try:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                except:
                    hypno_name = self.label_type.replace('arousal','hypnogram')
                    hypno_name_old = hypno_name
                    if ('stanford' in pat) & ('_3' in hypno_name):
                        hypno_name = 'hypnogram_rater_0'
                    label_hypno = read_label(opj(pat,hypno_name+'.ids'),fs=self.data_hz)
            
            label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

            if ('stanford_t' in pat):
                if ('_3' not in hypno_name_old):
                    label_arousal[label_arousal<2]=0
                    label_arousal[label_arousal>1]=1
                
            
            max_len = np.min((len(label_hypno)/self.data_hz,len(label_arousal)/self.label_hz))//(num_30s_seg*30)*(num_30s_seg*30)
            start_idx = 0
            max_hypno = int(max_len*self.data_hz)
            max_arousal = int(max_len*self.label_hz)
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                if 'mesa' in pat:
                    channels_to_load = ['CZ-OZ','C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'mros' in pat:
                    channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'shhs' in pat:
                    channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'mgh' in pat:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                if 'robert' in pat:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                if 'stanford' in pat:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                
            
                #load data
                label_arousal = label_arousal[start_idx:max_arousal].reshape(-1,num_30s_seg*30*self.label_hz)
                eeg = load_h5_data(f,channels_to_load,0,max_hypno,num_30s_epochs=num_30s_seg)

                #eeg
                if self.scaled ==False:
                    eeg[:,:,:,[0,1,3]] = eeg[:,:,:,[0,1,3]]*1000000
                    eeg[:,:,:,[2]] = eeg[:,:,:,[2]]*1000000
                    eeg[:,:,:,[4]] = eeg[:,:,:,[4]]*100000

                
        except:
            eeg = np.zeros((1,35,3840,5))
            label_arousal = np.zeros((1,2100))

        # #test 
        # for xxxx in range(eeg.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,eeg[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')

        return eeg, label_arousal

def load_patients_val_arousal_aux(self,index,pat_list,num_30s_seg=35):
            
        pat = pat_list[index]
        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
            label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
            max_len = np.min((len(label_hypno)/self.data_hz,len(label_arousal)/self.label_hz))//(num_30s_seg*30)*(num_30s_seg*30)
            start_idx = 0
            max_hypno = int(max_len*self.data_hz)
            max_arousal = int(max_len*self.label_hz)
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_6(channels,self.chan_type)
                
                #load data
                label_hypno = np.mean(label_hypno[start_idx:max_hypno].reshape(-1,num_30s_seg,30*self.data_hz),axis=2)
                label_arousal = label_arousal[start_idx:max_arousal].reshape(-1,num_30s_seg*30*self.label_hz)
                eeg = load_h5_data(f,channels_to_load,0,max_hypno,num_30s_epochs=num_30s_seg)

                #eeg
                if self.scaled ==False:
                    eeg[:,:,:,[0,1,3]] = eeg[:,:,:,[0,1,3]]*1000000
                    eeg[:,:,:,[2]] = eeg[:,:,:,[2]]*1000000
                    eeg[:,:,:,[4]] = eeg[:,:,:,[4]]*100000

                
        except:
            eeg = np.zeros((1,35,3840,5))
            label_hypno = np.zeros((1,35))
            label_arousal = np.zeros((1,2100))

        # #test 
        # for xxxx in range(eeg.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,eeg[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')

        

        return eeg, [label_hypno,label_arousal]

def load_patients_val_arousal_yield(hparams,pat,num_30s_seg=35):
            

        try:
            #read label
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=hparams['data']['data_Hz'])
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=hparams['data']['data_Hz'])
            label_arousal = read_label(opj(pat,hparams['data']['label_type']+'.ids'),fs=hparams['data']['label_Hz'])
            
            max_len = np.min((len(label_hypno)/hparams['data']['data_Hz'],len(label_arousal)/hparams['data']['label_Hz']))//(num_30s_seg*30)*(num_30s_seg*30)
            start_idx = 0
            max_hypno = int(max_len*hparams['data']['data_Hz'])
            max_arousal = int(max_len*hparams['data']['label_Hz'])
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                channels = [x for x in f['channels'].keys()]
                channels_to_load = load_channels_6(channels,hparams['channels'])
                
                #load data
                label_arousal = label_arousal[start_idx:max_arousal].reshape(-1,num_30s_seg*30*hparams['data']['label_Hz'])
                eeg = load_h5_data(f,channels_to_load,0,max_hypno,num_30s_epochs=num_30s_seg)

                #eeg
                if hparams['data']['scaled'] ==False:
                    eeg[:,:,:,[0,1,3]] = eeg[:,:,:,[0,1,3]]*1000000
                    eeg[:,:,:,[2]] = eeg[:,:,:,[2]]*1000000
                    eeg[:,:,:,[4]] = eeg[:,:,:,[4]]*100000

                
        except:
            eeg = np.zeros((1,35,3840,5))
            label_arousal = np.zeros((1,2100))

        # #test 
        # for xxxx in range(eeg.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,eeg[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')
        label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)

        return [eeg, label_arousal]


def load_patients_arousal_shallow_n1(self,pat_list,start_label,batch_size=32,num_30s_seg=35,ref=False):
    
    #pre-allocate data
    EEG = np.zeros((batch_size,num_30s_seg,128*30,self.n_channels))
    LABEL_arousal = np.zeros((batch_size,30*num_30s_seg*int(self.label_hz)))

    #loop though patients
    i = 0
    success = 0
    while success<self.batch_size:
        

            pat =pat_list[i]
            
            try:
                #read hypno
                try:
                    label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                    label_hypno_labhz = read_label(opj(pat,'hypnogram.ids'),fs=self.label_hz)
                except:
                    label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                    label_hypno_labhz = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.label_hz)
                label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
                #if no arousals skip
                if max(label_arousal)==0:
                    i+=1
                    continue

                loc = np.where(label_hypno_labhz==3)[0]

                len_is_sufficient=True
                label_arousal = label_arousal[loc]
                if len(loc)<int(30*35*self.label_hz):
                    #print('not enough n1')
                    len_is_sufficient=False
                    label_arousal_ = np.zeros((int(30*35*self.label_hz)))
                    
                    label_arousal_[:int(len(loc))] = label_arousal
                    label_arousal = label_arousal_
                    start_loc=0


                else:
                    poss_start_loc = len(loc)-int(30*35*self.label_hz)
                    start_loc = np.random.choice(np.arange(poss_start_loc),size=1,replace=False)[0]

                    label_arousal = label_arousal[start_loc:start_loc+int(30*35*self.label_hz)]

                
                with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
                    if 'mesa' in pat:
                        channels_to_load = ['CZ-OZ','C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mros' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'shhs' in pat:
                        channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                    if 'mgh' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    if 'robert' in pat:
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                    
                    loc_data = np.where(label_hypno==3)[0]
                    loc_data = loc_data[int(start_loc/self.label_hz*self.data_hz):int(start_loc/self.label_hz*self.data_hz)+int(30*35*self.data_hz)]

                    #load data
                    if len_is_sufficient:
                        eeg = load_h5_data_idx(f,channels_to_load,loc_data,num_30s_epochs=num_30s_seg)
                    else:
                        eeg = np.zeros((int(30*35*self.data_hz),len(channels_to_load)))
                        for j,chan in enumerate(channels_to_load):
                                eeg[:len(loc_data),j] = f['channels'][chan][loc_data][()]
                        eeg = eeg.reshape((1,35,30*128,5))

                    # #test 
                    # for xxx in range(35):
                    #     for xx in range(5):
                    #         plt.plot(np.arange(3840)/3840+xxx,eeg[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                    #     plt.plot(np.arange(60)/60+xxx,label_arousal[60*xxx:60*(xxx+1)]-(7*(xx+1)))
                    # plt.show()
                    # print('a')
                    
                    EEG[success,:,:,:] = eeg

                    LABEL_arousal[success,:]= label_arousal
                    

                    i+=1
                    success+=1
            except:
                #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
                i+=1
        

                
    EEG=EEG[:success,:,:,:]
    if self.scaled ==False:
        #eeg
        EEG[:,:,:,[0,1,3]] = EEG[:,:,:,[0,1,3]]*1000000
        EEG[:,:,:,[2]] = EEG[:,:,:,[2]]*1000000
        EEG[:,:,:,[4]] = EEG[:,:,:,[4]]*100000
    LABEL_arousal=LABEL_arousal[:success,:]

        
    return EEG, LABEL_arousal



def load_patients_val_arousal_n1(self,index,pat_list,num_30s_seg=35,ref=False):
            
        pat = pat_list[index]
        try:
            try:
                label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
                label_hypno_labhz = read_label(opj(pat,'hypnogram.ids'),fs=self.label_hz)
            except:
                label_hypno = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.data_hz)
                label_hypno_labhz = read_label(opj(pat,'hypnogram_rater_majority.ids'),fs=self.label_hz)
            label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
            loc = np.where(label_hypno_labhz==3)[0]
            len_is_sufficient = True
            if len(loc)<(35*30*self.label_hz):
                len_is_sufficient=False
                loc_data = np.where(label_hypno==3)[0]
            else:
                max_len_label = len(loc)//(35*30*self.label_hz)*(35*30*self.label_hz)
                loc = loc[:max_len_label]

                loc_data = np.where(label_hypno==3)[0]
                max_len_data = len(loc_data)//(35*30*self.data_hz)*(35*30*self.data_hz)
                loc_data = loc_data[:max_len_data]


            label_arousal=label_arousal[loc]

            if not len_is_sufficient:
                factor = int(np.ceil((35*30*self.label_hz)/len(label_arousal)))
                label_arousal_ = np.tile(label_arousal, factor)
                label_arousal = label_arousal_[:int((35*30*self.label_hz))]

            start_idx = 0
            
            with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                if 'mesa' in pat:
                    channels_to_load = ['CZ-OZ','C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'mros' in pat:
                    channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'shhs' in pat:
                    channels_to_load = ['C3-M2', 'C4-M1', 'CHIN', 'E1-M2', 'ECG']
                if 'mgh' in pat:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
                if 'robert' in pat:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels_6(channels,self.chan_type,ref=ref)
            
                #load data
                label_arousal = label_arousal.reshape(-1,num_30s_seg*30*self.label_hz)
                if len_is_sufficient:
                    eeg = load_h5_data_idx(f,channels_to_load,loc_data,num_30s_epochs=num_30s_seg)
                else:
                    end = len(f['channels'][channels_to_load[0]])
                    eeg = np.zeros((end,len(channels_to_load)))
                    for i,chan in enumerate(channels_to_load):
                            eeg[:,i] = np.array(f['channels'][chan])
                    
                    eeg = eeg[loc_data,:]
                    eeg = [np.tile(eeg[:,i],factor) for i in range(eeg.shape[1])]
                    eeg= np.array(eeg)
                    eeg = np.swapaxes(eeg,0,1)
                    eeg = eeg.reshape((1,35,30*128,5))


                #eeg
                if self.scaled ==False:
                    eeg[:,:,:,[0,1,3]] = eeg[:,:,:,[0,1,3]]*1000000
                    eeg[:,:,:,[2]] = eeg[:,:,:,[2]]*1000000
                    eeg[:,:,:,[4]] = eeg[:,:,:,[4]]*100000

                
        except:
            eeg = np.zeros((1,35,3840,5))
            label_arousal = np.zeros((1,2100))

        # #test 
        # for xxxx in range(eeg.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,eeg[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')

        return eeg, label_arousal


# def load_patients_val_arousal_idx(self,index,pat_list,num_30s_seg=35):
            
#         pat = pat_list[index]
#         try:
#             #read label
#             label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#             label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
#             max_len = np.min((len(label_hypno)/self.data_hz,len(label_arousal)/self.label_hz))//(num_30s_seg*30)*(num_30s_seg*30)
#             with open(opj(pat,'arousal_start.idx')) as f:
#                 start_idx = int(f.readlines()[0])
#             end_idx = start_idx+(4*num_30s_seg*30*self.data_hz)
            
#             with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
#                 channels = [x for x in f['channels'].keys()]
#                 channels_to_load = load_channels_6(channels,self.chan_type)
                
#                 #load data
#                 label_hypno = np.mean(label_hypno[start_idx:end_idx].reshape(-1,num_30s_seg,30*self.data_hz),axis=2)
#                 label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)].reshape(-1,num_30s_seg*30*self.label_hz)
#                 eeg = load_h5_data(f,channels_to_load,start_idx,end_idx,num_30s_epochs=num_30s_seg)
                
#         except:
#             eeg = np.zeros((1,35,3840,5))
#             label_hypno = np.zeros((1,35))
#             label_arousal = np.zeros((1,2100))

#         return eeg, [label_hypno,label_arousal]


# def load_patients_val_arousal_PSD(self,index,pat_list,num_30s_seg=35):
            
#         pat = pat_list[index]
#         try:
#             #read label
#             label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#             label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
            
#             max_len = np.min((len(label_hypno)/self.data_hz,len(label_arousal)/self.label_hz))//(num_30s_seg*30)*(num_30s_seg*30)
#             start_idx = 0
#             max_hypno = int(max_len*self.data_hz)
#             max_arousal = int(max_len*self.label_hz)
            
#             with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
#                 channels = [x for x in f['channels'].keys()]
#                 channels_to_load = load_channels_6(channels,self.chan_type)
                
#                 #load data
#                 label_hypno = np.mean(label_hypno[start_idx:max_hypno].reshape(-1,num_30s_seg,30*self.data_hz),axis=2)
#                 label_arousal = label_arousal[start_idx:max_arousal].reshape(-1,num_30s_seg*30*self.label_hz)
#                 eeg = load_h5_data(f,channels_to_load,0,max_hypno,num_30s_epochs=num_30s_seg)
#                 try:
#                     pat_temp = glob(opj(pat,'*_psd.h5'))[0]
#                     pat_temp = pat_temp.replace('views/8k/fixed_split/val/','')
#                     pat_temp = pat_temp.replace('_psd.h5','_psd_for_training.npy')
#                     PSD = np.load(pat_temp)
#                 except:
#                     with h5.File(glob(opj(pat,'*_psd.h5'))[0],'r') as f:
                        
#                         times = np.array(f['PSD']['freq']) #incorrectly saved!! so freq == times
#                         start_idx_psd = np.where(np.abs(times - start_idx/self.label_hz) == np.min(np.abs(times - start_idx/self.label_hz)))[0][0]#idx for time in sec
#                         end_idx_psd = np.where(np.abs(times - max_arousal/self.label_hz) == np.min(np.abs(times - max_arousal/self.label_hz)))[0][0]#idx for time in sec
#                         PSD_ = np.array(f['PSD']['psd'][:,start_idx_psd:end_idx_psd])


#                         freq = np.array(f['PSD']['times'])#incorrectly saved!! so times == freq
#                         Alpha_range = np.arange(128,192)
#                         Beta_range = np.arange(192,319)
#                         Theta_range = np.arange(64,128)
#                         Delta_range = np.arange(8,64)
#                         PSD = np.zeros((4,self.label_shape_b2[1]*len(label_arousal)))
#                         len_psd = PSD_.shape[1]
#                         space = np.linspace(0,len_psd,self.label_shape_b2[1]*len(label_arousal)-2)
#                         for k in np.arange(self.label_shape_b2[1]*len(label_arousal)-3):
#                             PSD[0,k+3] = np.mean(np.mean(PSD_[Delta_range,int(space[k]):int(space[k+1])],axis=1))
#                             PSD[1,k+3] = np.mean(np.mean(PSD_[Theta_range,int(space[k]):int(space[k+1])],axis=1))
#                             PSD[2,k+3] = np.mean(np.mean(PSD_[Alpha_range,int(space[k]):int(space[k+1])],axis=1))
#                             PSD[3,k+3] = np.mean(np.mean(PSD_[Beta_range,int(space[k]):int(space[k+1])],axis=1))

#                         PSD = PSD.T.reshape(-1,2100,1,4)
#                         pat_temp = glob(opj(pat,'*_psd.h5'))[0]
#                         pat_temp = pat_temp.replace('views/8k/fixed_split/val/','')
#                         pat_temp = pat_temp.replace('_psd.h5','_psd_for_training.npy')
#                         np.save(pat_temp,PSD)
#         except:
#             eeg = np.zeros((64,num_30s_seg,30*self.data_hz,5))
#             label_hypno = np.zeros((64,num_30s_seg*30*self.data_hz,self.n_classes_b1))
#             label_arousal = np.zeros((64,num_30s_seg*30*self.data_hz,self.n_classes_b2))

#         return [eeg,PSD], [label_hypno,label_arousal]


# def load_patients_arousal_hem(self,pat_list,start_label,batch_size=32):
    
#     #pre-allocate data
#     EEG = np.zeros((batch_size,35,128*30,self.n_channels))
#     LABEL_hypno = np.zeros((batch_size,int(30*35*self.hypno_hz)))
#     LABEL_arousal = np.zeros((batch_size,30*35*int(self.label_hz)))

#     #loop though patients
#     i = 0
#     success = 0
#     while success<self.batch_size:
        
#         while success< self.load_hypno:
#             pat =pat_list[i]

#             try:
#                 #read hypno
#                 label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#                 label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
                
#                 #if no arousals skip
#                 if max(label_arousal)==0:
#                     i+=1
#                     continue

#                 #find suiting 17,5 minutes of data
#                 start_idx = find_start_idx_label(label_hypno,start_label[np.min((i,len(start_label)-1))])
#                 end_idx = start_idx+(35*30*self.data_hz)

#                 with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                    
#                     channels = [x for x in f['channels'].keys()]
#                     channels_to_load = load_channels_6(channels,self.chan_type)
                    
#                     #load data
#                     eeg = load_h5_data(f,channels_to_load,int(start_idx),int(end_idx))

#                     label_hypno_ = np.mean(label_hypno[int(start_idx):int(end_idx)].reshape(35,-1),axis=1)
                    
#                     if len(np.unique(label_hypno_))>6:
#                         a=b
#                     label_hypno = label_hypno_
#                     label_arousal = label_arousal[int(start_idx/self.data_hz*self.label_hz):int(end_idx/self.data_hz*self.label_hz)]
                    
#                     EEG[success,:,:,:] = eeg
#                     LABEL_hypno[success,:]= label_hypno
#                     LABEL_arousal[success,:]= label_arousal

#                     i+=1
#                     success+=1
#             except:
#                 #f = h5.File(glob(opj(pat,'*.h5'))[0],'r') 
#                 i+=1
        
#         while success>= self.load_hypno and success <self.load_hypno+self.load_arousal:
#             pat =pat_list[i]
#             try:
#                 #read hypno
#                 label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#                 label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)
#                 #if no arousals skip
#                 if max(label_arousal)==0:
#                     i+=1
#                     continue

#                 #find suiting 17,5 minutes of data
#                 diff_arousal_idx = np.diff(label_arousal)
#                 diff_up = np.where(diff_arousal_idx==1)[0]

#                 #choose random segment with arousal
#                 Poss_idx1 = np.random.choice(len(diff_up),size=1,replace=False)[0]
#                 start_idx = np.min((diff_up[int(Poss_idx1)],int(len(label_arousal)-(17.5*60*self.label_hz))))
#                 start_idx = np.max((start_idx,0))
#                 #match with hypno
#                 diff_hypno_idx = np.diff(label_hypno)
#                 diff = np.where(diff_hypno_idx)[0][0]%(35*30*self.data_hz)+1
#                 start_idx = int(diff/self.data_hz*self.label_hz)
#                 end_idx = start_idx+(35*30*self.label_hz)

#                 #load data
#                 with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                        
#                         channels = [x for x in f['channels'].keys()]
#                         channels_to_load = load_channels_6(channels,self.chan_type)
                        
#                         #load data
#                         print('please check line 385 from load_selecot_functions_for_loader.py')
#                         eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))
#                         label_hypno_ = np.mean(label_hypno[int(start_idx/self.label_hz*self.data_hz):int(end_idx/self.label_hz*self.data_hz)].reshape(35,-1),axis=1)
#                         label_hypno = label_hypno_
#                         label_arousal = label_arousal[int(start_idx):int(end_idx)]
                        
#                         EEG[success,:,:,:] = eeg
#                         LABEL_hypno[success,:]= label_hypno
#                         LABEL_arousal[success,:]= label_arousal

#                         i+=1
#                         success+=1
#             except: 
#                 i+=1

#         while success >=self.load_hypno+self.load_arousal and success <self.batch_size:
#             pat =pat_list[i]
#             try:
#                 #read hypno
#                 label_hypno = read_label(opj(pat,'hypnogram.ids'),fs=self.data_hz)
#                 label_arousal = read_label(opj(pat,self.label_type+'.ids'),fs=self.label_hz)

#                 pat_id = pat.split('/')[-1]+'_arousal.csv'
#                 label_pred,_ = cvs2lab(opj(self.pred_path,pat_id),data_fs=200,output_fs=2,label_name='pred_arousal',str='arousal')
#                 max_len = np.min((len(label_arousal),len(label_pred)))
#                 label_diff = label_pred[:max_len]-label_arousal[:max_len]
                

                
#                 #if no arousals skip
#                 if max(label_arousal)==0:
#                     i+=1
#                     continue

#                 #find suiting 17,5 minutes of data
#                 max_epoch = np.mean(label_diff.reshape(-1,30*self.label_hz),axis=1)
#                 max_epoch = moving_average(max_epoch,35)
#                 max_epoch = np.where(max_epoch==np.max(max_epoch))[0][0]
#                 start_idx = max_epoch*30*self.label_hz

                
#                 #match with hypno
#                 diff_hypno_idx = np.diff(label_hypno)
#                 diff = np.where(diff_hypno_idx)[0][0]%(35*30*self.data_hz)+1
#                 start_idx = int(diff/self.data_hz*self.label_hz)
#                 end_idx = start_idx+(35*30*self.label_hz)

#                 #load data
#                 with h5.File(glob(opj(pat,'*.h5'))[0],'r') as f:
                        
#                         channels = [x for x in f['channels'].keys()]
#                         channels_to_load = load_channels_6(channels,self.chan_type)
                        
#                         #load data
#                         eeg = load_h5_data(f,channels_to_load,int(start_idx/self.label_hz*self.data_hz),int(end_idx/self.label_hz*self.data_hz))

#                         label_hypno = np.mean(label_hypno[int(start_idx/self.label_hz*self.data_hz):int(end_idx/self.label_hz*self.data_hz)].reshape(35,-1),axis=1)
#                         label_arousal = label_arousal[int(start_idx):int(end_idx)]
                        
#                         EEG[success,:,:,:] = eeg
#                         LABEL_hypno[success,:]= label_hypno
#                         LABEL_arousal[success,:]= label_arousal

#                         i+=1
#                         success+=1
#             except: 
#                 i+=1

                
#     EEG=EEG[:success,:,:,:]
#     LABEL_hypno=LABEL_hypno[:success,:]
#     LABEL_arousal=LABEL_arousal[:success,:]

        
#     return EEG, [LABEL_hypno,LABEL_arousal]
