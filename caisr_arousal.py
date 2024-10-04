import os, sys, time, argparse, mne, logging, warnings
import h5py as h5 
import numpy as np 
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
from glob import glob
from typing import List, Tuple
from scipy.stats import iqr 
from scipy.signal import resample
from arousal.utils.models.model_init import *
from arousal.utils.load_write.ids2label import *
from arousal.utils.pre_processing.scaling import *
from arousal.utils.post_processing.label_class import *
from arousal.utils.hyperparameters.hyperparameters import *
from arousal.utils.pre_processing.quality_control_funcs import *
from arousal.utils.post_processing.smoothing_arousal import movav
from arousal.utils.post_processing.rem_post_processing import rem_post_processing
from arousal.utils.post_processing.smoothing_arousal import post_process_after_smoothing
from mne.preprocessing import create_ecg_epochs, EOGRegression
from os.path import join as opj
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
mne.set_log_level(verbose='CRITICAL')

################################
# functions for pre_processing #
################################
def timer(tag: str) -> None:
    """
    Displays a simple progress bar for the given tag, printing dots incrementally for aesthetics.
    
    Args:
    - tag (str): The string label for which the progress bar is shown.
    """

    print(tag)
    # simple progress bar for aesthetics
    for i in range(1, len(tag) + 1):
        print('.' * i + '     ', end='\r')
        time.sleep(1.5 / len(tag))  # Time delay proportional to the length of the tag
    print()

def extract_run_parameters(param_csv: str) -> List[bool]:
    assert os.path.exists(param_csv), 'run parameter file is not found.'
    
    # load run parameters from .csv
    params = pd.read_csv(param_csv)
    overwrite = params['overwrite'].values[0]
    
    return overwrite

def set_output_paths(input_paths: List[str], csv_folder: str, overwrite: bool) -> Tuple[List[str], List[str]]:
    """
    Sets up output paths for CSV files based on input paths and folders.
    
    Args:
    - input_paths (List[str]): List of input file paths.
    - csv_folder (str): Folder to save CSV files.
    - overwrite (bool): Whether to overwrite already processed files.

    Returns:
    - Tuple[List[str], List[str]]: Filtered input paths, CSV paths.
    """

    total = len(input_paths)
    # Extract file IDs from input paths
    IDs = [p.split('/')[-1].split('.')[0] for p in input_paths]
    # Create corresponding CSV paths
    csv_paths = [f'{csv_folder}arousal/{ID}_arousal.csv' for ID in IDs]

    # Ensure the number of paths is consistent
    assert len(input_paths) == len(csv_paths), 'SETUP ERROR: The number of input and CSV files is not equal.'

    # If overwrite is False, filter out already processed files
    input_paths, csv_paths = filter_already_processed_files(input_paths, csv_paths, overwrite)

    return input_paths, csv_paths

def filter_already_processed_files(input_paths: List[str], csv_paths: List[str], overwrite: bool) -> Tuple[List[str], List[str]]:
    """
    Filters out already processed files, unless overwrite is specified.
    
    Args:
    - input_paths (List[str]): List of input file paths.
    - csv_paths (List[str]): List of CSV output file paths.
    - overwrite (bool): Whether to overwrite already processed files.

    Returns:
    - Tuple[List[str], List[str]]: Updated lists of input, CSV paths.
    """

    total = len(input_paths)
    todo_indices = [p for p, path in enumerate(csv_paths) if not os.path.exists(path)]

    # If not overwriting, keep only unprocessed files
    if not overwrite:
        input_paths, csv_paths = filter_todo_files(input_paths, csv_paths, todo_indices)

    tag = '(overwrite) ' if overwrite else ''
    print(f'>> {total - len(todo_indices)}/{total} files already processed\n>> {len(input_paths)} to go.. {tag}\n')

    return input_paths, csv_paths

def filter_todo_files(input_paths: List[str], csv_paths: List[str], keep_indices: List[int]) -> Tuple[List[str], List[str]]:
    """
    Filters input, CSV paths based on indices of files to process.
    
    Args:
    - input_paths (List[str]): List of input file paths.
    - csv_paths (List[str]): List of CSV output file paths.
    - keep_indices (List[int]): Indices of files to keep for processing.

    Returns:
    - Tuple[List[str], List[str]]: Filtered lists of input, CSV paths.
    """

    input_paths = np.array(input_paths)[keep_indices].tolist()
    csv_paths = np.array(csv_paths)[keep_indices].tolist()

    return input_paths, csv_paths

def movav(data,window_width,center=True):
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

def clip_chin(image,filtered=False):

    #preallocating
    rolling_abs_diff_image = np.zeros(len(image))
    rolling_abs_diff_image_ref = np.zeros(len(image))

    if filtered == False:
        #remove baseline drifts and low frequencies
        drift = np.squeeze(pd.DataFrame({'image':image}).rolling(50,center=True,min_periods=0).mean())
        image = image-drift

    #rolling averages
    rolling_abs_diff_image[:-1] = np.squeeze(pd.DataFrame({'image':np.abs(np.diff(image))}).rolling(50,center=True,min_periods=0).mean())
    rolling_abs_diff_image_ref[:-1] = np.squeeze(pd.DataFrame({'image':np.abs(np.diff(image))}).rolling(1000,center=True,min_periods=0).mean())

    #thresholds
    thres_max = rolling_abs_diff_image_ref+rolling_abs_diff_image*1.5
    thres_min =-rolling_abs_diff_image_ref-rolling_abs_diff_image*1.5

    #thresholding
    image[image>thres_max]=thres_max[image>thres_max]
    image[image<thres_min]=thres_min[image<thres_min]

    if filtered == False:
        #adding drift back in 
        image = image+drift

    #clip high amplitude artifacts
    
    rolling = np.squeeze(pd.DataFrame({'image':np.abs(image)}).rolling(200000,center=True,min_periods=0).mean())*3
    rolling2 = np.squeeze(pd.DataFrame({'image':np.abs(image)}).rolling(10000,center=True,min_periods=0).std())
    image_save = image.copy()
    #clip positive side
    image[rolling2>rolling]=np.min((rolling2[rolling2>rolling],image[rolling2>rolling]),axis=0)
    #clip negative side
    image[-rolling2<-rolling]=np.max((-rolling2[-rolling2<-rolling],image[-rolling2<-rolling]),axis=0)
    return image

def chin_unscaled(data):


    ####################
    # filter EKG 
    ####################

    #add zeros for referencing
    data_extra = np.zeros((data.shape[0]+1,data.shape[1]))
    data_extra[:data.shape[0],:] = data.copy()

    # create mne file 
    info  = mne.create_info(ch_names=['M2','ECG','M1'],
                            sfreq=128,
                            ch_types=['eeg','ecg','eeg']
                            )
    raw = mne.io.RawArray(data_extra, info, first_samp=0, verbose=None)

    #reference with zeros, (already referenced)
    raw.set_eeg_reference(ref_channels=['M1'])

    #make montage
    ten_twenty_montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(ten_twenty_montage)
    #set epochs
    #We need to explicitly specify that we want to average the EOG channel too.
    ecg_evoked = create_ecg_epochs(raw).average('all')
    # perform regression on the evoked ecg response
    model_evoked = EOGRegression(picks='eeg', picks_artifact='ecg').fit(ecg_evoked)
    ecg_evoked_clean = model_evoked.apply(ecg_evoked)
    ecg_evoked_clean.apply_baseline()
    raw_clean_ = model_evoked.apply(raw)
    data_clean = raw_clean_.get_data()

    #threshold using rolling mean and std
    emg_model_input = data_clean[-3,:].copy()
    emg_mov_mean = pd.DataFrame({'emg':emg_model_input}).rolling(128, min_periods=1,center=True).mean().values
    emg_mov_std = pd.DataFrame({'emg':emg_model_input}).rolling(128, min_periods=1,center=True).std().values*2.5
    emg_mov_mean[np.isnan(emg_mov_mean)]=1
    emg_mov_std[np.isnan(emg_mov_std)]=1
    emg_mov_mean = np.squeeze(emg_mov_mean)
    emg_mov_std = np.squeeze(emg_mov_std)
    emg_mov_std_upper = emg_mov_mean+emg_mov_std
    emg_mov_std_lower = emg_mov_mean-emg_mov_std
    emg_model_input[emg_model_input>emg_mov_std_upper] = emg_mov_std_upper[emg_model_input>emg_mov_std_upper]
    emg_model_input[emg_model_input<emg_mov_std_lower] = emg_mov_std_lower[emg_model_input<emg_mov_std_lower]
    std = pd.DataFrame({'emg':emg_model_input}).std().values*20
    emg_model_input[emg_model_input>std]=std
    emg_model_input[emg_model_input<-std]=-std
    
    #Filter EMG data for rem post processing
    info  = mne.create_info(ch_names=['M2','ECG','M1'],
                            sfreq=128,
                            ch_types=['eeg','ecg','eeg']
                            )
    raw = mne.io.RawArray(data_clean, info, first_samp=0, verbose=None)

    #reference with zeros, (already referenced)
    raw.set_eeg_reference(ref_channels=['M1'])
    
    #filter for post processing
    raw = raw.filter(l_freq=10, h_freq=45)
    emg = raw.get_data()[-3,:]

    #threshold using rolling mean and std
    emg_mov = pd.DataFrame({'emg':emg}).rolling(128, min_periods=1,center=True).std().values*2.5
    emg_mov[np.isnan(emg_mov)]=1
    emg_mov = np.squeeze(emg_mov)
    emg[emg>emg_mov] = emg_mov[emg>emg_mov]
    emg[emg<-emg_mov] = -emg_mov[emg<-emg_mov]
    std = pd.DataFrame({'emg':emg}).std().values*20
    emg[emg>std]=std
    emg[emg<-std]=-std

    return emg,emg_model_input

def do_initial_preprocessing(signals, new_Fs,chin_filter=False):
    from mne.filter import filter_data, notch_filter
    from scipy.signal import resample_poly
    notch_freq_us = 60.                 # [Hz]
    notch_freq_eur = 50.                # [Hz]
    bandpass_freq_eeg = [0.2, 35]       # [Hz] [0.5, 40]
    bandpass_freq_airflow = [0., 10]    # [Hz]
    bandpass_freq_ecg = [0.2, 35]     # [Hz]
    original_Fs = 200                   # [Hz]
    # setup new signal DF
    new_df = pd.DataFrame([], columns=signals.columns)

    # 1. Notch filter
    for sig in signals.columns:
        image = signals[sig].values
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1','cz-oz', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2','chin','abd', 'chest', 'airflow', 'ptaf', 'cflow', 'ecg']:
            image = notch_filter(image, 200, notch_freq_us, verbose=False)
            # image = notch_filter(image, 200, notch_freq_eur, verbose=False)

        # 2. Bandpass filter
        if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'cz-oz','o1-m2', 'o2-m1', 'e1-m2']:
            image = filter_data(image, 200, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['chin1-chin2', 'chin']:
            image = clip_chin(image,chin_filter)
            image = filter_data(np.array(image), 200, bandpass_freq_eeg[0], bandpass_freq_eeg[1], verbose=False)
        if sig in ['abd', 'chest', 'airflow', 'ptaf', 'cflow']:
            image = filter_data(image, 200, bandpass_freq_airflow[0], bandpass_freq_airflow[1], verbose=False)
        if sig == 'ecg':
            image = filter_data(image, 200, bandpass_freq_ecg[0], bandpass_freq_ecg[1], verbose=False)

        # 3. Resample data
        if new_Fs != original_Fs:
            if sig in ['f3-m2', 'f4-m1', 'c3-m2', 'c4-m1', 'o1-m2', 'o2-m1', 'e1-m2', 'chin1-chin2', 'chin','abd', 'chest', 'airflow', 'ptaf', 'cflow', 'ecg']:
                image = resample_poly(image, new_Fs, original_Fs)
            else:
                image = image[::2]

        # 4. Insert in new DataFrame
        new_df.loc[:, sig] = image
    
    del signals
    return new_df

def len_lines(lines):
    lst =  [pos for pos, char in enumerate(lines) if char == ',']

    return int(lines[:lst[0]])+ int(lines[lst[0]+1:lst[1]])

def apply_pre_process(data, channel_type,time_min=10):
    EEG = np.zeros((2,len(data)))
    EEG[0,:] = data
    EEG[1,:] = data
    #clip
    psg_scaled_clipped = clip_noisy_values(EEG.T, 128, len(EEG.T)/128,min_max_times_global_iqr=20)[0]
    #copy
    psg_scaled_clipped_cut = psg_scaled_clipped[:,0].copy()
    #get time res
    temp_res = time_min*60*128
    #clip to time res
    psg_scaled_clipped_cut = psg_scaled_clipped_cut[:len(psg_scaled_clipped_cut)//temp_res*temp_res]
    #reshape
    psg_scaled_clipped_cut = psg_scaled_clipped_cut.reshape(-1,temp_res)
    #get features
    median_data = np.median(psg_scaled_clipped_cut,axis=1)
    iqr_data = iqr(psg_scaled_clipped_cut,axis=1)
    if channel_type== 'EEG': 
        iqr_data[iqr_data<8.3e-06] = 8.3e-06
        iqr_data[iqr_data>3.5e-05] = 3.5e-05
    if channel_type== 'ECG': 
        iqr_data[iqr_data<2.4e-05] = 2.4e-05
        iqr_data[iqr_data>0.0002] = 0.0002
    if channel_type== 'CHIN': 
        iqr_data[iqr_data<7.2e-07] = 7.2e-07
        iqr_data[iqr_data>2e-05] = 2e-05
    if channel_type== 'EYE': 
        iqr_data[iqr_data<8.4e-06] = 8.4e-06
        iqr_data[iqr_data>3.7e-05] = 3.7e-05

    iqr_data = iqr_data.repeat(temp_res)
    median_data = median_data.repeat(temp_res)

    #create the data trace with correct length
    #create zeros and add the iqr of the cutted segmetn
    iqr_data_trace = np.zeros(len(psg_scaled_clipped[:,0]))+iqr(psg_scaled_clipped[len(psg_scaled_clipped_cut)//temp_res*temp_res:,0])
    iqr_data_trace[:len(iqr_data)]=iqr_data
    median_data_trace = np.zeros(len(psg_scaled_clipped[:,0]))+np.median(psg_scaled_clipped[:,0])
    median_data_trace[:len(median_data)]=median_data
    iqr_ma = movav(iqr_data_trace,temp_res*4)
    med_ma = movav(median_data_trace,temp_res*4)


    eeg_scaled_clipped = (psg_scaled_clipped[:,0]-med_ma)/iqr_ma
    return eeg_scaled_clipped
    
def load_h5_signals(path):

    # read in signals
    import h5py
    f = h5py.File(path, 'r')
    keys = [k for k in f.keys()]
    signals = {}
    for key in keys:
        signals[key] = f[key][:]

    if 'channel_names' in signals.keys():
        signals['channel_names'] = list(signals['channel_names'].astype(str))
        
    return signals

def pre_process_temporal_scaling(file,path_write,channels,channel_type,temporal_resolution=5,chin_filter=False):


    #find patient name
    patient_name = file.split('/')[-1][:-3]

    #create folder and file
    #os.makedirs(path_write+'/'+patient_name+'/',exist_ok=True)
    
    # read pat file
    with h5.File(file, 'r') as f:
        for i,c in enumerate(channels):
            if i==0:
                data = np.zeros((len(channels),len(f['signals'][c])))
            data[i,:] = np.squeeze(np.array(f['signals'][c]))

    # #grab data for CHIN_FOR_REM
    if 'CHIN_RAW' in channel_type:
        #grab ecg
        idx = channel_type.index('ECG')
        ecg_unprocessed = data[idx,:].copy()
        #set channel name to chin_raw
        idx_c = [i for i,c in enumerate(channels) if 'chin' in c][-1]
        channels[idx_c]='chin_raw'
        #grab chin
        idx = channel_type.index('CHIN_RAW')
        chin_raw_unprocessed = data[idx,:].copy()
        #combine for post_processing
        chin_ecg_unprocessed = np.vstack((chin_raw_unprocessed,ecg_unprocessed))
        #calculate chin channel
        chin,chin_model_input = chin_unscaled(chin_ecg_unprocessed)

        #replace regular chin with new chin channel
        idx = channel_type.index('CHIN')
        data[idx,:] = chin_model_input
        


    data = pd.DataFrame(data.T, columns=channels)
    data = do_initial_preprocessing(data, 200,chin_filter)
    data = data[channels].values.T 

    #replace chin with  chin_raw channel
    idx_chin_raw = channel_type.index('CHIN_RAW')
    data[idx_chin_raw,:] = chin

    #resample
    data = resample(data,int(len(data.T)/200*128),axis=1)

    #write
    hf = h5.File(path_write, 'w')
    hf.attrs['sample_rate']=128
    dtypef = 'float32'
    for i, [c,ct] in enumerate(zip(channels,channel_type)):
        if 'raw' not in c:
            if c == 'chin':
                c = 'chin1-chin2'
            hf.create_dataset(f'channels/{c.upper()}', data=apply_pre_process(data[i,:],ct,time_min=temporal_resolution), dtype=dtypef, compression="gzip")
    hf.create_dataset(f'channels/CHIN_REM', data=data[idx_chin_raw,:], dtype=dtypef, compression="gzip")
    hf.close()

    return path_write


##############
# prediction #
##############
def predict(f,model_path,channels_to_load,hypno,EEG_chan=6):    
    
    hparams = YAMLHParams('arousal/utils/models/FINAL_MODELS/hparams.yaml')
    input_Hz = 128
    output_Hz = 2
    build = False
    
    #load data
    EMG = f['channels']['CHIN_REM'][()]

    if EEG_chan==2:
        channels_to_load = [['C3-M2', 'C4-M1', 'CHIN1-CHIN2', 'E1-M2', 'ECG']]
    if EEG_chan==6: 
        channels_to_load = [['F3-M2', 'F4-M1', 'CHIN1-CHIN2', 'E1-M2', 'ECG'],['C3-M2', 'C4-M1', 'CHIN1-CHIN2', 'E1-M2', 'ECG'],['O1-M2', 'O2-M1', 'CHIN1-CHIN2', 'E1-M2', 'ECG']]

    prediction = np.zeros((len(channels_to_load),int(len(f['channels'][channels_to_load[0][0]])/input_Hz*output_Hz),2))
    k=0

    #load eeg
    EEG = np.zeros((len(f['channels'][channels_to_load[0][0]]),len(channels_to_load[0])))*np.nan
    for i,c in enumerate(channels_to_load[0]):
        EEG[:,i] = np.array(f['channels'][c])
    EEG = EEG[:EEG.shape[0]//3840*3840,:]

    #Build Model
    if build == False:
        if i%90==0:
            pass #reset_keras()
        hparams["build"]["batch_shape"][1] = int(len(EEG[:,0])/3840)
        hparams["build"]["batch_shape"][0] = 1  # Should not matter
        model = init_model(hparams["build"]) 
        model.load_weights(model_path+'Best_Model.h5')
        build = True

    #Predict
    # print('PREDICTING CHANNELS: {}-{}'.format(channels_to_load[0][0],channels_to_load[0][1]))
    if hparams["build"]["data_per_prediction"] ==64:
        prediction[k,:int(EEG.shape[0]/64),:] = model.predict(EEG.reshape((1,-1,hparams['data']['data_Hz']*30,5)))
        k+=1 

    #load more channels
    if len(channels_to_load)>1:
        for chan_to_load in channels_to_load[1:]:
            #load eeg
            EEG = np.zeros((len(f['channels'][chan_to_load[0]]),len(chan_to_load)))
            for i,c in enumerate(chan_to_load):
                EEG[:,i] = np.array(f['channels'][c])
            EEG = EEG[:EEG.shape[0]//3840*3840,:]

            #predict 
            # print('PREDICTING CHANNELS: {}-{}'.format(chan_to_load[0],chan_to_load[1]))
            prediction[k,:int(EEG.shape[0]/64),:] = model.predict(EEG.reshape((1,-1,hparams['data']['data_Hz']*30,5))) 
            k+=1

        
    # mean
    prediction = prediction.mean(axis=0) #sqeeze and average over all channels
    
    #rem post processing for final prediction
    arousal_pred = np.sqrt(prediction[:,1].copy())
    arousal_pred = arousal_pred.repeat(64)
    arousal_pred = movav(arousal_pred,window_width=192*3,center=True)
    
    #pp trace 
    arousal_pred_pp_trace = arousal_pred.copy()
    threshold_pp_trace = np.zeros(len(arousal_pred))+0.2

    arousal_pred_pp_trace[arousal_pred_pp_trace<threshold_pp_trace]=0
    arousal_pred_pp_trace[arousal_pred_pp_trace>0]=1
    pp_trace = rem_post_processing(EMG,arousal_pred_pp_trace,hypno,output_pp_trace=True)
    pp_trace2 = np.round(np.mean(pp_trace.reshape(-1,64),axis=1))

    #threshold arousal

    threshold = np.zeros(len(arousal_pred))+1.1
    thres_n1 = 0.35
    thres_n2 = 0.45
    thres_n3 = 0.42
    thres_r = 0.51

    n1_loc = np.where(hypno==3)[0]
    n2_loc = np.where(hypno==2)[0]
    n3_loc = np.where(hypno==1)[0]
    r_loc = np.where(hypno==4)[0]

    threshold[n1_loc]=thres_n1
    threshold[n2_loc]=thres_n2
    threshold[n3_loc]=thres_n3
    threshold[r_loc]=thres_r

    arousal_pred[arousal_pred<threshold]=0
    arousal_pred[arousal_pred>0]=1
    
    #rem_post_processing
    arousal_pred = rem_post_processing(EMG,arousal_pred,hypno)

    
    input_dict = {'hypno':hypno,
            'arousal':arousal_pred,
            'limp':hypno*0,
            'resp':hypno*0
                }
    
    pred_lab = CAISER_labels(input_dict,fs=128)
    pred_lab.set_uncertainty_label(0)
    pred_lab.set_unstable_sleep_label(0)
    pred_lab = pred_lab.get_masked_arousal_label_eval()
    pred_lab = post_process_after_smoothing(pred_lab,[1],N_remove_length=2.5,N_max_len=30,Hz=128)
    arousal_pred = np.round(np.mean(arousal_pred.reshape(-1,64),axis=1))


    pred_per_subject = arousal_pred
    t1 = ((np.arange(pred_per_subject.shape[0]))/output_Hz*200).astype(int)
    t2 = ((np.arange(pred_per_subject.shape[0])+1)/output_Hz*200).astype(int)

    df = pd.DataFrame({'start_idx':t1,
                            'end_idx':t2,
                            'arousal':pred_per_subject,
                            'prob_no':prediction[:,0],
                            'prob_arousal':prediction[:,1],
                            'pp_trace':pp_trace2
                            })

    return df


def main(in_paths, save_paths,
         Channels=['f3-m2','f4-m1','c3-m2','c4-m1','o1-m2','o2-m1','e1-m2','chin1-chin2','ecg','chin1-chin2'],
         Channel_type = ['EEG','EEG','EEG','EEG','EEG','EEG','EYE','CHIN','ECG','CHIN_RAW'],
         temporal_resolution = 5
    ):

    # Run over all files
    for num, (input_path, write_path) in enumerate(zip(in_paths, save_paths)):
        each_id = input_path.split('/')[-1]
        the_id = each_id[:-3]
        tag = the_id if len(the_id)<21 else the_id[:20]+'..'
        print(f'(# {num + 1}/{len(in_paths)}) Processing "{tag}"')

        # Set paths
        hypno_path = write_path.replace('arousal', 'stage')
        write_path_pp = f'{the_id}.h5'

        # Skip files that do not have a matching stage .csv
        if not os.path.exists(hypno_path):
            print('No matching "..stage.csv" was found.')
            continue

        try:
            channels = Channels.copy()
            channel_type = Channel_type
    
            #check for FLS pats
            df_FLS_pats = pd.read_csv('arousal/utils/splits/model_path_for_split.csv')
            
            if (df_FLS_pats['File_name'] == the_id).any():
                model_path = 'arousal/' + df_FLS_pats[df_FLS_pats['File_name']==the_id]['Model_path'].values[0] + '/'
            else:
                model_path = f'arousal/utils/models/FINAL_MODELS/Model_CAISR_AROUSAL/'  

            #pre-process file
            write_path_pp = pre_process_temporal_scaling(input_path,write_path_pp,channels,channel_type,temporal_resolution,chin_filter=False)
            
            #read in Hypno
            hypnogram = pd.read_csv(hypno_path)['stage'].fillna(5).values.repeat(128)

            #read in patient and predict
            with h5.File(write_path_pp,'r') as hf:
                try:
                    df = predict(hf,model_path,channels,hypnogram,EEG_chan=6)
                except:
                    df = predict(hf,model_path,channels,hypnogram,EEG_chan=2)

            #write csv 
            df.to_csv(write_path,index=False)
            os.remove(write_path_pp)
        except:
            try:
                channels=['f3-m2','f4-m1','c3-m2','c4-m1','o1-m2','o2-m1','e1-m2','chin','ecg','chin']
                channel_type = Channel_type
                
                #check for FLS pats
                df_FLS_pats = pd.read_csv('arousal/utils/splits/model_path_for_split.csv')
                
                if (df_FLS_pats['File_name'] == the_id).any():
                    model_path = 'arousal/' + df_FLS_pats[df_FLS_pats['File_name']==the_id]['Model_path'].values[0] + '/'
                else:
                    model_path = 'arousal/utils/models/FINAL_MODELS/Model_CAISR_AROUSAL/'  

                #pre-process file
                write_path_pp = pre_process_temporal_scaling(input_path,write_path_pp,channels,channel_type,temporal_resolution,chin_filter=False)
                
                #read in Hypno
                hypnogram = pd.read_csv(hypno_path)['stage'].fillna(5).values.repeat(128)

                #read in patient and predict
                with h5.File(write_path_pp,'r') as hf:
                    try:
                        df = predict(hf,model_path,channels,hypnogram,EEG_chan=6)
                    except:
                        df = predict(hf,model_path,channels,hypnogram,EEG_chan=2)

                #write csv 
                df.to_csv(write_path,index=False)
                os.remove(write_path_pp)
            except:
                channels=['f3-m2','f4-m1','cz-oz','c4-m1','o1-m2','o2-m1','e1-m2','chin','ecg','chin']
                channel_type = Channel_type
                
                #check for FLS pats
                df_FLS_pats = pd.read_csv('arousal/utils/splits/model_path_for_split.csv')
                
                if (df_FLS_pats['File_name'] == the_id).any():
                    model_path = 'arousal/' + df_FLS_pats[df_FLS_pats['File_name']==the_id]['Model_path'].values[0] + '/'
                else:
                    model_path = f'arousal/utils/models/FINAL_MODELS/Model_CAISR_AROUSAL/'  

                #pre-process file
                write_path_pp = pre_process_temporal_scaling(input_path,write_path_pp,channels,channel_type,temporal_resolution,chin_filter=False)
                
                #read in Hypno
                hypnogram = pd.read_csv(hypno_path)['stage'].fillna(5).values.repeat(128)

                #read in patient and predict
                with h5.File(write_path_pp,'r') as hf:
                    try:
                        df = predict(hf,model_path,channels,hypnogram,EEG_chan=6)
                    except:
                        df = predict(hf,model_path,channels,hypnogram,EEG_chan=2)

                #write csv 
                df.to_csv(write_path,index=False)
                os.remove(write_path_pp)
            
    
    

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run CAISR arousal event detection.")
    parser.add_argument("--input_path", type=str, default='./data/', help="Folder containing the prepared .h5 data files.")
    parser.add_argument("--save_path", type=str, default='./caisr_output/intermediate/', help="Folder where caisr's output CSV files will be stored.")
    args = parser.parse_args()

    # Log the start of the process
    timer('* Starting "caisr_arousal" (created by Erik-Jan Meulenbrugge)')

    # Extract run parameters
    input_files = glob(args.input_path + '*.h5')
    overwrite = extract_run_parameters(args.input_path + 'run_parameters/arousal.csv')

    # Set output paths
    in_paths, save_paths = set_output_paths(input_files, args.save_path, overwrite)

    # Run CAISR resp
    main(in_paths, save_paths)

    # Log the end of the process
    timer('* Finishing "caisr_arousal"')


#Models you can choose from
# --model Model_CAISR_AROUSAL
# --model Model_2_FSL_BITS_FOLD_1
# --model Model_2_FSL_BITS_FOLD_2
# --model Model_2_FSL_STS_FOLD_1
# --model Model_2_FSL_STS_FOLD_2
# --model Model_1