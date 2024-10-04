from arousal.utils.pre_processing.scaling import *
from arousal.utils.pre_processing.quality_control_funcs import *

def apply_pre_process(data):
    EEG = np.zeros((2,len(data)))
    EEG[0,:] = data
    EEG[1,:] = data
    psg_scaled_clipped = apply_scaling(clip_noisy_values(EEG.T, 128, len(EEG.T)/128,min_max_times_global_iqr=20)[0], 'RobustScaler')[0]
    return psg_scaled_clipped[:,0]
    