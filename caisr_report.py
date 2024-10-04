import sys, os, argparse, h5py, time
import numpy as np
import pandas as pd
from tqdm import tqdm

# Custom functions
sys.path.append('./report/')
from sleep_indices import compute_ahi, compute_plmi, compute_sfi, sleep_indices_stages, compute_arousal_index
from eeg_fct import filter_data, notch_filter, eeg_filter, spectrogram
sys.path.append('./preprocess/')
from prepare_data import compute_hr_from_ecg

""" Sleep Report Generation Functions

`combine_caisr_outputs(study_id, path_model_outputs)` combines the outputs from multiple sleep analysis models, such as sleep staging, arousal detection, respiratory event detection, and limb movement detection. It consolidates the individual model predictions for a specific study ID into a unified data structure for further processing.
**Parameters**:
- `study_id` (str): The unique identifier for the study session, typically referring to a subject's dataset.
- `path_model_outputs` (str): The base directory where the model outputs are stored. The function expects subdirectories or files corresponding to the study ID, containing the results of the different sleep models in standardized formats (e.g., `.csv` or `.h5`).

`generate_sleep_report(combined_output, path_report_output)` takes the combined output from the various models and generates a detailed sleep report. The report includes metrics such as total sleep time, sleep efficiency, the number of arousals, respiratory event count, and limb movements. The report is saved as a `.pdf` or `.html` file.
**Parameters**:
- `combined_output` (dict): A dictionary or data structure containing the combined outputs of the sleep event detection models, including sleep stages, arousals, respiratory events, and limb movements.
- `path_report_output` (str): The file path where the generated sleep report will be saved. The function supports output in formats like `.pdf`, `.html`, or `.txt`.

Note: The function assumes that the event detection models are run separately and their results are stored in the specified directory. It supports standardized formats for sleep event data, but the code can be adapted to other models and formats.
"""


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

def compute_macro_features(stage, arousal, resp, limb, fs, verbose=False):

    if resp is not None:
        ahi, rdi, oai, cai, mai, hyi, rerai, ahi_nrem, ahi_rem = compute_ahi(resp, stage=stage, fs=fs)
    else:
        if verbose:
            print('No resp annotation, skip computing respiratory indices')
        ahi, rdi, oai, cai, mai, hyi, rerai, ahi_nrem, ahi_rem = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        
    if limb is not None:
        lmi, plmi = compute_plmi(limb, stage=stage, fs=fs)
    else:
        lmi, plmi = np.nan, np.nan
    
    if stage is not None:
        sfi = compute_sfi(stage, fs)
        sleep_indices = sleep_indices_stages(stage, fs)
    else:
        if verbose:
            print('No stage annotation, skip computing sleep indices')
        sfi = np.nan
        sleep_indices = [np.nan] * 10

    if arousal is not None:
        arousal_index = compute_arousal_index(arousal, stage=stage, fs=fs)
    else:
        if verbose:
            print('No arousal annotation, skip computing arousal index')
        arousal_index = np.nan

    features = np.array(list(sleep_indices) + [sfi, ahi, ahi_nrem, ahi_rem, rdi, oai, cai, mai, hyi, 
        rerai, lmi, plmi, arousal_index])
    feature_names = ['hours_sleep', 'hours_psg', 'sleep_efficiency', 'perc_r', 'perc_n1', 'perc_n2', 'perc_n3', 'waso',
                    'sleep_latency', 'r_latency',  'sfi', 'ahi', 'ahi_nrem', 'ahi_rem', 'rdi', 'oai', 'cai', 'mai', 
                    'hyi', 'rerai', 'lmi', 'plmi', 'arousal_index']

    res = pd.DataFrame(data=features.reshape(1, -1), columns=feature_names)

    return res

def compute_sleep_metrics(df_caisr_annotations, fs=2):
    """Compute sleep metrics from the combined output of sleep analysis models.
    
    Args:
        df_caisr_annotations (pd.DataFrame): DataFrame containing the combined output from sleep analysis models.
        fs (int): Sampling frequency of the combined output annotations. Default is 2 Hz.
    
    Returns:
        df_metrics_study (pd.DataFrame): DataFrame containing the computed sleep metrics for the study.
    """
    df_metrics_study = pd.DataFrame()

    stage = df_caisr_annotations['stage']
    arousal = df_caisr_annotations['arousal']
    resp = df_caisr_annotations['resp']
    limb = df_caisr_annotations['limb']

    df_metrics_macro = compute_macro_features(stage, arousal, resp, limb, fs)

    # for now, only compute macro features:
    df_metrics_study = pd.concat([df_metrics_study, df_metrics_macro], axis=1)

    return df_metrics_study

def combine_caisr_outputs(file_id, path_dir_caisr_output, combined_folder, task_names):
    """Process multiple sleep data files in a specified directory.
    
    Args:
        path_dir_input (str): The directory containing the .edf and .csv files to be processed.
        path_dir_output (str): The directory where the processed .h5 files will be saved.
    """
    
    fs_prepared = 200
    fs_caisr_output = {
        'stage': 1,
        'arousal': 2,
        'resp': 1,
        'limb': 1,
    }

    df_annotations_combined = pd.DataFrame()

    for task_tmp in task_names:
        fs_caisr_output_tmp = fs_caisr_output[task_tmp]
        
        path_file_task = os.path.join(path_dir_caisr_output[task_tmp], f'{file_id}_{task_tmp}.csv')

        if not os.path.exists(path_file_task):
            print(f'File not found: {path_file_task}. Skip this file for now.')
            break

        df_annotation_task = pd.read_csv(path_file_task)
        for col in df_annotation_task.columns:
            if col not in [task_tmp, 'start_idx', 'end_idx']:
                df_annotation_task.rename(columns={col: f"{task_tmp}_{col}"}, inplace=True)
        
        # Combine and resample
        df_annotation_task = pd.DataFrame(np.repeat(df_annotation_task.values, fs_prepared // fs_caisr_output_tmp, axis=0), columns=df_annotation_task.columns)
        
        if len(df_annotation_task) != len(df_annotations_combined) and len(df_annotations_combined) != 0:
            raise Exception(f'Lenth combined output does not match "{task_tmp}": {len(df_annotations_combined)} vs {len(df_annotation_task)}')

        if task_tmp != 'arousal':
            df_annotation_task = df_annotation_task.drop(['start_idx', 'end_idx'], axis=1)

        # Add model task-specific output to the combined output
        df_annotations_combined = pd.concat([df_annotations_combined, df_annotation_task], axis=1)


    # replace any NaN with integer 9:
    df_annotations_combined = df_annotations_combined.fillna(9)

    # every col that starts with 'prob', round to 5 decimals. all other columns round to 0 and convert to int:
    for col in df_annotations_combined.columns:
        if col.startswith('prob'):
            df_annotations_combined[col] = df_annotations_combined[col].round(5)
        else:
            df_annotations_combined[col] = df_annotations_combined[col].round(0)
            df_annotations_combined[col] = df_annotations_combined[col].astype(int)
            
    df_annotations_combined = df_annotations_combined.drop(['limb_prob_no', 'limb_prob_limb'], axis=1)

    # move ['start_idx', 'end_idx'] to the front:
    if 'start_idx' in df_annotations_combined.columns:
        df_annotations_combined = df_annotations_combined[['start_idx', 'end_idx'] + [col for col in df_annotations_combined.columns if col not in ['start_idx', 'end_idx']]]

    # resample output annotations from 200 to 2 Hz:
    df_annotations_combined = df_annotations_combined.iloc[::100]

    # Save DF
    path_combined_csv = f'{combined_folder}caisr_{file_id}.csv'
    df_annotations_combined.to_csv(path_combined_csv, index=False) 


    return path_combined_csv

def generate_sleep_report_pdf(df_caisr_annotations, df_signals, params, sleep_metrics=None, legend=True):
    """Generate a sleep report in PDF format based on the combined output from different sleep analysis models.
    
    Args:
        df_caisr_annotations (pd.DataFrame): DataFrame containing the combined output from sleep analysis models.
        df_signals (pd.DataFrame): DataFrame file containing the signals data (e.g., EEG, ECG, respiratory signals) for visualization.
        sleep_metrics (pd.DataFrame): DataFrame containing the computed sleep metrics for the study.
    
    Returns:
        fig_sleep_report (matplotlib.figure.Figure): Matplotlib figure object containing the sleep report.
    """
    # Generate the sleep report using the combined output and signals data
    # ...

    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams['xtick.major.size'] = 1
    plt.rcParams['ytick.major.size'] = 2
    fs_signals = params['fs_signals']
    fs_annotations = params['fs_annotations']

    if 'position' in df_signals.columns:
        n_rows = 9
        height_ratios = [4, 4, 0.3, 1.5, 0.3, 2, 2, 1, 0.3]
    else:
        n_rows = 8
        height_ratios = [4, 4, 0.3, 1.5, 2, 2, 1, 0.3]
    
    # Events during Wake are not considered:
    sleep = np.isin(df_caisr_annotations['stage'].values.astype(float), [1, 2, 3, 4])
    df_caisr_annotations['resp'][~sleep] = 0
    df_caisr_annotations['arousal'][~sleep] = 0
    df_caisr_annotations['limb'][~sleep] = 0
    
    fig, ax = plt.subplots(n_rows, 1, figsize=(11, 9),
                           gridspec_kw={'height_ratios': height_ratios},
                           sharex=True)

    x_hours = np.arange(0, len(df_signals) / fs_signals * 2) / 3600 / 2 # 2 Hz

    ###### EEG spectrogram
    i = 0
    ch = 'c4-m1'
    eeg = df_signals[ch].values.copy().astype(float)[np.newaxis, :]
    eeg = eeg_filter(eeg, fs_signals)
    eeg = eeg * 1e6  # convert to microvolts
    specs, freq, eeg_segs = spectrogram(eeg, fs_signals, signaltype='eeg', epoch_time=2, epoch_step_time=1, bandwidth=2, fmin=0, fmax=20)
    specs = specs.squeeze().T

    spec_db_vmin = 0
    spec_db_vmax = 20
    im = ax[i].imshow(specs, cmap='turbo', origin='lower', aspect='auto',
                      vmin=spec_db_vmin, vmax=spec_db_vmax,
                      extent=(0, x_hours[-1], freq.min(), freq.max()))
    ax[i].set_ylabel(f'EEG {ch.upper()}')
    ax[i].set_xticklabels([])

    ####### hypnogram
    i = 1
    hypnogram_stages = df_caisr_annotations['stage'].values.astype(float)
    hypnogram_stages[hypnogram_stages == 9] = np.nan
    hypnogram_resp = df_caisr_annotations['resp'].values.astype(float)
    arousal = df_caisr_annotations['arousal'].values.copy().astype(float)
    arousal[arousal != 1] = np.nan
    arousal[arousal == 1] = 5.25
    ax[i].plot(x_hours, hypnogram_stages, c='k')
    rem_only = hypnogram_stages.copy()
    rem_only[rem_only != 4] = np.nan
    ax[i].plot(x_hours, rem_only, c='r')
    ax[i].plot(x_hours, arousal, c='orange', alpha=1, lw=3)
    ax[i].set_yticks([1, 2, 3, 4, 5])
    ax[i].set_yticklabels(['N3', 'N2', 'N1', 'R', 'W'])

    ax2 = ax[i].twinx()
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.plot(x_hours, hypnogram_resp, c='blue', alpha=0.3, lw=0.4)
    ax2.set_yticks([0, 1, 2, 3, 4, 5])
    ax2.set_yticklabels(['', 'OA', 'CA', 'MA', 'HY', 'RA'])

    ax[i].set_ylim([0.9, 5.5])
    ax2.set_ylim([0, 5.5])
    ax[i].set_xlim([0, x_hours[-1]])
    ax[i].set_xticklabels([])
    ax[i].set_ylabel('Hypnogram')

    ###### respiratory events
    i = 2
    palette_resp = ['magenta', 'green', 'cyan', 'blue', 'orangered', 'white']

    def resp_to_resp_df(resp, value_rera=5):
        resp_df = pd.DataFrame(columns=['Hypopnea', 'Central', 'Mixed', 'Obstructive', 'RERA', 'No Event'], index=range(len(resp)))
        resp_df[:] = 0
        resp_df.loc[resp == 0, 'No Event'] = 1
        resp_df.loc[resp == 1, 'Obstructive'] = 1
        resp_df.loc[resp == 2, 'Central'] = 1
        resp_df.loc[resp == 3, 'Mixed'] = 1
        resp_df.loc[resp == 4, 'Hypopnea'] = 1
        resp_df.loc[resp == value_rera, 'RERA'] = 1
        return resp_df

    df_resp = resp_to_resp_df(hypnogram_resp[::fs_annotations])
    df_resp.iloc[:, :-1] = df_resp.iloc[:, :-1].rolling(window=60, min_periods=1, axis=0, center=True).max()
    df_resp.iloc[df_resp.iloc[:, :-1].sum(axis=1) > 0, -1] = 0
    df_resp = df_resp.divide(df_resp.sum(axis=1).values, axis=0)  # normalization

    df_resp.index = np.linspace(0, x_hours[-1], len(df_resp))
    df_resp.plot(kind='area', color=palette_resp, alpha=1, sharex=True, ax=ax[i],
                 stacked=True, lw=0, legend=False)
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylim(0, 1)
    ax[i].set_yticks([])

    # SpO2
    i = 3
    spo2_values = df_signals['spo2'].values
    spo2_values = spo2_values[::fs_signals // fs_annotations]
    spo2_values[spo2_values < 40] = np.nan
    spo2_q50 = pd.Series(spo2_values).rolling(int(10 * fs_signals), min_periods=1, center=True).mean()
    spo2_q10 = pd.Series(spo2_values).rolling(int(10 * fs_signals), min_periods=1, center=True).quantile(0.025)
    spo2_q90 = pd.Series(spo2_values).rolling(int(10 * fs_signals), min_periods=1, center=True).quantile(0.975)

    ax[i].plot(x_hours, spo2_q50, c='darkblue', lw=0.5)
    ax[i].fill_between(x_hours, spo2_q10, spo2_q90, color='cornflowerblue', alpha=0.4)
    ylim = 80 if np.nanquantile(spo2_values, 0.1) > 80 else spo2_q10
    ax[i].set_ylim([ylim, 100])
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylabel('SpO2')

    # Position
    
    if 'position' in df_signals.columns:
        i += 1
        cm_position = ['white', 'black', 'blue', 'green', 'red', 'gray', 'white']
        cm_position = LinearSegmentedColormap.from_list("cm_reds", cm_position)

        position = df_signals['position'].values.copy().astype(float)

        ax[i].imshow(df_signals['position'].values[np.newaxis, :], cmap=cm_position, origin='lower', aspect='auto',
                    vmin=0, vmax=6,
                    extent=(0, x_hours[-1], 0, 1))
        ax[i].set_ylabel('Pos')
        ax[i].yaxis.set_label_position("right")
        ax[i].set_yticks([])

    # Heart rate
    i += 1
    hr_values = df_signals['hr'].values
    hr_values[hr_values < 30] = np.nan
    df_signals.loc[hr_values < 30, 'hr'] = np.nan
    hr_q50 = df_signals['hr'].rolling(int(30 * fs_signals), min_periods=1, center=True).mean()
    hr_q10 = df_signals['hr'].rolling(int(30 * fs_signals), min_periods=1, center=True).quantile(0.025)
    hr_q90 = df_signals['hr'].rolling(int(30 * fs_signals), min_periods=1, center=True).quantile(0.975)

    ax[i].plot(x_hours[::2], hr_q50[::fs_signals], c='darkred', lw=0.5)
    ax[i].fill_between(x_hours[::2], hr_q10[::fs_signals], hr_q90[::fs_signals], color='salmon', alpha=0.4)
    hr_ylim_min = max(20, np.nanquantile(hr_values, 0.01))
    hr_ylim_max = min(120, np.nanquantile(hr_values, 0.99))
    ax[i].set_ylim([hr_ylim_min, hr_ylim_max])
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylabel('HR')
    
    # Chin EMG
    i += 1
    chin = df_signals['chin1-chin2'].values
    chin = chin * 1e6  # convert to microvolts
    rolling_duration = int(0.1 * fs_signals)
    chin_q50 = pd.Series(chin).rolling(rolling_duration, min_periods=1, center=True).mean()
    chin_q10 = pd.Series(chin).rolling(rolling_duration, min_periods=1, center=True).quantile(0.025)
    chin_q90 = pd.Series(chin).rolling(rolling_duration, min_periods=1, center=True).quantile(0.975)
    ax[i].plot(x_hours, chin[::fs_signals // fs_annotations], c='saddlebrown', lw=0.4)
    ax[i].set_ylim(np.nanquantile(chin, 0.001), np.nanquantile(chin, 0.999))
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylabel('Chin')
    # remove any negative tick labels:
    yticklabels = [str(int(x)) if int(x) >= 0 else '' for x in ax[i].get_yticks()]
    ax[i].set_yticklabels(yticklabels)

    # Leg EMG
    i += 1
    leg = (df_signals['lat'].values + df_signals['rat'].values) / 2
    leg = leg * 1e6  # convert to microvolts
    ax[i].plot(x_hours, leg[::fs_signals//fs_annotations], c='darkgreen', lw=0.4)
    ax[i].set_ylim(min(-250, np.nanquantile(leg, 0.001)), max(250, np.nanquantile(leg, 0.999)))
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylabel('Leg') #  \nMoves')
    # ax[i].yaxis.set_label_position("right")
    yticklabels = [str(int(x)) if int(x) >= 0 else '' for x in ax[i].get_yticks()]
    ax[i].set_yticklabels(yticklabels)
    
    # Limb annotations
    i += 1
    palette_limb = ['black', 'red', 'white']

    def limb_to_limb_df(limb):
        df = pd.DataFrame(columns=['Isolated', 'Periodic', 'No Event'], index=range(len(limb)))
        df[:] = 0
        df.loc[limb == 0, 'No Event'] = 1
        df.loc[limb == 1, 'Isolated'] = 1
        df.loc[limb == 3, 'Isolated'] = 1
        df.loc[limb == 4, 'Isolated'] = 1
        df.loc[limb == 2, 'Periodic'] = 1
        return df

    df_limb = limb_to_limb_df(df_caisr_annotations['limb'].values[::fs_annotations])
    df_limb.iloc[:, :-1] = df_limb.iloc[:, :-1].rolling(window=30, min_periods=1, axis=0, center=True).max()
    df_limb.iloc[df_limb.iloc[:, :-1].sum(axis=1) > 0, -1] = 0
    df_limb = df_limb.divide(df_limb.sum(axis=1).values, axis=0)
    df_limb.index = np.linspace(0, x_hours[-1], len(df_limb))
    df_limb.plot(kind='area', color=palette_limb, alpha=1, sharex=True, ax=ax[i],
                 stacked=True, lw=0, legend=False)
    ax[i].set_xlim(0, x_hours[-1])
    ax[i].set_ylim(0, 1)
    ax[i].set_yticks([])

    ax[-1].set_xlabel('Time (hours)', labelpad=0.1)
    ax[-1].set_xticks(range(int(x_hours[-1]) + 1))
    ax[-1].set_xticklabels([str(int(x)) for x in ax[-1].get_xticks()])
    
    if 0:
        # Add time of day to the x-axis
        hour = params['start_time'].hour
        minute = params['start_time'].minute
        x_time = [str((hour + x) % 24).zfill(2) + ':' + str(minute).zfill(2) for x in range(int(x_hours[-1]) + 1)]
        ax[-1].set_xticklabels(x_time)

    ax[-1].tick_params(length=2)

    # Legend
    fig.align_ylabels(ax)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, bottom=0.16)

    fontsize_legend = 8
    axe_legend = fig.add_axes([0.05, 0.17, 0.9, 0.1])
    axe_legend.text(0, 1, 'Events:', fontsize=fontsize_legend, va='top')
    
    # Add legend details (Arousal, Apnea types, etc.)
    xpos = 0.15
    ypos = 0.88
    axe_legend.scatter([xpos], [ypos], c='orange', marker='s', alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Arousal', fontsize=fontsize_legend, va='center')    
    
    xpos = 0.37
    axe_legend.scatter([xpos], [ypos], c='blue', marker='s', alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Obstructive apnea (OA)', fontsize=fontsize_legend, va='center')
    
    xpos = 0.64
    axe_legend.scatter([xpos], [ypos], c='green', marker='s',alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Central apnea (CA)', fontsize=fontsize_legend, va='center')
    
    xpos = 0.15
    ypos = ypos - 0.22
    axe_legend.scatter([xpos], [ypos], c='cyan', marker='s',alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Mixed apnea (MA)', fontsize=fontsize_legend, va='center')
    
    xpos = 0.37
    axe_legend.scatter([xpos], [ypos], c='magenta', marker='s',alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Hypopnea (HY)', fontsize=fontsize_legend, va='center')
    
    xpos = 0.64
    axe_legend.scatter([xpos], [ypos], c='red', marker='s',alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Respiratory related arousal (RA)', fontsize=fontsize_legend, va='center')
    
    ypos = ypos - 0.22
    xpos = 0.15
    # black "Limb movement (LM)":
    axe_legend.scatter([xpos], [ypos], c='black', marker='s', alpha=1, lw=3)
    axe_legend.text(xpos+0.02, ypos-0.01, 'Limb movement', fontsize=fontsize_legend, va='center')
    
    if 'position' in df_signals.columns:
        axe_legend.text(0, 0.5, 'Body Position:', fontsize=fontsize_legend, va='top')
        xpos = 0.15
        ypos = 0.36
        axe_legend.scatter([xpos], [ypos], c='black', marker='s', alpha=1, lw=3)
        axe_legend.text(xpos+0.02, ypos-0.01, 'Supine', fontsize=fontsize_legend, va='center')
        
        xpos = 0.37
        axe_legend.scatter([xpos], [ypos], c='blue', marker='s', alpha=1, lw=3)
        axe_legend.text(xpos+0.02, ypos-0.01, 'Lateral left', fontsize=fontsize_legend, va='center')
        
        xpos = 0.64
        axe_legend.scatter([xpos], [ypos], c='green', marker='s', alpha=1, lw=3)
        axe_legend.text(xpos+0.02, ypos-0.01, 'Lateral right', fontsize=fontsize_legend, va='center')
        
        xpos = 0.82
        axe_legend.scatter([xpos], [ypos], c='red', marker='s', alpha=1, lw=3)
        axe_legend.text(xpos+0.02, ypos-0.01, 'Prone', fontsize=fontsize_legend, va='center')
        
        xpos = 0.94
        axe_legend.scatter([xpos], [ypos], c='gray', marker='s', alpha=1, lw=3)
        axe_legend.text(xpos+0.02, ypos-0.01, 'Upright', fontsize=fontsize_legend, va='center')
            
    # Additional positions in the legend
    axe_legend.set_xlim(0, 1)
    axe_legend.set_ylim(0, 1)
    axe_legend.axis('off')

    ### Add some text with the main variables
    
    # sleep_metrics: DataFrame with the following columns:
    # hours_sleep	hours_psg	sleep_efficiency	perc_r	perc_n1	perc_n2	perc_n3	waso	sleep_latency	r_latency	sfi	ahi	ahi_nrem	ahi_rem	rdi	oai	cai	mai	hyi	rerai	lmi	plmi	arousal_index
    metric_labels = {
        'hours_sleep': 'TST (h)',
        'hours_psg': 'Recording (h)',
        'sleep_efficiency': 'Eff (%)',

        'perc_r': 'REM (%)',
        'perc_n1': 'N1 (%)',
        'perc_n2': 'N2 (%)',

        'perc_n3': 'N3 (%)',
        'waso': 'WASO (min)',
        'sleep_latency': 'SL (min)',

        'sfi': 'SFI',
        'arousal_index': 'Arousal I.',
        'lmi': 'LMI',

        'ahi': 'AHI',
        'ahi_nrem': 'AHI NREM',
        'ahi_rem': 'AHI REM',

        'rdi': 'RDI',
        'oai': 'OAI',
        'cai': 'CAI',

        'mai': 'MAI',
        'hyi': 'HYI',
        'rerai': 'RERAI',
    }

    fontsize_statistics = 12  # Adjust as needed
    line_height = 0.025  # Adjust line spacing

    if sleep_metrics is not None:
        for i, (key, label) in enumerate(metric_labels.items()):
            value = sleep_metrics[key].item()
            if '%' in label:
                value *= 100
            if key in ['hours_sleep', 'hours_psg', 'sfi']:
                decimals = 1
            else:
                decimals = 0
            value = np.round(value, decimals)
            # Positioning text under the legend, adjust y-position as needed # make monospace font
            fig.text(0.1 + 0.30 * (i % 3), 0.17 - line_height * (i // 3), 
                        f"{label:<13} {value:.{decimals}f}",
                        fontsize=fontsize_statistics, va='center', ha='left', family='monospace')

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, bottom=0.32) 

    return fig

def generate_sleep_report(path_caisr_annotations, path_report_output=None, path_csv_metrics=None, path_signals_h5=None):
    """Generate a sleep report based on the combined output from different sleep analysis models.
    
    Args:
        path_caisr_annotations (str): Path to the combined output CSV file containing sleep annotations.
        path_report_output (str): Directory where the sleep report will be saved. Default is './caisr_output/caisr_reports/'.
        path_csv_metrics (str): Path to save the computed sleep metrics as a CSV file. Default is './caisr_output/caisr_reports/caisr_sleep_metrics.csv'.
        path_signals_h5 (str): Path to the signals data (e.g., EEG, ECG, respiratory signals) for visualization.
    """
    if path_report_output is None:
        path_report_output = path_repo
    if not os.path.exists(path_report_output):
        os.makedirs(path_report_output)
    if path_csv_metrics is None:
        path_csv_metrics = os.path.join(path_report_output, 'caisr_sleep_metrics_all_studies.csv')

    # Load the combined output CSV file
    df_caisr_annotations = pd.read_csv(path_caisr_annotations)

    """
    df_combined contains the following columns:
    'start_idx', 'end_idx': Start and end index in the prepared data (h5) file.
    'stage': Sleep stage. 5: Wake, 4: REM, 3: N1, 2: N2, 1: N3; 9: No stage
    'stage_prob_n3', 'stage_prob_n2', 'stage_prob_n1', 'stage_prob_r', 'stage_prob_w': Probabilities for each sleep stage
    'arousal': Arousal annotation. 1: Arousal, 0: No arousal
    'arousal_prob_no', 'arousal_prob_arousal',  Probabilities for arousal detection
    'arousal_pp_trace': Post-proessed arousal trace?
    'resp': Respiratory event annotation. 0: No event, 1: Obstructive apnea, 2: Central apnea, 3: Hypopnea, 4: RERA
    'resp_platinum_labels': To remove
    'limb': Limb movement annotation. 0: No movement, 1: Movement
    """

    params = {
        'fs_signals': 200,
        'fs_annotations': 2,
        'start_time': pd.Timestamp('2022-01-01 00:00:00'),
    }

    study_id = os.path.basename(path_caisr_annotations).replace('caisr_', '').replace('.csv', '')

    do_compute_metrics = True
    if do_compute_metrics:
        # Compute sleep metrics from the combined output (pandas DataFrame)
        df_metrics_study = compute_sleep_metrics(df_caisr_annotations)
        df_metrics_study.index = [study_id]
        df_metrics_study = df_metrics_study.round(2)

        if os.path.exists(path_csv_metrics):
            # Append the metrics to an existing file
            df_metrics_all = pd.read_csv(path_csv_metrics, index_col='study_id')
            assert len(df_metrics_all.columns) == len(df_metrics_study.columns), f"Number of columns in the existing metrics file ({len(df_metrics_all.columns)}) do not match the computed metrics ({len(df_metrics_study.columns)})"
            assert all(df_metrics_all.columns == df_metrics_study.columns), "Columns in the existing metrics file do not match the computed metrics."
            df_metrics_all = pd.concat([df_metrics_all, df_metrics_study])
            df_metrics_all = df_metrics_all[~df_metrics_all.index.duplicated(keep='last')]
            df_metrics_all.to_csv(path_csv_metrics, index_label='study_id')
            
        else:
            # Create a new metrics file
            df_metrics_all = df_metrics_study
            df_metrics_all.to_csv(path_csv_metrics, index_label='study_id')
    else:
        df_metrics_study = None


    do_generate_pdf_report = True
    if do_generate_pdf_report:
        assert path_signals_h5 is not None, "Path to the signals data (e.g., EEG, ECG, respiratory signals) is required for PDF report generation."
        #  Load the signals data (e.g., EEG, ECG, respiratory signals) for visualization, make dataframe array with column names as specifieid in the h5 file:
        # default signal names are:
        # ['abd', 'airflow', 'c3-m2', 'c4-m1', 'cflow', 'chest', 'chin1-chin2', 'cpap_on', 'cpres', 'e1-m2', 'e2-m1', 'ecg', 'f3-m2', 'f4-m1', 'lat', 'o1-m2', 'o2-m1', 'ptaf', 'rat', 'spo2']

        df_signals = pd.DataFrame()
        with h5py.File(path_signals_h5, 'r') as h5_file:
            for key in h5_file['signals'].keys():
                df_signals[key] = h5_file['signals'][key][:].flatten()
        
        if not 'hr' in df_signals.columns:
            df_signals['hr'] = compute_hr_from_ecg(df_signals.ecg.values, fs=params['fs_signals'])
            # anything outside of 30-200 is set to NaN
            df_signals.loc[(df_signals['hr'] < 30) | (df_signals['hr'] > 200), 'hr'] = np.nan

        # assert all of the minimum expected signals are available:

        # Generate the sleep report; returns matplotlib image
        fig_sleep_report = generate_sleep_report_pdf(df_caisr_annotations, df_signals, params, df_metrics_study)
        # Save the sleep report to the specified directory
        path_sleep_report = os.path.join(path_report_output, f'caisr_report_{study_id}.pdf')
        fig_sleep_report.savefig(path_sleep_report)

    # Generate the sleep report
    # ...

    # Save the sleep report to the specified directory
    path_sleep_report = os.path.join(path_report_output, f'sleep_report_{os.path.basename(path_combined_csv).replace(".csv", ".pdf")}')
    # ...

    return path_sleep_report


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run CAISR Report.")
    parser.add_argument("--data_folder", type=str, default='./data/', help="Folder containing the prepared .h5 data files.")
    parser.add_argument("--csv_folder", type=str, default='./caisr_output/', help="Folder where CAISR's output CSV files will be stored.")
    
    # Parse the arguments
    args = parser.parse_args()
    data_folder = args.data_folder.rstrip('/')
    csv_folder = args.csv_folder.rstrip('/')
    
    # Create combined and report folders within the output folder
    combined_folder = os.path.join(csv_folder, 'caisr_annotations/')
    path_repo = os.path.join(csv_folder, 'reports/')
    os.makedirs(combined_folder, exist_ok=True)
    os.makedirs(path_repo, exist_ok=True)
    
    # Add the report path to system paths
    sys.path.append(path_repo)
    
    # Set up paths to CAISR's task outputs
    task_names = ['stage', 'arousal', 'resp', 'limb']
    intermediate_folder = csv_folder + 'intermediate/'
    task_output_dirs = {task_name: os.path.join(intermediate_folder, task_name) for task_name in task_names}
    task_output_files = {}
    for task_name, task_dir in task_output_dirs.items():
        task_output_files_ = [file.replace(f"_{task_name}.csv", "") for file in os.listdir(task_dir)]
        task_output_files[task_name] = task_output_files_

    # Log the start of the process
    timer('* Starting "caisr_report" (created by Wolfgang Ganglberger, PhD)')

    # Identify studies with complete data for all tasks
    study_ids = list(set.intersection(*map(set, task_output_files.values())))
    print(f"Number of study IDs with complete data for all tasks: {len(study_ids)}\n")
    
    # Generate reports for each study
    completed = 0
    for study_id in tqdm(study_ids, desc="Generating sleep reports"):
        try:
            # Combine the outputs from different models into a unified CSV
            path_combined_csv = combine_caisr_outputs(study_id, task_output_dirs, combined_folder, task_names)
        except Exception as e:
            print(f'ERROR: {e} when combining CAISR\'s task CSV files for study {study_id}')
            continue

        try:
            # Generate a sleep report based on the combined output
            h5_file = os.path.join(data_folder, f'{study_id}.h5')
            if not os.path.exists(h5_file):
                raise FileNotFoundError(f"Signals data file not found: {h5_file}")
            
            path_sleep_report = generate_sleep_report(path_combined_csv, path_repo, path_signals_h5=h5_file)
            completed += 1
        except Exception as e:
            print(f'ERROR: {e} when creating report for study {study_id}')
            continue

    print(f'Reports created for {completed}/{len(study_ids)} studies.')

    # Log the end of the process
    timer('* Finishing "caisr_report"')