import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Plotting functions:

def final_plot(data: pd.DataFrame, hdr: dict, algo_version: str, event_tag: str = '') -> None:
    """
    Plots the final report for the patient, including respiration trace, respiratory effort belts,
    and saturation with EEG arousals.

    Args:
    - data: DataFrame containing patient data and annotations.
    - hdr: Dictionary with metadata information, including 'newFs', 'patient_tag', 'test_type', and 'rec_type'.
    - algo_version: String specifying the version of the algorithm being used.
    - event_tag: Optional string for event tags (default is '').
    """
    Fs = hdr['newFs']
    patient_tag = hdr['patient_tag'].split('~')[0]
    test_type = hdr['test_type']
    rec_type = hdr['rec_type']
    fontdict = {'fontsize': 8}
    
    # Set patient asleep state based on sleep stages
    data['patient_asleep'] = data.Stage < 5

    # Plot respiration trace and events
    fig = plt.figure(figsize=(9.5, 6))

    ##########################################################################################################
    # SUBPLOT 1: Respiration trace with events
    ax1 = fig.add_subplot(311)
    if algo_version == 'full':
        plt.suptitle(f'{rec_type} - {test_type} - {patient_tag}', fontsize=10)

        # Plot split line for PTAF <--> CPAP if applicable
        if hdr.get('cpap_start') is not None:
            ax1.plot([hdr['cpap_start'], hdr['cpap_start']],
                     [np.min(data.Ventilation_combined), np.max(data.Ventilation_combined)],
                     c='r', linestyle='dashed', zorder=10)

    # Plot Ventilation baseline and combined signal
    ax1.plot(data['Ventilation_baseline'].mask(np.isnan(data.Ventilation_combined)), 'k', lw=0.5)
    ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep == 0), 'k', lw=0.5, alpha=0.5)
    ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep == 1), 'k', lw=0.5, alpha=0.1)
    ax1.plot(data['Ventilation_combined'].mask(data.bad_signal == 0), 'm', lw=0.5, alpha=0.5)

    # Plot airflow if available
    if 'airflow' in data.columns:
        ax1.plot(data['airflow'].mask(data.patient_asleep == 0) - 7, 'b', lw=0.5)
        ax1.plot(data['airflow'].mask(data.patient_asleep == 1) - 7, 'b', lw=0.5, alpha=0.1)
        ax1.plot(data['airflow'].mask(data.bad_signal == 0) - 7, 'm', lw=0.5, alpha=0.5)

    # excursions
    # ax1.plot(data['pos_excursion_hyp'].mask(np.isnan(data.Ventilation_combined)),'m', lw=0.5, alpha=0.7)
    # ax1.plot(data['neg_excursion_hyp'].mask(np.isnan(data.Ventilation_combined)),'m', lw=0.5, alpha=0.7)
    # ax1.plot(data['pos_excursion_apnea'].mask(np.isnan(data.Ventilation_combined)),'b', lw=0.5, alpha=0.7)
    # ax1.plot(data['neg_excursion_apnea'].mask(np.isnan(data.Ventilation_combined)),'b', lw=0.5, alpha=0.7)

# Plot apnea labels by experts
    colors = ['b', 'g', 'c', 'm', 'r']
    if 'resp-h3_platinum' in data.columns:
        for i, color in enumerate(colors):
            mask1 = np.logical_or(data['resp-h3_platinum'] != i + 1, data.patient_asleep == 0)
            mask2 = np.logical_or(data['resp-h3_platinum'] != i + 1, data.patient_asleep == 1)
            ax1.plot(data['resp-h3_platinum'].mask(mask1) - 0.2, color + '*', lw=2)
            ax1.plot(data['resp-h3_platinum'].mask(mask2) - 0.2, color + '*', lw=2, alpha=0.2)

    # Handle datasets with multiple scorers or special cases
    if hdr['dataset'] in ['jedi', 'stanford']:
        for exp in range(3):
            for i, color in enumerate(colors):
                mask1 = np.logical_or(data[f'resp-h3_expert_{exp}'] != i + 1, data.patient_asleep == 0)
                mask2 = np.logical_or(data[f'resp-h3_expert_{exp}'] != i + 1, data.patient_asleep == 1)
                ax1.plot(data[f'resp-h3_expert_{exp}'].mask(mask1) + (exp * 0.3), color, lw=1.5)
                ax1.plot(data[f'resp-h3_expert_{exp}'].mask(mask2) + (exp * 0.3), color, lw=1.5, alpha=0.2)

    elif hdr['dataset'] == 'qa':
        scorers = [s for s in data.columns if 'resp-h3_' in s]
        for s, scorer in enumerate(scorers):
            for i, color in enumerate(colors):
                mask1 = np.logical_or(data[scorer] != i + 1, data.patient_asleep == 0)
                mask2 = np.logical_or(data[scorer] != i + 1, data.patient_asleep == 1)
                ax1.plot(data[scorer].mask(mask1) + (s * 0.1), color, lw=2)
                ax1.plot(data[scorer].mask(mask2) + (s * 0.1), color, lw=2, alpha=0.2)
            if s == 9:
                break

    elif hdr['dataset'] in ['mros', 'mesa']:
        scorers = ['Apnea', 'resp-h0_expert_0']
        for s, scorer in enumerate(scorers):
            for i, color in enumerate(colors):
                alpha = 1 if s==0 else 0.2
                mask1 = np.logical_or(data[scorer]!= i+1, data.patient_asleep==0)
                mask2 = np.logical_or(data[scorer]!= i+1, data.patient_asleep==1)
                ax1.plot(data[scorer].mask(mask1), color, lw=2, alpha=alpha)
                ax1.plot(data[scorer].mask(mask2), color, lw=2, alpha=0.2)
    else:
        for i, color in enumerate(colors):
            mask1 = np.logical_or(data['Apnea']!= i+1, data.patient_asleep==0)
            mask2 = np.logical_or(data['Apnea']!= i+1, data.patient_asleep==1)
            ax1.plot(data['Apnea'].mask(mask1), color, lw=2)
            ax1.plot(data['Apnea'].mask(mask2), color, lw=2, alpha=0.2)

    # Plot algorithm-detected events
    for i, color in enumerate(colors):
        mask1 = np.logical_or(data['algo_apneas'] != i + 1, data.patient_asleep == 0)
        mask2 = np.logical_or(data['algo_apneas'] != i + 1, data.patient_asleep == 1)
        ax1.plot(-data['algo_apneas'].mask(mask1), color, lw=2)
        ax1.plot(-data['algo_apneas'].mask(mask2), color, lw=2, alpha=0.2)

    ax1.plot(-data['effort_RERAs'].mask(data['effort_RERAs'] != 6) + 0.5, 'b', lw=2)
    ax1.plot(-data['morph_RERAs'].mask(data['morph_RERAs'] != 6), 'k', lw=2)   

    # plot REJECTED hypopnea ventilation drops
    # ax1.plot(-5*data['rejected_saturation_hypopneas'].mask(data['rejected_saturation_hypopneas'] != 1), 'k')
    # ax1.plot(-5*data['rejected_EEG_hypopneas'].mask(data['rejected_EEG_hypopneas'] != 1), 'k')

    # plot apneas based on mean [abd, chest]
    # for i, color in enumerate(colors):
    #     ax1.plot(-0.5-data['Ventilation_drop_apnea1'].mask(data['Ventilation_drop_apnea1'] != i+1), color+'--', lw=2)


    # ax1.plot(-6*data['either_hypes'].mask(data['either_hypes'] != 1), 'm')
    # ax1.plot(-6.5*data['long_events'].mask(data['long_events'] != 1), 'r')
    # ax1.plot(-7*data['too_long_events'].mask(data['too_long_events'] != 1), 'r--')

    # subplot layout
    ax1.set_title('Respiration trace', fontdict=fontdict, pad=-1)
    ax1.set_ylim([-11.1,8])
    # ax1.xaxis.set_visible(False)


    ##########################################################################################################
    # SUBPLOT 2: Respiratory effort belts
    abd = data.ABD.rolling(int(0.5 * Fs), center=True).median().mask(data.patient_asleep == 0)
    abd_s = data.ABD.rolling(int(0.5 * Fs), center=True).median().mask(data.patient_asleep == 1)
    chest = data.CHEST.rolling(int(0.5 * Fs), center=True).median().mask(data.patient_asleep == 0)
    chest_s = data.CHEST.rolling(int(0.5 * Fs), center=True).median().mask(data.patient_asleep == 1)
    effort_trace = abd + chest
    effort_trace_s = abd_s + chest_s

    ax2 = fig.add_subplot(312, sharex=ax1)
    ax2.plot(abd, 'b', lw=0.5, alpha=0.7)
    ax2.plot(abd_s, 'b', lw=0.5, alpha=0.1)
    ax2.plot(chest - 5, 'g', lw=0.5, alpha=0.7)
    ax2.plot(chest_s - 5, 'g', lw=0.5, alpha=0.1)
    ax2.plot(effort_trace + 5, 'k', lw=1, alpha=0.7)
    ax2.plot(effort_trace_s + 5, 'k', lw=1, alpha=0.1)

    ax2.set_title('Effort Belts', fontdict=fontdict, pad=-1)
    ax2.set_ylim([-8, 8])
    # ax2.xaxis.set_visible(False)

    ##########################################################################################################
    # SUBPLOT 3: Saturation and EEG arousals
    ax3 = fig.add_subplot(313, sharex=ax1)
    ax3.plot(data['SpO2'].mask(data.patient_asleep == 0), 'k', alpha=0.5)
    ax3.plot(data['SpO2'].mask(data.patient_asleep == 1), 'r', alpha=0.5)
    ax3.plot(data['smooth_saturation'], 'b')

    ax3.plot(data['saturation_drop'].mask(data['saturation_drop'] == 0) + 99, 'm', lw=2)
    ax3.plot(data['EEG_arousals'].mask(data['EEG_arousals'] == 0) + 100, 'g', lw=2)
    
    if 'soft_3%_desats' in data.columns:
        ax3.plot(data['soft_3%_desats'].mask(data['soft_3%_desats'] == 0)+98.75, 'r', lw=2)
    if 'arousal_platinum' in data.columns:
        ax3.plot(data['arousal_platinum'].mask(data['arousal_platinum']==0)+100.25, 'g', lw=2, alpha=0.4)
    
    # subplot layout
    ax3.set_title('Saturation (m) and EEG arousals (g)', fontdict=fontdict, pad=-1)
    ax3.set_ylim([90,103])
    ax3.set_ylabel('%')






