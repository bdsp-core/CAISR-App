import os, sys, glob, h5py
import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.signal import convolve, detrend

# set parent folder
RB_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# import utils
sys.path.insert(0, f'{RB_folder}/utils_functions/')
from Event_array_modifiers import *
# import additional algorithm functions
sys.path.insert(0, f'{RB_folder}/algorithm_functions/')
from Ventilation_envelope import compute_envelope
from Post_processing import remove_short_events

# RERA analysis functions:
def RERA_detection(data: pd.DataFrame, hdr: dict, thresh: float = 0.55, quantile: float = 0.80, plot: bool = False) -> pd.DataFrame:
    """
    Detects Respiratory Effort-Related Arousals (RERAs) in the respiratory signal using convolution with
    predefined kernel templates and identifying increases in inspiratory effort.

    Args:
    - data: DataFrame containing respiratory and EEG signals.
    - hdr: Dictionary containing metadata (e.g., newFs, test_type, rec_type, patient_tag).
    - thresh: Threshold value for detecting RERAs using convolution scores.
    - quantile: Quantile used for selecting high-convolution scores.
    - plot: Boolean flag to plot the detection results if set to True.

    Returns:
    - Updated DataFrame with detected RERA events.
    """
    
    Fs = hdr['newFs']
    test_type = hdr['test_type']
    rec_type = hdr['rec_type']
    patient_tag = hdr['patient_tag'].split('~')[0].replace('_', ' ')
    
    # Initialize columns for RERA detection
    data['algo_reras'] = 0
    data['RERA_morphology'] = np.nan
    data['RERA_morphology_score_flat'] = np.nan
    data['RERA_morphology_score_nonflat'] = np.nan
    data['morph_RERAs'] = 0
    data['effort_RERAs'] = 0

    # Return early if no EEG arousals are found
    data['potential_RERA_arousals'] = data['EEG_arousals']
    eeg_arousals = find_events(data['potential_RERA_arousals'].fillna(0) > 0)
    if len(eeg_arousals) == 0:
        return data

    # Load RERA kernel templates (flat and non-flat)
    kernel_paths = glob.glob(f'{RB_folder}/RERA_kernels/*.hf5')
    kernels_flat, kernels_nonflat = None, None
    for path in kernel_paths:
        kernels = load_kernels(path)
        if "non_flat_kernels" in path:
            kernels_nonflat = kernels
        else:
            kernels_flat = kernels

    # Identify inspiratory flattening using the pre-defined kernels
    data = find_RERA_morphology(data, eeg_arousals, kernels_flat, kernels_nonflat, Fs, quantile, thresh)

    # Detect increased inspiratory effort using chest and abdominal belts
    data = find_increased_inspiratory_effort(data, hdr)

    # Locate recovery breaths in the ventilation signal
    data = find_recovery_breaths(data, Fs)

    # Create labels for detected RERA events
    data = create_RERA_labels(data, eeg_arousals, Fs)

    # Remove short RERAs that last less than 3 seconds
    data['algo_reras'] = remove_short_events(data['algo_reras'], 3 * hdr['newFs'])

    # Plot results if the flag is set to True
    if plot:
        fig = plt.figure(figsize=(9.6, 6))
        fontdict = {'fontsize': 9}

        #################
        # Plot respiratory signals + labels + arousals
        ax1 = fig.add_subplot(311)
        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep == 0), 'y')
        ax1.plot(data['Ventilation_combined'].mask(data.patient_asleep == 1), 'r')
        ax1.plot(data.EEG_arousals.mask(data.EEG_arousals == 0) * 6, 'k', alpha=0.5)

        # Plot apnea labels by the experts
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 1)), 'b')
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 2)), 'g')
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 3)), 'c')
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 4)), 'm')
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 5)), 'r')
        ax1.plot(data.algo_reras.mask(data.algo_reras != 5) + 0.5, 'r', lw=3)

        # Add recovery breaths
        peaks = np.array(data['Ventilation_combined'])
        peaks[data['recovery_breaths'] != 1] = np.nan
        ax1.plot(peaks, '*', c='b')

        # Layout settings
        ax1.set_ylim([-7.1, 8])
        ax1.set_title(f'Respiration trace - {patient_tag} - {rec_type} - {test_type}', fontdict=fontdict, pad=-1)

        #################
        # RERA analysis
        ax2 = fig.add_subplot(312, sharex=ax1)
        ax2.set_title('Best scores, flat (r) vs non-flat (k)', fontdict=fontdict, pad=-1)

        # Convolution scores
        ax2.scatter(data.index, data['RERA_morphology_score_flat'].values, c='r', s=4)
        ax2.scatter(data.index, data['RERA_morphology_score_nonflat'].values, c='k', s=4)
        ax2.plot(np.arange(len(data)), [thresh] * len(data), 'b--', lw=0.5)

        # Different RERA types
        ax2.plot(data['morph_RERAs'].mask(data['morph_RERAs'] == 0) / 6 * 1.2, 'r')
        ax2.plot(data['flowlim_RERAs'].mask(data['flowlim_RERAs'] == 0) / 6 * 1.3, 'b')

        # Original RERAs
        ax2.plot(data.Apnea.mask(data.Apnea != 5) / 6 * 1.5, 'r')
        ax2.set_ylim([-0.1, 1.7])

        #################
        # RIP signals
        ax3 = fig.add_subplot(313, sharex=ax1)
        ax3.set_title('Abdominal and Chest effort (RIP)', fontdict=fontdict, pad=-1)
        ax3.plot(data['ABD'].mask(data.patient_asleep == 0), 'b')
        ax3.plot(data['ABD'].mask(data.patient_asleep == 1), 'r')
        ax3.plot(data['CHEST'].mask(data.patient_asleep == 0), 'g')
        ax3.plot(data['CHEST'].mask(data.patient_asleep == 1), 'r')
        ax3.set_ylim([-10, 10])

        plt.show()
        import pdb; pdb.set_trace()

    return data

def find_RERA_morphology(data: pd.DataFrame, locs: List[Tuple[int, int]], kernels_flat: np.ndarray, kernels_nonflat: np.ndarray, Fs: int, quantile: float, thresh: float) -> pd.DataFrame:
    """
    Identifies Respiratory Effort-Related Arousals (RERAs) by convolving the respiratory signal with 
    predefined kernels and comparing the results for inspiratory flattening.

    Args:
    - data: DataFrame containing respiratory and EEG signals.
    - locs: List of tuples indicating start and end locations of EEG spontaneous arousals.
    - kernels_flat: Array of kernels for detecting flat inspiratory shapes.
    - kernels_nonflat: Array of kernels for detecting non-flat inspiratory shapes.
    - Fs: Sampling frequency of the data.
    - quantile: Quantile used for selecting high-convolution scores.
    - thresh: Threshold for convolution score below which a breath is considered non-related.

    Returns:
    - Updated DataFrame with RERA morphology scores.
    """

    # Iterate over all EEG spontaneous arousals
    for st, end in locs:
        # Specify region of interest: 20 seconds before start to 5 seconds after end
        candidate = data.Ventilation_combined[st - 20 * Fs : end + 5 * Fs]
        if np.all(np.isnan(candidate)):
            continue

        # Apply flat and non-flat kernels to the data
        _, conv_score_flat_kernels = apply_kernels(candidate.values, kernels_flat, Fs)
        _, conv_score_nonflat_kernels = apply_kernels(candidate.values, kernels_nonflat, Fs)

        # Compute the mean kernel scores using the specified quantile
        score_flat = np.quantile(conv_score_flat_kernels, quantile, axis=0)
        score_nonflat = np.quantile(conv_score_nonflat_kernels, quantile, axis=0)

        # Compare best flat vs best non-flat convolutions
        is_flat = score_flat > score_nonflat

        # Remove results with low convolution scores (<thresh) as they are likely unrelated to breaths
        is_flat[score_flat < thresh] = 0
        is_flat = is_flat.astype(float)

        # Replace zeros with NaN for better handling of missing data
        score_flat[score_flat == 0] = np.nan
        score_nonflat[score_nonflat == 0] = np.nan
        is_flat[is_flat == 0] = np.nan

        # Insert convolution scores and flatness indicators into the DataFrame
        data.loc[candidate.index, 'RERA_morphology'] = is_flat
        data.loc[candidate.index, 'RERA_morphology_score_flat'] = score_flat
        data.loc[candidate.index, 'RERA_morphology_score_nonflat'] = score_nonflat

    return data

def apply_kernels(candidate: np.ndarray, kernels: List[np.ndarray], Fs: int) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Applies a set of kernels to a candidate signal to detect patterns. The function convolves
    the candidate signal with each kernel and keeps track of the highest convolution scores.

    Args:
    - candidate: The respiratory signal around an arousal event.
    - kernels: A list of kernels to be applied for convolution.
    - Fs: Sampling frequency.

    Returns:
    - best_scores: Array containing the highest convolution scores at each point.
    - individual_scores: List of arrays, each containing the convolution scores for a corresponding kernel.
    """
    # Initialize empty arrays for best scores and individual kernel scores
    best_scores = np.zeros(candidate.shape)
    individual_scores = []

    # Iterate over each kernel
    for kernel in kernels:
        # Initialize the convolution score for each kernel
        ind_conv_score = np.zeros(candidate.shape)

        # Iterate through the candidate signal in steps of Fs//2
        for i_start in range(0, len(candidate) - len(kernel), Fs // 2):

            # Select segment and normalize it
            candidate_seg = candidate[i_start: i_start + len(kernel)]
            candidate_seg = (candidate_seg - np.mean(candidate_seg)) / (np.std(candidate_seg) + 1e-6)
            
            # Skip if the segment contains all NaN values
            if np.all(np.isnan(candidate_seg)):
                continue

            # Apply convolution and get the maximum score
            convolution_score = np.nanmax(convolve(candidate_seg, kernel, mode='same')) / len(kernel)

            # Determine the center index of the segment
            center_index = i_start + len(kernel)

            # Update individual kernel score for the current position if higher than the existing score
            if convolution_score > ind_conv_score[center_index]:
                ind_conv_score[center_index] = convolution_score

            # Update the best convolution score across all kernels
            if convolution_score > best_scores[center_index]:
                best_scores[center_index] = convolution_score

        # Append the individual convolution score for this kernel
        individual_scores.append(ind_conv_score)

    return best_scores, individual_scores

def find_increased_inspiratory_effort(data: pd.DataFrame, hdr: dict, insp_effort_thresh: float = 1.5, plot: bool = False) -> pd.DataFrame:
    """
    Identifies periods of increased inspiratory effort based on abdominal and chest signals.
    
    Args:
    - data: DataFrame containing the respiratory effort data.
    - hdr: Dictionary containing header information, including 'newFs' (sampling frequency).
    - insp_effort_thresh: Threshold for identifying significant effort increases.
    - plot: If True, plots the inspiratory effort and detected increases.

    Returns:
    - Updated DataFrame with 'RERA_increased_effort' column indicating increased inspiratory effort.
    """
    Fs = hdr['newFs']  # Sampling frequency
    data['RERA_increased_effort'] = 0  # Initialize the column for increased effort detection

    # Iterate over both abdominal (ABD) and chest (CHEST) signals
    for eff in ['ABD', 'CHEST']:
        # Smooth the effort signal using a rolling median
        effort = data[eff].rolling(Fs, center=True).median()

        # Compute the envelope and baseline for the smoothed effort signal
        new_df = compute_envelope(effort, Fs, env_smooth=20)
        baseline = new_df['baseline']
        pos_envelope = new_df['pos_envelope']

        # Calculate the distance to the baseline to find the sudden high peaks
        dist_to_base = pos_envelope - baseline

        # Apply threshold to detect significant effort increases
        effort[effort < insp_effort_thresh] = 0
        data['RERA_increased_effort'] += effort > (1.5 * dist_to_base)

    # Plotting the inspiratory effort increase, if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.6, 6), sharex=True)
        fontdict = {'fontsize': 9}

        # Plot the abdominal and chest traces
        ax1.set_title('Abdominal and Chest Effort Traces', fontdict=fontdict, pad=-1)
        ax1.plot(data['ABD'].mask(data.patient_asleep == 0), 'b', label='ABD (awake)')
        ax1.plot(data['ABD'].mask(data.patient_asleep == 1), 'r', label='ABD (asleep)')
        ax1.plot(data['CHEST'].mask(data.patient_asleep == 0), 'g', label='CHEST (awake)')
        ax1.plot(data['CHEST'].mask(data.patient_asleep == 1), 'r', label='CHEST (asleep)')

        # Plot apnea labels by experts and PhysioNet labels
        apneas = data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea == 6))
        apneas[apneas > 0] = 5
        ax1.plot(apneas, 'k', label='Apneas')
        ax1.plot(data.Apnea.mask((data.patient_asleep == 0) | (data.Apnea != 6)), 'r')
        ax1.plot(data.PhysioNet_labels.mask(data.PhysioNet_labels == 0) * 7, 'm', label='PhysioNet Labels')
        ax1.legend(loc='upper right')

        # Plot the effort analysis trace
        ax2.set_title('Effort Analysis Trace', fontdict=fontdict, pad=-1)
        ax2.plot(effort, 'y', label='Effort')
        ax2.plot(baseline.mask(data.patient_asleep == 0), 'k', label='Baseline')
        ax2.plot(pos_envelope.mask(data.patient_asleep == 0), 'b', label='Positive Envelope')

        # Plot the detected increased effort
        ax2.plot(data['RERA_increased_effort'].mask(data['RERA_increased_effort'] == 0) * 5, 'r', label='Increased Effort')
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    return data

def find_recovery_breaths(data: pd.DataFrame, Fs: float) -> pd.DataFrame:
    """
    Identifies recovery breaths based on ventilation spikes in the respiratory signal.

    Args:
    - data: DataFrame containing the 'Ventilation_combined' signal.
    - Fs: Sampling frequency of the signal.

    Returns:
    - Updated DataFrame with 'recovery_breaths' column indicating detected recovery breaths.
    """
    # Ensure Fs is an integer
    assert Fs == int(Fs), 'Provided "Fs" should be an integer.'
    Fs = int(Fs)
    
    # Extract the ventilation signal
    sig = data['Ventilation_combined'].values

    # Compute mean envelopes for the signal
    df = compute_envelope(sig, Fs, env_smooth=5)
    
    # Compute normalized positive and negative envelopes
    Q = df['pos_envelope'].mean() - df['neg_envelope'].mean()
    sig = detrend(sig) / Q
    
    # Smooth the signal to create surrogate ventilation signal
    df['pos_env'] = pd.Series(sig).rolling(5 * Fs, center=True, min_periods=1).quantile(0.95)
    df['neg_env'] = pd.Series(sig).rolling(5 * Fs, center=True, min_periods=1).quantile(0.05)
    
    # Compute a Ventilation surrogate trace
    Ventilation = df['pos_env'].values[::10] - df['neg_env'].values[::10]
    Ventilation[Ventilation < 0] = 0
    df['Ventilation_surrogate'] = np.repeat(Ventilation, 10)[:len(df)]

    # Identify potential arousal locations based on large ventilation spikes
    trailing_min = df['Ventilation_surrogate'].rolling(3 * Fs, min_periods=Fs).min()
    trailing_max = df['Ventilation_surrogate'].rolling(3 * Fs, min_periods=Fs).max()
    biggest_arousal = trailing_max - trailing_min
    biggest_arousal = biggest_arousal.rolling(10 * Fs, min_periods=Fs).max()

    df['recovery_breaths'] = 0
    for loc, _ in find_events(biggest_arousal > 0.25):
        # Skip arousal locations if the ventilation signal is descending
        if any(df.loc[loc - 2 * Fs: loc, 'Ventilation_surrogate'] > df.loc[loc, 'Ventilation_surrogate']):
            continue
        if df.loc[loc, 'Ventilation_surrogate'] <= 0:
            continue

        # Set location based on largest breathing amplitude
        peak = np.argmax(data.loc[loc - 2.5 * Fs: loc + 5 * Fs, 'Ventilation_combined'])
        df.loc[loc - 2.5 * Fs + peak, 'recovery_breaths'] = 1

    # Update the original 'data' DataFrame with the identified recovery breaths
    data['recovery_breaths'] = df['recovery_breaths']
    
    return data

def create_RERA_labels(data: pd.DataFrame, eeg_arousals: list, Fs: int) -> pd.DataFrame:
    """
    Creates RERA (Respiratory Effort-Related Arousals) labels based on morphology and flow reductions.

    Args:
    - data: DataFrame containing respiratory and EEG arousal data.
    - eeg_arousals: List of EEG arousal events (start, end pairs).
    - Fs: Sampling frequency.

    Returns:
    - Updated DataFrame with RERA labels.
    """

    # Smooth the RERA morphology output
    morphology_labels = data['RERA_morphology'].fillna(0).values.astype(int)
    data['RERA_morphology_smooth'] = smooth(morphology_labels, win=Fs, zeros=int(0.9 * Fs))
    min_dur = 5  # Minimum duration in seconds for valid RERA

    # Iterate over all EEG arousal events
    for start, ending in eeg_arousals:
        # Extend the search region to include 15 seconds before the arousal
        start = max(0, start - 15 * Fs)
        region = list(range(int(start), int(ending)))
        found = False

        # Check for morphology-related RERAs within the region
        morph_events = find_events(data.loc[region, 'RERA_morphology_smooth'].fillna(0) > 0)
        if morph_events:
            st = region[morph_events[0][0]]
            end = region[morph_events[-1][1] - 1]
            label_range = list(range(st, end))

            # Criteria: Recovery breath found and duration >= 10 seconds
            check1 = any(data.loc[st:end + 10 * Fs, 'recovery_breaths'] > 0)
            check2 = len(label_range) >= min_dur * Fs
            if check1 and check2:
                data.loc[label_range, 'algo_reras'] = 5
                data.loc[label_range, 'morph_RERAs'] = 6
                found = True

        # Look for flow reductions with arousal
        if any(data.loc[region, 'flow_reductions'] > 0):
            ev = find_events(data.loc[region, 'flow_reductions'] > 0)[0]
            st, end = region[0] + ev[0], region[0] + ev[1]

            # Skip if the event meets hypopnea/apnea criteria
            if any(data.loc[st:end, 'final_flow_reductions'] > 0):
                continue

            # Skip if no recovery breath is found
            if not any(data.loc[st:end + 10 * Fs, 'recovery_breaths'] > 0):
                continue

            # Save as a flow-limitation RERA
            data.loc[st:end, 'algo_reras'] = 5
            data.loc[st:end, 'flowlim_RERAs'] = 6

        # Remove RERA if it follows an apnea/hypopnea within 15 seconds
        if found and np.any(data.loc[label_range[0] - 15 * Fs:label_range[0], 'final_flow_reductions'] > 0):
            data.loc[label_range, ['algo_reras', 'morph_RERAs', 'effort_RERAs']] = 0
            continue

    return data

def smooth(y: np.ndarray, win: int = 10, zeros: int = 3) -> np.ndarray:
    """
    Smooth a binary array `y` by applying a sliding window of length `win`. 
    If more than `zeros` 0s are found in the window, replace the entire window with 0.
    Otherwise, assign the most frequent non-zero label to the window.
    
    Args:
    - y: Input binary array to smooth.
    - win: Window size for smoothing.
    - zeros: Maximum number of allowed 0s in the window before smoothing to all 0s.
    
    Returns:
    - y_smooth: Smoothed array.
    """

    # Create array of sliding windows
    seg_ids = np.arange(0, y.shape[0] - win + 1, 1)
    label_segs = np.array([y[x:x+win] for x in seg_ids])

    # Apply smoothing over all sliding windows
    for s, seg in enumerate(label_segs):
        lab, cn = np.unique(seg, return_counts=True)
        # If the window contains more than `zeros` 0's, make the entire window 0
        if lab[0] == 0 and cn[0] >= zeros:
            label_segs[s] = 0
        # Otherwise, assign the most frequent non-zero label
        else:
            label_segs[s] = lab[np.argmax(cn[1:], axis=0) + 1]  # +1 to skip 0s

    # Initialize smoothed output array
    y_smooth = np.zeros_like(y)
    y_smooth[:len(label_segs)] = label_segs[:, 0]

    # Determine the boundaries for shifts
    ys_binary = (y_smooth > 0).astype(int)
    y_diff = np.diff(np.concatenate([[0], ys_binary]))
    beg = np.where(y_diff > 0)[0]
    end = np.where(y_diff < 0)[0]

    # Ensure that the start and end points match correctly
    if len(beg) != len(end):
        if beg[0] > end[0]:
            beg = np.insert(beg, 0, 0)
        if beg[-1] > end[-1]:
            end = np.append(end, len(y_smooth) - 1)

    # Adjust the labels and apply the shift correction
    half_win = win // 2
    for start, stop in zip(beg, end):
        lab, cn = np.unique(y_smooth[start:stop], return_counts=True)
        most_frequent_label = lab[np.argmax(cn, axis=0)]
        
        # Apply the label to the entire region and shift correction
        y_smooth[start:stop] = most_frequent_label
        if stop + half_win < len(y_smooth):
            y_smooth[start + half_win:stop + half_win] = most_frequent_label
        y_smooth[start:start + half_win] = 0

    # Append remaining unsmoothed elements to handle edges
    y_smooth = np.concatenate([y_smooth, y[len(y_smooth):]])

    return y_smooth


    # RERA analysis functions:

def load_kernels(path: str) -> List[np.ndarray]:
    """
    Load kernel data from an HDF5 file.

    Args:
    - path: Path to the HDF5 file.

    Returns:
    - data: List of kernel arrays loaded from the file.
    """

    # Load the HDF5 file
    with h5py.File(path, 'r') as ff:
        # Extract and store kernels from all keys in the file
        data = [ff[sig][:] for sig in ff.keys()]

    return data

# Plotting
def plot_kernels(flat_kernels: List[np.ndarray], non_flat_kernels: List[np.ndarray]) -> None:
    """
    Plot a selection of flat and non-flat kernels.

    Args:
    - flat_kernels: List of flat kernels (arrays).
    - non_flat_kernels: List of non-flat kernels (arrays).
    """

    fontdict = {'fontsize': 8}
    kernel_types = [flat_kernels, non_flat_kernels]
    titles = ['Flat Kernels', 'Non-flat Kernels']

    # Loop through the flat and non-flat kernels for plotting
    for i, kernels in enumerate(kernel_types):
        fig = plt.figure(figsize=(9.5, 6))
        plt.suptitle(titles[i])

        # Plot each kernel in a 6x5 grid
        for n, kernel in enumerate(kernels[:30]):  # Show up to 30 kernels to avoid overcrowding
            ax = fig.add_subplot(6, 5, n + 1)
            ax.plot(kernel)
            ax.set_yticklabels([])
            ax.set_xticklabels([])

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to prevent overlap with title
        plt.show()

