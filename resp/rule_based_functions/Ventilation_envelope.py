import numpy as np
import pandas as pd
from typing import List, Tuple
from scipy.signal import find_peaks, savgol_filter

# Ventilation Analysis Functions

def compute_ventilation_envelopes(data: pd.DataFrame, Fs: int, channels: List[str] = ['Ventilation_combined']) -> pd.DataFrame:
    """
    Computes the positive and negative envelopes, baselines, and corrections for ventilation data.

    Args:
    - data: A pandas DataFrame containing ventilation data.
    - Fs: Sampling frequency in Hz.
    - channels: List of column names from data to sum for combined ventilation analysis.

    Returns:
    - Updated DataFrame with calculated envelopes and baselines.
    """
    
    if len(channels) == 1:
        new_df = compute_envelope(data[channels[0]].values, Fs)
    else:
        new_df = compute_envelope(data[channels].sum(axis=1).values, Fs)
    
    data['Ventilation_pos_envelope'] = new_df['pos_envelope'].values
    data['Ventilation_neg_envelope'] = new_df['neg_envelope'].values
    data['Ventilation_default_baseline'] = new_df['baseline'].values
    data['Ventilation_baseline'] = new_df['correction_baseline'].values
    data['baseline2'] = new_df['baseline2'].values
    
    return data

def compute_envelope(signal: np.ndarray, Fs: int, base_win: int = 30, env_smooth: int = 5) -> pd.DataFrame:
    """
    Computes the positive and negative envelopes of a signal, along with baselines.

    Args:
    - signal: 1D array-like signal data.
    - Fs: Sampling frequency in Hz.
    - base_win: Window size for baseline calculation in seconds.
    - env_smooth: Smoothing window size for the envelope in seconds.

    Returns:
    - A DataFrame containing positive/negative envelopes and baselines.
    """
    
    new_df = pd.DataFrame()
    new_df['x'] = np.squeeze(signal)

    # Find positive and negative peaks
    pos_peaks, _ = find_peaks(new_df['x'], distance=int(Fs * 1.5), width=int(0.4 * Fs), rel_height=1)
    neg_peaks, _ = find_peaks(-new_df['x'], distance=int(Fs * 1.5), width=int(0.4 * Fs), rel_height=1)

    new_df['pos_envelope'] = new_df['x'].iloc[pos_peaks].reindex_like(new_df).interpolate(method='cubic', limit_area='inside')
    new_df['neg_envelope'] = new_df['x'].iloc[neg_peaks].reindex_like(new_df).interpolate(method='cubic', limit_area='inside')

    # Smooth envelopes
    new_df['pos_envelope'] = new_df['pos_envelope'].rolling(env_smooth * Fs, center=True).median()
    new_df['neg_envelope'] = new_df['neg_envelope'].rolling(env_smooth * Fs, center=True).median()

    # Ensure pos_envelope >= neg_envelope
    check_invalids = new_df['pos_envelope'] < new_df['neg_envelope']
    new_df.loc[check_invalids, ['pos_envelope', 'neg_envelope']] = 0

    # Compute baselines
    new_df['baseline'], new_df['baseline2'], new_df['correction_baseline'] = compute_baseline(new_df, Fs, base_win)

    return new_df

def compute_baseline(new_df: pd.DataFrame, Fs: int, base_win: int, correction_ratio: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Computes the baseline, baseline2, and corrected baseline of a signal.

    Args:
    - new_df: DataFrame containing the positive and negative envelopes of a signal.
    - Fs: Sampling frequency in Hz.
    - base_win: Window size for baseline calculation in seconds.
    - correction_ratio: Ratio used for baseline correction.

    Returns:
    - A tuple containing the baseline1, baseline2, and correction baseline.
    """
    
    pos = new_df['pos_envelope'].rolling(base_win * Fs, center=True).mean()
    neg = new_df['neg_envelope'].rolling(base_win * Fs, center=True).mean()
    base = (pos + neg) / 2

    base1 = new_df['x'].rolling(base_win * Fs, center=True).median().rolling(base_win * Fs, center=True).mean()
    base2 = base.rolling(base_win * Fs, center=True).mean()

    base_corr = (correction_ratio * base1 + base2) / (1 + correction_ratio)
    base_corr = base_corr.rolling(base_win * Fs, center=True).mean()

    return base1, base2, base_corr

def compute_smooth_envelope(data: pd.DataFrame, region: List[int]) -> pd.DataFrame:
    """
    Applies Savitzky-Golay smoothing to positive and negative ventilation envelopes.

    Args:
    - data: A pandas DataFrame containing the ventilation data.
    - region: The region of the DataFrame to apply smoothing to.

    Returns:
    - Updated DataFrame with smoothed envelopes.
    """
    
    for env_tag in ['Smooth_pos_envelope', 'Smooth_neg_envelope']:
        original_env = env_tag.replace('Smooth', 'Ventilation')
        data.loc[region, env_tag] = savgol_filter(data.loc[region, original_env], 51, 1)
    
    return data
