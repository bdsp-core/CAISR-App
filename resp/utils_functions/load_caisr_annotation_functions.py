import os
import numpy as np
import pandas as pd
import h5py


def assert_and_adjust_annotation_len(len_original_data: int, df_annotation: pd.DataFrame, fs_original: int, fs_annotation: int, task_name: str) -> pd.DataFrame:
    """
    Asserts and adjusts the length of the annotation data (output of CAISR) to match the length of the original data.

    Args:
        len_original_data (int): Length of the original data in samples.
        df_annotation (pd.DataFrame): Annotation data (CAISR).
        fs_original (int): Sampling frequency of the original data.
        fs_annotation (int): Sampling frequency of the annotation data (CAISR).
        task_name (str): Name of the task (column name to check in annotation data).

    Returns:
        pd.DataFrame: Adjusted annotation data with the same length as the original data.
    """

    # Check for required columns in the annotation data
    required_columns = ['start_idx', 'end_idx', task_name]
    columns_available = list(df_annotation.columns)
    assert all(col in columns_available for col in required_columns), \
        f"Missing columns in annotation data for task '{task_name}'. Expected: {required_columns}"

    # Ensure the original Fs is a multiple of the annotation Fs
    assert fs_original % fs_annotation == 0, \
        f"Fs of original data ({fs_original}) is not a multiple of annotation data ({fs_annotation})"
    
    fs_factor = int(fs_original / fs_annotation)

    # Check that the expected Fs matches the annotation file data
    first_start, first_end = df_annotation.iloc[0]['start_idx'], df_annotation.iloc[0]['end_idx']
    assert first_end - first_start == fs_original / fs_annotation, \
        f"Expected Fs of annotation ({fs_annotation}) with original input Fs ({fs_original}) does not match annotation file data, sample indices of first annotation: ({first_start}, {first_end})."

    # Ensure the length of annotation data matches the length of the original data
    len_annotation = len(df_annotation)
    len_annotation_expected = len_original_data / fs_factor
    assert abs(len_annotation - len_annotation_expected) <= fs_original, \
        f"Annotation length ({len_annotation / fs_annotation} seconds) does not match expected length of original data ({len_original_data / fs_original} seconds). Please check annotation file."

    # Upsample the annotation to match the original data frequency
    upsampled_data = np.repeat(df_annotation.values, fs_factor, axis=0)
    df_annotation = pd.DataFrame(upsampled_data, columns=df_annotation.columns).drop(columns=['start_idx', 'end_idx'])

    # Adjust the length of the annotation data to match the original data length
    if len(df_annotation) < len_original_data:
        diff = len_original_data - len(df_annotation)
        last_row = df_annotation.iloc[-1]
        add_rows = pd.DataFrame([last_row] * diff, columns=df_annotation.columns)
        df_annotation = pd.concat([df_annotation, add_rows], ignore_index=True)
    elif len(df_annotation) > len_original_data:
        df_annotation = df_annotation.iloc[:len_original_data].reset_index(drop=True)

    # Final check to ensure lengths match
    assert len(df_annotation) == len_original_data, \
        f"Annotation length after upsampling does not match original data length. Possible bug in assert_and_adjust_annotation_len() function."

    return df_annotation

def load_annotation_in_original_fs(task: str, path_prepared_file: str, path_caisr_output_csv: str, fs_original: int = 200, fs_annotation: int = None, verbose: bool = False) -> pd.DataFrame:
    """
    Load annotation data and ensure it matches the original data's sampling frequency.

    Args:
        task (str): The task name to look for in the annotation data.
        path_prepared_file (str): Path to the prepared file.
        path_caisr_output_csv (str): Path to the CAISR output CSV file.
        fs_original (int): Sampling frequency of the original data (default: 200Hz).
        fs_annotation (int, optional): Sampling frequency of the annotation data. If not provided, inferred from the file.
        verbose (bool): Whether to print additional details (default: False).

    Returns:
        pd.DataFrame: Loaded and adjusted annotation data.
    """

    # Load CAISR annotation data
    df_annotation = pd.read_csv(path_caisr_output_csv)
    
    # Load original data length from prepared file (example using h5py for .h5 file)
    with h5py.File(path_prepared_file, 'r') as f:
        len_original_data = len(f['signals']['c4-m1'])

    if verbose:
        print(f"fs_original: {fs_original}, len_original_data: {len_original_data}")

    # load the annotation data
    df_annotation = pd.read_csv(path_caisr_output_csv)
    if verbose:
        print(f"CAISR output annotation csv file loaded, saved in 1 Hz for stage, resp, limb; in 2 Hz for arousal. N samples = {len(df_annotation)}")
    if fs_annotation is None:
        if task in ['stage', 'resp', 'limb']:
            fs_annotation = 1
        elif task == 'arousal':
            fs_annotation = 2
        if verbose: 
            print(f"fs_annotation automatically set for task {task}: {fs_annotation} Hz.")

    # Adjust annotation length to match original data length
    df_annotation_adjusted = assert_and_adjust_annotation_len(len_original_data, df_annotation, fs_original, fs_annotation, task)

    if verbose:
        print(f"Loaded and adjusted annotation data for task '{task}' with shape {df_annotation_adjusted.shape}")

    return df_annotation_adjusted

