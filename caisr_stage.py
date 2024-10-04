import sys, os, time, glob, warnings, h5py, argparse, logging
import numpy as np
import pandas as pd
from scipy.signal import resample_poly
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from collections import Counter
from typing import List, Tuple
tf.get_logger().setLevel(logging.ERROR)

# Import custom modules
sys.path.insert(1, './stage/graphsleepnet')
from DE_PSD import *
from gcn_features import *
sys.path.insert(1, './stage')
from utils_docker import *
from ProductGraphSleepNet import *
from utils_model import *


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
    csv_paths = [f'{csv_folder}stage/{ID}_stage.csv' for ID in IDs]

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

def CAISR_stage(in_paths: List[str], save_paths: List[str], model_path: str) -> None:
    """
    Main function for processing sleep stage prediction using a trained model.

    Parameters:
    - input_path (str): List of paths where the signals are stored.
    - save_path (str): List of strings where the results will be saved.
    - model_path (str): Path to the pre-trained model weights.
    """
    # Log the start of the process
    timer('* Starting "caisr_stage" (created by Samaneh Nasiri, PhD)')
    
    # Model and training settings
    model_type = 'graphsleepnet'    # Type of model
    optimizer = 'adam'              # Optimizer type
    learn_rate = 0.0001             # Learning rate
    lr_decay = 0.0                  # Learning rate decay
    l1, l2 = 0.001, 0.001           # L1 and L2 regularization

    # Initialize the optimizer based on the specified type
    if optimizer == "adam":
        opt = keras.optimizers.Adam(lr=learn_rate, decay=lr_decay, clipnorm=1)
    else:
        raise ValueError('Config: check optimizer')

    # Regularizer for the model
    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)

    # Model parameters
    w, h, context = 7, 9, 7         # Window size, height, and context for input
    sample_shape = (context, w, h)  # Shape of the input data
    conf_adj = 'GL'                 # Graph Laplacian configuration
    GLalpha = 0.0                   # Alpha value for the GL configuration
    num_of_chev_filters = 128       # Number of Chebyshev filters
    num_of_time_filters = 128       # Number of time convolution filters
    time_conv_strides = 1           # Stride for time convolution
    time_conv_kernel = 3            # Kernel size for time convolution
    num_block = 1                   # Number of blocks in the model
    cheb_k = 3                      # Chebyshev polynomial degree
    cheb_polynomials = None         # Placeholder for Chebyshev polynomials
    dropout = 0.60                  # Dropout rate
    GRU_Cell = 256                  # Number of GRU cells
    attn_heads = 40                 # Number of attention heads

    # Build the model using the predefined parameters
    model = build_ProductGraphSleepNet(
        cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
        time_conv_kernel, sample_shape, num_block, opt, conf_adj == 'GL', GLalpha, regularizer,
        GRU_Cell, attn_heads, dropout
    )

    # Load pre-trained model weights
    model.load_weights(os.path.join(model_path, 'weights_fold_3.h5'))
    
    # Run over all files
    for num, (path, save_path) in enumerate(zip(in_paths, save_paths)):
        each_id = path.split('/')[-1]
        the_id = each_id[:-3]
        tag = the_id if len(the_id)<21 else the_id[:20]+'..'
        print(f'(# {num + 1}/{len(in_paths)}) Processing "{tag}"')
        
        # Attempt to load the signals for each file
        try:
            signals, Fs, sig_tages, length_data = select_signals_cohort(path)
        except Exception as e:
            print(f'Error loading signals: {e}')
            continue

        # Segment the signals for unseen data
        segs = segment_data_unseen(signals)
        window = 30  # Window size for feature extraction

        try:
            # Perform feature extraction
            MYpsd, MYde = graph_feat_extraction_unseen_docker(segs, sig_tages, Fs, window)
            image = AddContext(MYde, context)  # Add context to the extracted features
            image = np.squeeze(np.array(image))  # Squeeze array dimensions

            # Model prediction
            prediction = model.predict(image)
            pred_per_subject = prediction.argmax(axis=1) + 1  # Predicted sleep stages

            # Set input frequency and stage length
            fs_input = 200
            stage_length = 30

            # Handle padding of predictions for smooth output
            pred_per_subject = np.concatenate([[np.nan] * 3, pred_per_subject, [np.nan] * 3])
            pred_per_subject = np.repeat(pred_per_subject, stage_length, axis=0)

            # Pad the prediction probabilities with NaNs
            nan_row = np.empty((1, prediction.shape[1]))
            nan_row[:] = np.nan
            indices = [0, 1, 2]
            for index in indices:
                prediction = np.insert(prediction, index, nan_row, axis=0)
            for i in range(3):
                prediction = np.vstack((prediction, nan_row))
            prediction = np.repeat(prediction, stage_length, axis=0)

            # Prepare the dataframe for the output CSV
            t1 = (np.arange(len(prediction)) * fs_input).astype(int)
            t2 = (np.arange(len(prediction)) + 1) * fs_input
            df = pd.DataFrame({
                'start_idx': t1, 'end_idx': t2, 'stage': pred_per_subject,
                'prob_n3': prediction[:, 0], 'prob_n2': prediction[:, 1],
                'prob_n1': prediction[:, 2], 'prob_r': prediction[:, 3],
                'prob_w': prediction[:, 4]
            })

            # Match with the signal data length
            t1 = (np.arange(length_data/ fs_input) * fs_input).astype(int)
            t2 = ((np.arange(length_data / fs_input) + 1) * fs_input).astype(int)
            df_matched = pd.DataFrame({
                'start_idx': t1, 'end_idx': t2, 'stage': np.nan,
                'prob_n3': np.nan, 'prob_n2': np.nan,
                'prob_n1': np.nan, 'prob_r': np.nan,
                'prob_w': np.nan
            })
            df_matched.iloc[0:len(df)] = df  # Replace NaN rows with actual data

            # Save the results to a CSV file
            df_matched.to_csv(save_path, index=False)

        except Exception as error:
            print(f'({num}) Failure during feature extraction: {error}')
    
    # Log the end of the process
    timer('* Finishing "caisr_stage"')


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run CAISR sleep staging.")
    parser.add_argument("--input_path", type=str, default='./data/', help="Path to the input data folder")
    parser.add_argument("--save_path", type=str, default='./caisr_output/intermediate/', help="Path to save the output features")
    parser.add_argument("--model_path", type=str, default='./stage/models/', help="Path to the pre-trained model")

    # Parse the arguments
    args = parser.parse_args()

    # Extract run parameters
    input_files = glob.glob(args.input_path + '*.h5')
    overwrite = extract_run_parameters(args.input_path + 'run_parameters/stage.csv')

    # Set output paths
    in_paths, save_paths = set_output_paths(input_files, args.save_path, overwrite)
    
    # Run CAISR stage
    CAISR_stage(in_paths, save_paths, args.model_path)

