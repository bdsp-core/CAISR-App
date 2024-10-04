import sys, os, argparse
import pandas as pd
from os.path import expanduser


sys.path.append('./preprocess/')
from prepare_data import process_file, process_files

""" Sleep Data Preprocessing Functions

`process_file(path_input, path_output)` processes a single sleep data file by loading the signal and corresponding annotations, preprocessing the data, and saving the output to an HDF5 (.h5) file.
**Parameters**:
- `path_input` (str): The base path to the input signal and annotation files. The function expects the signal file to be in `.edf` format and the annotation file in `.csv` format. For example, if `path_input` is `'data/sub-S001'`, the function will look for `'data/sub-S001.edf'` and `'data/sub-S001.csv'`.
- `path_output` (str): The path where the processed `.h5` file will be saved.


process_files(path_dir_input, path_dir_output) processes multiple sleep data files in a specified directory. It iterates over all .edf files in the input directory, applies the preprocessing steps using the process_file function, and saves the processed data to the output directory.

path_dir_input (str): The directory containing the .edf and .csv files to be processed.
path_dir_output (str): The directory where the processed .h5 files will be saved.

Note: The function is designed to work with the MGH BDSP dataset.
    However, the functions can be modified to work with other datasets as well, the main points are:
    - Load the signal and annotations
    - Preprocess the signal and annotations:
    -- Standardize channel names
    -- Resample signals and annotations to the target frequency (200 Hz)
    -- Scale EEG/EOG to V from uV if necessary
    - Save the processed data to a HDF5 file.
"""

def extract_run_parameters(param_csv: str):
    assert os.path.exists(param_csv), 'run parameter file is not found.'
    
    # load run parameters from .csv
    params = pd.read_csv(param_csv)
    autoscale_signals = params['autoscale_signals'].values[0]
    overwrite = params['overwrite'].values[0]
    
    return autoscale_signals, overwrite


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Preprocess sleep data.")
    parser.add_argument("--path_dir_input", type=str, default='./data/raw/', help="Folder with raw MGH/BDSP files.")
    parser.add_argument("--path_dir_output", type=str, default='./data/', help="Folder where the prepared .h5 files will be stored.")
    parser.add_argument("--verbose", type=bool, default=False, help="Display progress messages.")
    
    # Parse arguments
    args = parser.parse_args()
    path_dir_input = args.path_dir_input
    path_dir_output = args.path_dir_output
    verbose = args.verbose

    # Extract run parameters
    autoscale_signals, overwrite = extract_run_parameters(path_dir_output + 'run_parameters/preprocess.csv')

    # Process the files
    process_files(path_dir_input, path_dir_output, autoscale_signals=autoscale_signals, overwrite=overwrite, verbose=verbose)

