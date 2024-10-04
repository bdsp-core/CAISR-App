import os, h5py, datetime
import numpy as np
import pandas as pd


# Data saving functions

def write_to_hdf5_file(df: pd.DataFrame, output_h5_path: str, hdr: dict = {}, 
                       default_dtype: str = 'float32', overwrite: bool = False) -> None:
    """
    Saves data from a DataFrame into an HDF5 file, including metadata in the header.

    Args:
        df (pd.DataFrame): DataFrame containing signals to save.
        output_h5_path (str): Output path for the HDF5 file.
        hdr (dict, optional): Header information to save as metadata.
        default_dtype (str, optional): Default data type for numerical values.
        overwrite (bool, optional): Overwrite the existing file if True.
    """

    chunk_size = 64
    output_h5_path = output_h5_path if output_h5_path.endswith('.hf5') else output_h5_path + '.hf5'

    if overwrite and os.path.exists(output_h5_path):
        os.remove(output_h5_path)

    with h5py.File(output_h5_path, 'a') as f:
        # Save signals from DataFrame
        for signal in df.columns:
            if signal not in f:
                dtype1 = default_dtype
                if signal.lower() in ['annotation', 'test_type', 'rec_type', 'patient_tag', 'dataset']:
                    dtype1 = h5py.string_dtype(encoding='utf-8')  # Save string data
                    dset_signal = f.create_dataset(signal, shape=(df.shape[0],), maxshape=(None,), 
                                                   chunks=(chunk_size,), dtype=dtype1)
                    dset_signal[:] = df[signal].astype(str)
                elif signal.lower() in ['stage', 'apnea', 'Fs', 'newFs', 'cpap_start']:
                    dtype1 = 'int32'  
                    df.loc[pd.isna(df[signal]), signal] = -1
                else:
                    dset_signal = f.create_dataset(signal, shape=(df.shape[0],), maxshape=(None,),
                                                   chunks=(chunk_size,), dtype=dtype1)
                    dset_signal[:] = df[signal].astype(dtype1)
            else:
                raise ValueError(f'Signal "{signal}" already exists in file and overwrite is not allowed.')

        # Save header metadata
        if hdr:
            for key, value in hdr.items():
                if value is None:
                    value = str(value)
                if isinstance(value, (datetime.datetime, pd.Timestamp)):
                    value = np.array([value.year, value.month, value.day, value.hour, value.minute, value.second, value.microsecond])
                if isinstance(value, (int, np.int32)):
                    f.create_dataset(key, shape=(1,), maxshape=(1,), chunks=True, dtype=np.int32)[:] = np.int32(value)
                elif isinstance(value, np.ndarray):
                    f.create_dataset(key, shape=value.shape, maxshape=(value.shape[0]+10,), chunks=True, dtype=np.int32)[:] = value.astype(np.int32)
                elif isinstance(value, str):
                    dtype_str = np.array([value + ' ' * (44-len(value))]).astype('<S44').dtype
                    f.create_dataset(key, shape=(1,), maxshape=(None,), chunks=True, dtype=dtype_str)[:] = value.encode('utf8')
                else:
                    raise ValueError(f'Unexpected datatype for header entry "{key}".')


