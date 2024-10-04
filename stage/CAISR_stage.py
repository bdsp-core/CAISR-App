import argparse
import os
import numpy as np
import pandas as pd
import sys
import h5py
from scipy.signal import resample_poly
import warnings
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from collections import Counter

# Import your custom modules here
sys.path.insert(1, './stage/graphsleepnet')
from DE_PSD import *
from gcn_features import *
sys.path.insert(1, './stage')
from utils_docker import *
from ProductGraphSleepNet import *
from utils_model import *

def process_graph_features(input_path, save_path, model_path):

    
    model_type = 'graphsleepnet'
    optimizer = 'adam'
    learn_rate = 0.0001
    lr_decay = 0.0
    l1, l2 = 0.001, 0.001

    # Initialize the optimizer
    if optimizer == "adam":
        opt = keras.optimizers.Adam(lr=learn_rate, decay=lr_decay, clipnorm=1)
    else:
        assert False, 'Config: check optimizer'

    regularizer = keras.regularizers.l1_l2(l1=l1, l2=l2)

    w, h, context = 7, 9, 7
    sample_shape = (context, w, h)
    conf_adj = 'GL'
    GLalpha = 0.0
    num_of_chev_filters = 128
    num_of_time_filters = 128
    time_conv_strides = 1
    time_conv_kernel = 3
    num_block = 1
    cheb_k = 3
    cheb_polynomials = None  
    dropout = 0.60
    GRU_Cell = 256
    attn_heads = 40

    # Setup paths
    input_path = input_path 
    files = [f for f in os.listdir(input_path) if f.endswith('.h5')]
    # graph_out_paths = set_out_paths(save_path, dataset)
    # files_is_already_extracted = os.listdir(graph_out_paths[0])
    # dataset_folder = dataset

    # Load the model
    model = build_ProductGraphSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials,
                                       time_conv_kernel, sample_shape, num_block, opt, conf_adj == 'GL', GLalpha, regularizer,
                                       GRU_Cell, attn_heads, dropout)

    model.load_weights(model_path + 'weights_fold_3.h5')
    save_path = save_path + 'stage/'
    os.makedirs(save_path, exist_ok=True)
    
    for num, each_id in enumerate(files):
        path = os.path.join(input_path, each_id)
        the_id = each_id[:-3]
        # print(f'\n==========================\n(# {num + 1}/{len(files)}) {dataset} {each_id}')
        # out_paths = [p + each_id for p in graph_out_paths]
        try:
            signals, Fs, sig_tages, length_data = select_signals_cohort(path)
        except Exception as e:
            print(e)
            continue

        segs = segment_data_unseen(signals)
        window = 30

        try:
            MYpsd, MYde = graph_feat_extraction_unseen_docker(segs, sig_tages, Fs, window)
            image = AddContext(MYde, context)
            image = np.squeeze(np.array(image))

            prediction = model.predict(image)
            pred_per_subject = prediction.argmax(axis=1) + 1

            fs_input = 200
            stage_length = 30

            pred_per_subject = np.concatenate([[np.nan] * 3, pred_per_subject, [np.nan] * 3])
            pred_per_subject = np.repeat(pred_per_subject, stage_length, axis=0)

            nan_row = np.empty((1, prediction.shape[1]))
            nan_row[:] = np.nan
            indices = [0, 1, 2]
            for index in indices:
                prediction = np.insert(prediction, index, nan_row, axis=0)
            for i in range(3):
                prediction = np.vstack((prediction, nan_row))
            prediction = np.repeat(prediction, stage_length, axis=0)

            t1 = (np.arange(len(prediction)) * fs_input).astype(int)
            t2 = (np.arange(len(prediction)) + 1) * fs_input
            df = pd.DataFrame({'start_idx': t1, 'end_idx': t2, 'stage': pred_per_subject,
                               'prob_n3': prediction[:, 0], 'prob_n2': prediction[:, 1],
                               'prob_n1': prediction[:, 2], 'prob_r': prediction[:, 3],
                               'prob_w': prediction[:, 4]})

            t1 = (np.arange(length_data/ fs_input) * fs_input).astype(int)
            t2 = ((np.arange(length_data / fs_input) + 1) * fs_input).astype(int)
            df_matched = pd.DataFrame({'start_idx': t1, 'end_idx': t2, 'stage': np.nan,
                                       'prob_n3': np.nan, 'prob_n2': np.nan,
                                       'prob_n1': np.nan, 'prob_r': np.nan,
                                       'prob_w': np.nan})
            df_matched.iloc[0:len(df)] = df
            df_matched.to_csv(os.path.join(save_path, the_id + '_stage.csv'), index=False)
            import pdb; pdb.set_trace()
        except Exception as error:
            print(f'({num}) Failure feature extraction GRAPH: {error}')
    print("Graph features processed!")

# def main():
#     parser = argparse.ArgumentParser(description="Process graph features for sleep staging.")
#     parser.add_argument("input_path", type=str, default='./data/', help="Path to the input data folder")
#     parser.add_argument("save_path", type=str, default='./caisr_output/', help="Path to save the output features")
#     parser.add_argument("model_path", type=str, default='./stage/models/', help="Path to the pre-trained model")
#     # parser.add_argument("dataset", type=str, help="Path to the pre-trained model")

#     args = parser.parse_args()
#     process_graph_features(args.input_path, args.save_path, args.model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process graph features for sleep staging.")
    parser.add_argument("--input_path", type=str, default='./data/', help="Path to the input data folder")
    parser.add_argument("--save_path", type=str, default='./caisr_output/', help="Path to save the output features")
    parser.add_argument("--model_path", type=str, default='./stage/models/', help="Path to the pre-trained model")
    # parser.add_argument("dataset", type=str, help="Path to the pre-trained model")

    args = parser.parse_args()
    process_graph_features(args.input_path, args.save_path, args.model_path)

