import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import time

def cvs2lab(path,data_fs=128,output_fs=2,label_name='prediction',str='Class'):
    label = pd.read_csv(path)
    start = label['start_idx'].to_numpy()
    end = label['end_idx'].to_numpy()
    pred = label[label_name].to_numpy()
    Label = label[label_name.replace('prob_','')].to_numpy()


    Prediction = pred.repeat((end[0]-start[0])/(data_fs/output_fs))
    Label = Label.repeat((end[0]-start[0])/(data_fs/output_fs))

    return Label,Prediction
