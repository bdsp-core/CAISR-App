#from tkinter import OUTSIDE
from keras import backend as K
import tensorflow as tf
import numpy as np


def cat_acc(y_true,y_pred):
    loc = tf.where(y_true[:,:,-1]!=2)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

def ar_acc(y_true,y_pred):
    loc = tf.where(y_true[:,:,1]==1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

def bg_acc(y_true,y_pred):
    loc = tf.where(y_true[:,:,0]==1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    
    return tf.keras.metrics.categorical_accuracy(y_true, y_pred)

def DiceLoss(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return 1 - dice

def Dice(targets, inputs, smooth=1e-6):
    
    #flatten label and prediction tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice

def Dice_ma(y_true, y_pred, smooth=1e-7):
    y_true = y_true[:,:,:3]
    y_pred = y_pred[:,:,:2]

    loc = tf.where(y_true[:,:,-1]!=1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    y_true = y_true[:,:-1]
    inputs = K.flatten(y_pred)
    targets = K.flatten(y_true)
    
    intersection = K.sum(K.dot(targets, inputs))
    dice = (2*intersection + smooth) / (K.sum(targets) + K.sum(inputs) + smooth)
    return dice

#dice coef 
def dc(y_true, y_pred, smooth=1e-7):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

#dice coef cut
def dc_cut(y_true, y_pred, smooth=1e-7):
    if y_true.shape[-1]<5:
        y_true = y_true[:,:,:2]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

#dice coef mask aux
def dc_ma(y_true, y_pred, smooth=1e-7):
    y_true = y_true[:,:,:3]
    y_pred = y_pred[:,:,:2]

    loc = tf.where(y_true[:,:,-1]!=1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    y_true = y_true[:,:-1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))

#dice coef mask aux
def dc_ma_l(y_true, y_pred, smooth=1e-7):
    y_true = y_true[:,:,:3]
    y_pred = y_pred[:,:,:2]

    loc = tf.where(y_true[:,:,-1]!=1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    y_true = y_true[:,:-1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return 1-K.mean((2. * intersect / (denom + smooth)))

#dice coef mask
def dc_m(y_true, y_pred, smooth=1e-7):
    loc = tf.where(y_true[:,:,-1]!=1)
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    y_true = y_true[:,:-1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2. * intersect / (denom + smooth)))



#focal loss
def fl(y_true, y_pred, gamma=2.0, alpha=0.25):

    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate cross entropy
    cross_entropy = -y_true*K.log(y_pred)
    # Calculate weight that consists of  modulating factor and weighting factor
    weight = alpha * K.pow((1-y_pred), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=1)
    return loss

#crossentropy cut
def ct_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
    out=K.mean(out)
    return out


#crossentropy 
def ct_keras(y_true,y_pred):
    return tf.keras.metrics.categorical_crossentropy(y_true, y_pred)

#weighted crossentropy cut
def wc_ct_cut(y_true,y_pred):
    n=3
    w0=1/n
    w1=(n-1)/n
    l0=ct_cut(y_true[:,:,0],y_pred[:,:,0]) # non-arousal
    l1=ct_cut(y_true[:,:,1],y_pred[:,:,1]) # arousal
    out = (w0 * l0 + w1 * l1)/(w0+w1)  # set custom weights for each class
    return out


#weighted focal
def wf(y_true,y_pred):
    n=3
    w0=1/n
    w1=(n-1)/n

    l0=fl(y_true[:,:,0],y_pred[:,:,0])  # non-aorusal
    l1=fl(y_true[:,:,1],y_pred[:,:,1]) # arousal

    out = (w0 * l0 + w1 * l1)/(w0+w1)  # set custom weights for each class
    return out

def wf_resp(y_true,y_pred):
    w0 = 0.7*1.5
    w1 = 5.3
    w2 = 5.8
    w3 = 83.2
    w4 = 1.5*3
    w5 = 3.3

    l0=fl(y_true[:,:,0],y_pred[:,:,0])
    l1=fl(y_true[:,:,1],y_pred[:,:,1])
    l2=fl(y_true[:,:,2],y_pred[:,:,2])  
    l3=fl(y_true[:,:,3],y_pred[:,:,3])
    l4=fl(y_true[:,:,4],y_pred[:,:,4]) 
    l5=fl(y_true[:,:,5],y_pred[:,:,5])

    out = (w0 * l0 + w1 * l1+w2 * l2 + w3 * l3 + w4 * l4 + w5 * l5)/(w0+w1+w2+w3+w4+w5)  # set custom weights for each class
    return out

def grab_metric(str):
    if str == 'cat_acc':
        return cat_acc
    if str == 'bg_acc':
        return bg_acc
    if str == 'ar_acc':
        return ar_acc
    if str == 'dc' or str == 'dice_coef':
        return dc
    if str == 'dc_cut' or str =='dice_coef_cut':
        return dc_cut
    if str == 'dc_m' or str =='dice_coef_mask':
        return dc_m
    if str == 'dc_ma' or str =='dice_coef_mask_aux':
        return dc_ma
    if str == 'dc_ma_l' or str =='dice_coef_mask_aux_loss':
        return dc_ma_l
    if str == 'Dice_ma':
        return Dice_ma
    if str == 'focal_loss' or str =='fl':
        return fl
    if str == 'crossentropy_cut' or str =='ct_cut':
        return ct_cut
    if str == 'crossentropy_keras':
        return ct_keras
    if str == 'weighted_categorical_ct_cut' or str=='wc':
        return wc_ct_cut
    if str == 'weighted_focal' or str =='wf':
        return wf
    if str == 'weighted_focal_resp' or str =='wf_resp':
        return wf_resp
   
   









