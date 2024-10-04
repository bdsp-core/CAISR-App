import sys
import numpy as np

from tensorflow import keras
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras import models
from tensorflow.keras.layers import Layer, Bidirectional, LSTM, Reshape, GRU, Dropout, BatchNormalization
# from tensorflow.keras.layers.core import Lambda
from graph_attention_layer import GraphWiseAttentionNetwork

# from get_mad_dbx import *
# _, dbx_pfx, _, _ = get_mad_dbx()
# sys.path.insert(1, dbx_pfx + 'CAISR_Codes/main_CAISR_functions/')
# from keras_model_evaluation_metrics import get_evaluation_metrics

# Model input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
# 
#     V: num_of_vertices
#     T: num_of_timesteps
#     F: num_of_features
#
# Model output: (*, 5)
# 
#     5: 5 sleep stages

#%%
class SpatialAttention(Layer):
    '''
    compute spatial attention scores
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, **kwargs):
        super(SpatialAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.W_1 = self.add_weight(name='W_1',
                                      shape=(num_of_timesteps, 1),
                                      initializer='uniform',
                                      trainable=True)
        self.W_2 = self.add_weight(name='W_2',
                                      shape=(num_of_features, num_of_timesteps),
                                      initializer='uniform',
                                      trainable=True)
        self.W_3 = self.add_weight(name='W_3',
                                      shape=(num_of_features, ),
                                      initializer='uniform',
                                      trainable=True)
        self.b_s = self.add_weight(name='b_s',
                                      shape=(1, num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        self.V_s = self.add_weight(name='V_s',
                                      shape=(num_of_vertices, num_of_vertices),
                                      initializer='uniform',
                                      trainable=True)
        super(SpatialAttention, self).build(input_shape)

    def call(self, x):
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        
        # shape of lhs is (batch_size, V, T)
        lhs=K.dot(tf.transpose(x,perm=[0,2,3,1]), self.W_1)
        lhs=tf.reshape(lhs,[tf.shape(x)[0],num_of_vertices,num_of_features])
        lhs = K.dot(lhs, self.W_2)
        
        # shape of rhs is (batch_size, T, V)
        rhs = K.dot(self.W_3, tf.transpose(x,perm=[1,0,3,2])) # K.dot((F),(T,batch_size,F,V))=(T,batch_size,V)
        rhs=tf.transpose(rhs,perm=[1,0,2]) # (batch_size, T, V)
        
        # shape of product is (batch_size, V, V)
        product = K.batch_dot(lhs, rhs)
        
        S = tf.transpose(K.dot(self.V_s, tf.transpose(K.sigmoid(product + self.b_s),perm=[1, 2, 0])),perm=[2, 0, 1])
        
        # normalization
        S = S - K.max(S, axis = 1, keepdims = True)
        exp = K.exp(S)
        S_normalized = exp / K.sum(exp, axis = 1, keepdims = True)
        return S_normalized

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[2],input_shape[2])

#%%
def diff_loss(diff, S):
    '''
    compute the differential (smoothness term) loss for Spatial/Temporal graph learning
    '''
    diff_loss_value = 0
    F = diff.shape[1]
    for i in range(int(F)):
        diff_loss_value = diff_loss_value + K.sum(K.sum(diff[:,i]**2,axis=3)*S)
    
    return diff_loss_value


def crossentropy_cut(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    y_pred_f= tf.clip_by_value(y_pred_f, 1e-7, (1. - 1e-7))
    mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    out = -(y_true_f * K.log(y_pred_f)*mask + (1.0 - y_true_f) * K.log(1.0 - y_pred_f)*mask)
    out=K.mean(out)
    return out

def weighted_categorical_crossentropy_cut(y_true,y_pred):
    w0 = 1/545702  
    w1 = 1/14457

    # w0 = 1
    # w1 = 100
    l0=crossentropy_cut(y_true[:,0],y_pred[:,0]) # non-arousal
    l1=crossentropy_cut(y_true[:,1],y_pred[:,1]) # arousal
    out = (w0 * l0 + w1 * l1)/(w0+w1)  # set custom weights for each class
    return out
    
def dice_coef(y_true, y_pred, smooth=1e-7):
    # y_true = y_true[:,:,:-1]  # remove "masked" in the validation dice coeff
    # y_pred = y_pred[:,:,:-1]
    # import pdb; pdb.set_trace()
    # y_true = y_true[:,:-1]
    # print(y_true.shape)
    # print(y_pred.shape)
    # y_true = y_true[:,0:2]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    # mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    # intersection = K.sum(y_true_f * y_pred_f * mask)
    return K.mean((2. * intersect / (denom + smooth)))

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    # Define epsilon so that the backpropagation will not result in NaN
    # for 0 divisor case
    # y_true_f = K.flatten(y_true)
    # y_pred_f = K.flatten(y_pred)
    epsilon = K.epsilon()
    # Add the epsilon to prediction value
    #y_pred = y_pred + epsilon
    # Clip the prediction value
    y_pred = K.clip(y_pred, epsilon, 1.0-epsilon)
    # Calculate cross entropy
    cross_entropy = -y_true*K.log(y_pred)
    # Calculate weight that consists of  modulating factor and weighting factor
    weight = alpha * K.pow((1-y_pred), gamma)
    # Calculate focal loss
    loss = weight * cross_entropy
    # Sum the losses in mini_batch
    loss = K.sum(loss, axis=-1)
    return loss

def dice_coef_mask(y_true, y_pred, smooth=1e-7):
    loc = tf.where(tf.not_equal(y_true[:,2],1))
    y_true = tf.gather_nd(y_true,indices=loc)
    y_pred = tf.gather_nd(y_pred,indices=loc)
    y_true = y_true[:,0:2]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    # mask=K.cast(K.greater_equal(y_true_f,-0.5),dtype='float32')
    # intersection = K.sum(y_true_f * y_pred_f * mask)
    return K.mean((2. * intersect / (denom + smooth)))

def weighted_focal_loss(y_true,y_pred):
    w0 = 1 #sample for class 0)
    w1 = 19 # sample for class 1
    l0=focal_loss(y_true[:,0],y_pred[:,0]) # non-arousal
    l1=focal_loss(y_true[:,1],y_pred[:,1]) # arousal
    # l2 = focal_loss(y_true[:,2],y_pred[:,2]) # arousal
    out = (w0 * l0 + w1 * l1)/(w0+w1)  # set custom weights for each class
    return out
#%%
def F_norm_loss(S, Falpha):
    '''
    compute the Frobenious norm loss cheb_polynomials
    '''
    if len(S.shape)==3:
        # batch input
        return Falpha * K.sum(K.mean(S**2,axis=0))
    else:
        return Falpha * K.sum(S**2)

#%%
class Graph_Learn_Spatial(Layer):
    '''
    Spatial graph structure learning
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_vertices, num_of_vertices)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S_Spatial = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff_Spatial = tf.convert_to_tensor([[[[[0.0]]]]])  # similar to placeholder
        super(Graph_Learn_Spatial, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a_Spatial = self.add_weight(name='a_Spatial',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(Graph_Learn_Spatial, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
                        
        # Spatial Graph Learning:
        for ff in range(int(F)):
            x_Spatial_ff = tf.transpose(x[:, :, :, ff], perm=[0, 2, 1]) #(N, V, T)

            diff_Spatial_temp = K.abs(tf.transpose(tf.transpose(tf.broadcast_to(x_Spatial_ff, [V,N,V,T]), perm=[2,1,0,3])
            - x_Spatial_ff, perm=[1,0,2,3])) #(N, V, V, T)
            
            diff_Spatial_temp = K.expand_dims(diff_Spatial_temp, axis=1) #(N, 1, V, V, T)

            if ff == 0:
                diff_Spatial = diff_Spatial_temp
            else:
                diff_Spatial = K.concatenate((diff_Spatial, diff_Spatial_temp), axis=1) #(N, F, V, V, T)
                
        tmpS = K.exp(K.relu(K.reshape(K.dot(tf.reduce_mean(tf.transpose(diff_Spatial, perm=[0, 4, 2, 3, 1]), axis=1), self.a_Spatial), [N,V,V]))) #(N, V, V)
        
        # normalization
        S_Spatial = tmpS / K.sum(tmpS, axis=1, keepdims=True)
        
        self.diff_Spatial = diff_Spatial
        self.S_Spatial = S_Spatial
        
        # add spatial graph learning loss in the layer
        self.add_loss(F_norm_loss(self.S_Spatial,self.alpha))
        self.add_loss(diff_loss(self.diff_Spatial,self.S_Spatial))

        return S_Spatial

    def compute_output_shape(self, input_shape):
        # shape: (N, V, V)
        return (input_shape[0], input_shape[2], input_shape[2])
#%%
class Graph_Learn_Temporal(Layer):
    '''
    Temporal graph structure learning 
    --------
    Input:  (batch_size, num_of_timesteps, num_of_vertices, num_of_features)
    Output: (batch_size, num_of_timesteps, num_of_timesteps)
    '''
    def __init__(self, alpha, **kwargs):
        self.alpha = alpha
        self.S_Temporal = tf.convert_to_tensor([[[0.0]]])  # similar to placeholder
        self.diff_Temporal = tf.convert_to_tensor([[[[0.0]]]])  # similar to placeholder
        super(Graph_Learn_Temporal, self).__init__(**kwargs)

    def build(self, input_shape):
        _, num_of_timesteps, num_of_vertices, num_of_features = input_shape
        self.a_Temporal = self.add_weight(name='a_Temporal',
                                 shape=(num_of_features, 1),
                                 initializer='uniform',
                                 trainable=True)
        super(Graph_Learn_Temporal, self).build(input_shape)

    def call(self, x):
        #Input:  [N, timesteps, vertices, features]
        _, T, V, F = x.shape
        N = tf.shape(x)[0]
        
        # Temporal Graph Learning:
        for ff in range(int(F)):
            x_Temporal_ff = x[:, :, :, ff] #(N, T, V)

            diff_Temporal_temp = K.abs(tf.transpose(tf.transpose(tf.broadcast_to(x_Temporal_ff, [T,N,T,V]), perm=[2,1,0,3])
            - x_Temporal_ff, perm=[1,0,2,3])) #(N, T, T, V)
            
            diff_Temporal_temp = K.expand_dims(diff_Temporal_temp, axis=1) #(N, 1, T, T, V)

            if ff == 0:
                diff_Temporal = diff_Temporal_temp
            else:
                diff_Temporal = K.concatenate((diff_Temporal, diff_Temporal_temp), axis=1) #(N, F, T, T, V)
                
        tmpS = K.exp(K.relu(K.reshape(K.dot(tf.reduce_mean(tf.transpose(diff_Temporal, perm=[0, 4, 2, 3, 1]), axis=1), self.a_Temporal), [N,T,T]))) #(N, T, T)
        
        # normalization
        S_Temporal = tmpS / K.sum(tmpS, axis=1, keepdims=True)
        
        self.diff_Temporal = diff_Temporal
        self.S_Temporal = S_Temporal
        
        # add temporal graph learning loss in the layer
        self.add_loss(F_norm_loss(self.S_Temporal,self.alpha))
        self.add_loss(diff_loss(self.diff_Temporal, self.S_Temporal))

        return S_Temporal

    def compute_output_shape(self, input_shape):
        # shape: (N, T, T)
        return  ((input_shape[0], input_shape[1], input_shape[1]))
#%%
class cheb_conv_with_SAt_GL(Layer):
    '''
    K-order chebyshev graph convolution after Graph Learn
    --------
    Input:  [x   (batch_size, num_of_timesteps, num_of_vertices, num_of_features),
             SAtt(batch_size, num_of_vertices, num_of_vertices),
             S   (batch_size, num_of_vertices, num_of_vertices)]
    Output: (batch_size, num_of_timesteps, num_of_vertices, num_of_filters)
    '''
    def __init__(self, num_of_filters, k, **kwargs):
        self.k = k
        self.num_of_filters = num_of_filters
        super(cheb_conv_with_SAt_GL, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        x_shape,SAtt_shape,S_shape=input_shape
        _, num_of_timesteps, num_of_vertices, num_of_features = x_shape
        self.Theta = self.add_weight(name='Theta',
                                     shape=(self.k, num_of_features, self.num_of_filters),
                                     initializer='uniform',
                                     trainable=True)
        super(cheb_conv_with_SAt_GL, self).build(input_shape)

    def call(self, x):
        #Input:  [x,SAtt,S]
        assert isinstance(x, list)
        assert len(x)==3,'cheb_conv_with_SAt_GL: number of input error'
        x, spatial_attention, W = x
        _, num_of_timesteps, num_of_vertices, num_of_features = x.shape
        #Calculating Chebyshev polynomials
        D = tf.matrix_diag(K.sum(W,axis=1))
        L = D - W
        '''
        Here we approximate λ_{max} to 2 to simplify the calculation.
        For more general calculations, please refer to here:
            lambda_max = K.max(tf.self_adjoint_eigvals(L),axis=1)
            L_t = (2 * L) / tf.reshape(lambda_max,[-1,1,1]) - [tf.eye(int(num_of_vertices))]
        '''
        lambda_max = 2.0
        L_t = (2 * L) / lambda_max - [tf.eye(int(num_of_vertices))]
        cheb_polynomials = [tf.eye(int(num_of_vertices)), L_t]
        for i in range(2, self.k):
            cheb_polynomials.append(2 * L_t * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])
        
        #Graph Convolution
        outputs=[]
        for time_step in range(num_of_timesteps):
            # shape of x is (batch_size, V, F)
            graph_signal = x[:, time_step, :, :]
            # shape of x is (batch_size, V, F')
            output = tf.zeros(shape = (tf.shape(x)[0], num_of_vertices, self.num_of_filters))
            
            for kk in range(self.k):
                # shape of T_k is (V, V)
                T_k = cheb_polynomials[kk]
                    
                # shape of T_k_with_at is (batch_size, V, V)
                T_k_with_at = T_k * spatial_attention

                # shape of theta_k is (F, num_of_filters)
                theta_k = self.Theta[kk]

                # shape is (batch_size, V, F)
                rhs = K.batch_dot(tf.transpose(T_k_with_at,perm=[0, 2, 1]), graph_signal)

                output = output + K.dot(rhs, theta_k)
            outputs.append(tf.expand_dims(output,1))
            
        return K.relu(K.concatenate(outputs, axis = 1))

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        # shape: (n, num_of_timesteps, num_of_vertices, num_of_filters)
        return (input_shape[0][0],input_shape[0][1],input_shape[0][2],self.num_of_filters)

#%%
def ProductGraphSleepBlock(x, k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, useGL, GLalpha, i=0):
    '''
    packaged Spatial-temporal convolution Block
    -------
    '''        
    # SpatialAttention
    # output shape is (batch_size, V, V)
    spatial_At = SpatialAttention()(x)
    
    # Graph Convolution with spatial attention
    # output shape is (batch_size, T, V, F)

    # use adaptive Graph Learn
    S_Spatial = Graph_Learn_Spatial(alpha=GLalpha)(x)
    spatial_gcn = cheb_conv_with_SAt_GL(num_of_filters=num_of_chev_filters, k=k)([x, spatial_At, S_Spatial])    
    S_Temporal = Graph_Learn_Temporal(alpha=GLalpha)(x)
        
    return spatial_gcn, S_Temporal

#%%
def build_ProductGraphSleepNet(k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel, 
                sample_shape, num_block, opt, useGL, GLalpha, regularizer, GRU_Cell, attn_heads, dropout):
    
    # Input:  (*, num_of_timesteps, num_of_vertices, num_of_features)
    data_layer = layers.Input(shape=sample_shape, name='Input-Data')
    
    # ProductGraphSleepBlock
    block_out, S_Temporal = ProductGraphSleepBlock(data_layer,k, num_of_chev_filters, num_of_time_filters, time_conv_strides, cheb_polynomials, time_conv_kernel,useGL,GLalpha)

    # BiLSTM
    block_out = Reshape((-1, int(block_out.shape[2]*block_out.shape[3])))(block_out)
    x_GRU = Bidirectional(GRU(GRU_Cell, dropout=dropout, recurrent_dropout=dropout, return_sequences=True))(block_out)    
    nclass = 5
    softmax = GraphWiseAttentionNetwork(nclass, attn_heads=attn_heads,
                                       attn_heads_reduction='average',
                                       dropout_rate=dropout,
                                       activation='softmax',
                                       kernel_regularizer=None,
                                       attn_kernel_regularizer=None)([x_GRU, S_Temporal])
    
    # define model
    model = models.Model(inputs=data_layer, outputs=softmax)
    
    # load evaluation metrics
    # metrics, _ = get_evaluation_metrics()
    # metric_names = ['accuracy', 'categorical_accuracy','PR-AUC', 'F1']
    metrics = ['acc', 'CategoricalAccuracy', tf.keras.metrics.AUC(curve='PR'), dice_coef]
    # metrics = [dice_coef_mask]
    # metrics = [dice_coef]

    # compile model
    # model.compile(
    #     optimizer=opt,
    #     loss='categorical_crossentropy',
    #     metrics=metrics)
    # model.compile(
    #     optimizer=opt,
    #     loss=weighted_categorical_crossentropy_cut,
    #     metrics=metrics)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=metrics)
    
    return model