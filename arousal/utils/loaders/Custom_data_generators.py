import keras
import numpy as np
import h5py as h5
from glob import glob
import os
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
#keras.utils.all_utils.all_utils

from arousal.utils.load_write.ids2label import *
from arousal.utils.loaders.load_select_functions_for_loader import *
import time
# import ray
# from multiprocessing import current_process


# ray.init(num_cpus=40)

class UTime_DataGenerator(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, label = load_patients(self,pat_list,self.dataset_params['dataset_label_list'])
                succes = True
            except:
                pass
        
        if self.label_shape == label.shape:
            if self.label_type == 'hypnogram':
                label = keras.utils.all_utils.to_categorical(label)[:,:,1:]
            elif self.label_type == 'arousal':
                label = keras.utils.all_utils.to_categorical(label, num_classes=2)
            elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
                label = keras.utils.all_utils.to_categorical(label, num_classes=3)
        else:
            #reshape to right shape
            label = label.reshape(self.label_shape+(-1,))
            label = np.mean(label,axis=3)
            if self.label_type == 'hypnogram':
                label = keras.utils.all_utils.to_categorical(label)[:,:,:,1:]
            elif self.label_type == 'arousal':
                label = keras.utils.all_utils.to_categorical(label, num_classes=2)
            elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
                label = keras.utils.all_utils.to_categorical(label, num_classes=3)

        
         
        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label))[0])>0:
            print('error')
        
        return Xdata, label
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator(keras.utils.all_utils.Sequence):
    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label = load_patients_val(self,index,self.dataset_params['dataset_path_list'])
        label = label.reshape(self.val_label_shape)
        
        if len(self.label_shape) == len(label.shape):
            if self.label_type == 'hypnogram':
                label = keras.utils.all_utils.to_categorical(label)[:,:,:,1:]
            elif self.label_type == 'arousal':
                label = keras.utils.all_utils.to_categorical(label, num_classes=2)
            elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
                label = keras.utils.all_utils.to_categorical(label, num_classes=3)
        else:
            #reshape to right shape
            label_shape = (Xdata.shape[0],self.label_shape[1],self.label_shape[2],-1)
            label = label.reshape(label_shape)
            label = np.mean(label,axis=3)
            if self.label_type == 'hypnogram':
                label = keras.utils.all_utils.to_categorical(label)[:,:,1:]
            elif self.label_type == 'arousal':
                label = keras.utils.all_utils.to_categorical(label, num_classes=2)
            elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
                label = keras.utils.all_utils.to_categorical(label, num_classes=3)

         
        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label))[0])>0:
            print('error')
        
        return Xdata, label
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])    

def custom_data_generator(dataset_params,hparams):
    
    i = 0
    run = 0
    while True:


        #random seed
        process_id = current_process()._identity[0] if current_process()._identity else 1
        now = int(time.time() * 1e23)
        seed = (process_id+run + now ) % (2 ** 32)
        np.random.seed(seed)

        #generate list for epoch
        pat_list =  get_patients(dataset_params['dataset_path_list'], dataset_params['dataset_chance_list'],batch_size=1001*hparams['fit']['n_batches_epoch']*hparams['fit']['batch_size'])

        for i in range(0,len(pat_list),hparams['fit']['batch_size']):

            #LOAD PAT FOR BATCH
            pl = pat_list[i:i+hparams['fit']['batch_size']]
            
            #PREALLOCATE BATCH
            Xdata = np.zeros((hparams['fit']['batch_size'],35,3840,5))
            label_arousal = np.zeros((hparams['fit']['batch_size'],2100,2))

            ########################
            # LOAD BASED UPON HYPNO
            ########################

            #LOAD LIST
            pl_hypno = pl[:hparams['fit']['load_hypno']]
            
            #LOAD PAT
            futures1 = [load_patients_arousal_hypno_yield(hparams,i,pat,dataset_params['dataset_label_list']) for i,pat in enumerate(pl_hypno)]
            labels1 = futures1
            
            #FILL IN BATCH FOR HYPNO PART
            for i,[eeg, lab] in enumerate(labels1):
                Xdata[i,:,:,:] = eeg
                label_arousal[i,:,:]=lab

            ########################
            # LOAD BASED UPON AROUSAL
            ########################

            #LOAD LIST
            pl_arousal = pl[hparams['fit']['load_hypno']:]
            
            #LOAD PAT
            futures2 = [load_patients_arousal_arousal_yield(hparams,pat,num_30s_seg=35) for pat in pl_arousal]
            labels2 = futures2
            
            #FILL IN BATCH FOR AROUSAL PART
            for i,[eeg, lab] in enumerate(labels2):
                Xdata[i+hparams['fit']['load_hypno'],:,:,:] = eeg
                label_arousal[i+hparams['fit']['load_hypno'],:,:]=lab

            yield Xdata , label_arousal


   


class UTime_DataGenerator_arousal(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.scaled = hparams['data']['scaled']
        self.ref = hparams['data']['reference']
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False

        # process_id = current_process()._identity[0] if current_process()._identity else 1
        # now = int(time.time() * 1e23)
        # seed = (process_id+index + now ) % (2 ** 32)
        # np.random.seed(seed)
        
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)                
                #np.save(f'/home/erikjan/Desktop/batches/batch_{seed}_{now}.npy',pat_list)
                Xdata, label_arousal = load_patients_arousal_shallow(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg,ref=self.ref)
                # for xxx in range(35):
                #     for xx in range(5):
                #         plt.plot(np.arange(3840)/3840+xxx,Xdata[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                #     plt.plot(np.arange(60)/60+xxx,label_arousal[0,60*xxx:60*(xxx+1)]-(7*(xx+1)))
                # plt.show()
                # print('a')
                succes = True
            except:
                pass
        
        

        if self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]            
        else:
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')

        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])


class UTime_DataGenerator_arousal_stage_start(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.scaled = hparams['data']['scaled']
        self.ref = hparams['data']['reference']
        self.stage_start = hparams['data']['stage_start']
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False

        # process_id = current_process()._identity[0] if current_process()._identity else 1
        # now = int(time.time() * 1e23)
        # seed = (process_id+index + now ) % (2 ** 32)
        # np.random.seed(seed)
        
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)                
                #np.save(f'/home/erikjan/Desktop/batches/batch_{seed}_{now}.npy',pat_list)
                Xdata, label_arousal = load_patients_arousal_shallow(self,pat_list,np.ones(self.batch_size)*self.stage_start,num_30s_seg=self.num_30s_seg,ref=self.ref)
                # for xxx in range(35):
                #     for xx in range(5):
                #         plt.plot(np.arange(3840)/3840+xxx,Xdata[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                #     plt.plot(np.arange(60)/60+xxx,label_arousal[0,60*xxx:60*(xxx+1)]-(7*(xx+1)))
                # plt.show()
                # print('a')
                succes = True
            except:
                pass
        
        

        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_0' or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')

        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])


class UTime_DataGenerator_arousal_n1(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.scaled = hparams['data']['scaled']
        self.ref = hparams['data']['reference']
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)                
                Xdata, label_arousal = load_patients_arousal_shallow_n1(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg,ref=self.ref)
                # for xxx in range(35):
                #     for xx in range(5):
                #         plt.plot(np.arange(3840)/3840+xxx,Xdata[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                #     plt.plot(np.arange(60)/60+xxx,label_arousal[0,60*xxx:60*(xxx+1)]-(7*(xx+1)))
                # plt.show()
                # print('a')
                succes = True
            except:
                pass
        
        

        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_0' or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        if self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]
        else:
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')

        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])


class UTime_DataGenerator_arousal_aux(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.scaled = hparams['data']['scaled']
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],index,batch_size=self.batch_size)
                
                Xdata, [label_hypno,label_arousal]  = load_patients_arousal_shallow(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                # for xxx in range(35):
                #     for xx in range(5):
                #         plt.plot(np.arange(3840)/3840+xxx,Xdata[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                #     plt.plot(np.arange(60)/60+xxx,label_arousal[0,60*xxx:60*(xxx+1)]-(7*(xx+1)))
                # plt.show()
                # print('a')
                succes = True
            except:
                pass
        
        
        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_0' or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_hypno))[0])>0:
            print('error')
        
        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator_arousal(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.scaled = hparams['data']['scaled']
        self.ref = hparams['data']['reference']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label_arousal = load_patients_val_arousal(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35,ref=self.ref)
        
        # #test 
        # for xxxx in range(Xdata.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,Xdata[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')


        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_2' or  self.label_type =='arousal-added_converted_0' or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]
        else:
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)


        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #np.random.shuffle(self.dataset_params['dataset_label_list']) 
        pass


class UTime_Val_DataGenerator_arousal_n1(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.scaled = hparams['data']['scaled']
        self.ref = hparams['data']['reference']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label_arousal = load_patients_val_arousal_n1(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35,ref=self.ref)

        
        # #test 
        # for xxxx in range(Xdata.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,Xdata[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')


        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_0'or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]


        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #np.random.shuffle(self.dataset_params['dataset_label_list']) 
        pass


class UTime_Val_DataGenerator_arousal_aux(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.scaled = hparams['data']['scaled']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, [label_hypno,label_arousal] = load_patients_val_arousal(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)

        # #test 
        # for xxxx in range(Xdata.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,Xdata[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')


        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type =='arousal' or self.label_type =='arousal-platinum_converted_0'or self.label_type =='arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal-shifted_converted_1_NREM':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)[:,:,:-1]


        return Xdata, [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #np.random.shuffle(self.dataset_params['dataset_label_list']) 
        pass




class UTime_DataGenerator_arousal_hypno_plus(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, label_hypno_plus  = load_patients_arousal_shallow_hypno_plus(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                # for xxx in range(35):
                #     for xx in range(5):
                #         plt.plot(np.arange(3840)/3840+xxx,Xdata[0,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

                #     plt.plot(np.arange(60)/60+xxx,label_arousal[0,60*xxx:60*(xxx+1)]-(7*(xx+1)))
                # plt.show()
                # print('a')
                succes = True
            except:
                pass
        
        
        label_hypno_plus = keras.utils.all_utils.to_categorical(label_hypno_plus,num_classes=7)[:,:,1:]


        return Xdata, label_hypno_plus
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator_arousal_hypno_plus(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label_hypno_plus= load_patients_val_arousal_hypno_plus(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)

        # #test 
        # for xxxx in range(Xdata.shape[0]):
        #     for xxx in range(35):
        #         for xx in range(5):
        #             plt.plot(np.arange(3840)/3840+xxx,Xdata[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

        #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
        #     plt.show()
        #     print('a')

        
        label_hypno_plus = keras.utils.all_utils.to_categorical(label_hypno_plus,num_classes=7)[:,:,1:]


        return Xdata, label_hypno_plus
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        #np.random.shuffle(self.dataset_params['dataset_label_list']) 
        pass



class UTime_DataGenerator_arousal_NREM(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, label_arousal  = load_patients_arousal_NREM(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                succes = True
            except:
                pass
        

        label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)

        hypno = np.zeros((32,35,1))
        hypno = keras.utils.all_utils.to_categorical(hypno, num_classes=5)
 
        
        return Xdata, [hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_DataGenerator_arousal_REM(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, label_arousal  = load_patients_arousal_REM(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                succes = True
            except:
                pass
        
        
        label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)


        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')

        
        return Xdata, label_arousal
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_val_DataGenerator_arousal_NREM(keras.utils.all_utils.Sequence):
    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label = load_val_patients_arousal_NREM(self,index,self.dataset_params['dataset_path_list'])
        label = keras.utils.all_utils.to_categorical(label, num_classes=2)


        hypno = np.zeros((label.shape[0],35,1))
        hypno = keras.utils.all_utils.to_categorical(hypno, num_classes=5)
 

        return Xdata, [hypno,label]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])    

class UTime_val_DataGenerator_arousal_REM(keras.utils.all_utils.Sequence):
    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, label = load_val_patients_arousal_REM(self,index,self.dataset_params['dataset_path_list'])
        label = keras.utils.all_utils.to_categorical(label, num_classes=2)

       
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label))[0])>0:
            print('error')
        
        return Xdata, label
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])    


class UTime_DataGenerator_arousal_idx(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, [label_hypno,label_arousal]  = load_patients_arousal_idx(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                succes = True
            except:
                pass
        
        
        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type =='arousal' or 'arousal_plus' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2' or 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_hypno))[0])>0:
            print('error')
        
        return Xdata, [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

# class UTime_Val_DataGenerator_arousal(keras.utils.all_utils.Sequence):

    
#     'Generates data for Keras'

#     def __init__(self,dataset_params,hparams):
#         self.dataset_params = dataset_params
#         self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
#         self.batch_size = hparams['fit']['batch_size']
#         self.batches_per_epoch = hparams['fit']['n_batches_epoch']
#         self.n_channels = hparams['build']['n_channels']
#         self.chan_type = hparams['channels']
#         self.n_classes = hparams['build']['n_classes']
#         self.n_classes_b1 = hparams['build']['n_classes_b1']
#         self.n_classes_b2 = hparams['build']['n_classes_b2']
#         self.data_per_prediction = hparams['build']['data_per_prediction']
#         self.label_type = hparams['data']['label_type']
#         self.label_hz = hparams['data']['label_Hz']
#         self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
#         self.val_label_shape = (-1,int(30*35*self.label_hz) )
#         self.data_hz = hparams['data']['data_Hz']
#         self.on_epoch_end()
        
        
    
#     def __len__(self):
#         'Denotes the number of batches per epoch'
#         return len(self.dataset_params['dataset_path_list'])

#     def __getitem__(self, index):

#         'Generate one batch of data'
#         Xdata, [label_hypno,label_arousal] = load_patients_val_arousal(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)

#         # #test 
#         # for xxxx in range(Xdata.shape[0]):
#         #     for xxx in range(35):
#         #         for xx in range(5):
#         #             plt.plot(np.arange(3840)/3840+xxx,Xdata[xxxx,xxx,:,xx]-(7*xx),linewidth=0.5,color='black')

#         #         plt.plot(np.arange(60)/60+xxx,label_arousal[xxxx,60*xxx:60*(xxx+1)]-(7*(xx+1)))
#         #     plt.show()
#         #     print('a')


#         label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
#         if self.label_type =='arousal' or 'arousal-shifted_converted_0' and self.label_type not in 'arousal_plus_masked':
#             label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
#         elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2' or 'arousal_plus_masked':
#             label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)


#         return Xdata, [label_hypno,label_arousal]
    

#     def on_epoch_end(self):
#         'Updates indexes after each epoch'
#         #np.random.shuffle(self.dataset_params['dataset_label_list']) 
#         pass

class UTime_Val_DataGenerator_arousal_idx(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, [label_hypno,label_arousal] = load_patients_val_arousal_idx(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)


        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type =='arousal' or 'arousal_plus' and self.label_type not in 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2' or 'arousal_plus_masked':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)

        return Xdata, [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list']) 


class UTime_DataGenerator_arousal_psd(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],hparams['build']['batch_shape'][1],30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                [Xdata,label_psd], [label_hypno,label_arousal]  = load_patients_arousal_shallow_psd(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                succes = True
            except:
                pass
        
        
        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type=='arousal':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_hypno))[0])>0:
            print('error')
        # Xdata = np.zeros((32, 35, 3840, 5))
        #label_psd = np.zeros((32, 2100, 1,4))
        # label_hypno = np.zeros((32, 35, 5))
        # label_arousal = np.zeros((32, 2100, 2))

        return [Xdata,label_psd], [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator_arousal_PSD(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.label_shape = (hparams['fit']['batch_size'],int(30*35*self.label_hz) )
        self.val_label_shape = (-1,int(30*35*self.label_hz) )
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, [label_hypno,label_arousal] = load_patients_val_arousal_PSD(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)


        label_hypno = keras.utils.all_utils.to_categorical(label_hypno)[:,:,1:]
        if self.label_type=='arousal':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)
        

        return Xdata, [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list']) 




class UTime_DataGenerator_breathing(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = hparams['build']['batch_shape']
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = self.dim[0]
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['batch_shape'][-1]
        self.chan_type = hparams['channels']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction_b1 = hparams['build']['data_per_prediction_b1']
        self.data_per_prediction_b2 = hparams['build']['data_per_prediction_b2']
        self.label_type = hparams['data']['label_type']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (self.dim[0],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (self.dim[0],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, [label_resp,label_arousal]  = load_patients_breathing(self,pat_list,self.dataset_params['dataset_label_list'],num_30s_seg=self.num_30s_seg)
                succes = True
            except:
                pass
        
        
        label_resp = keras.utils.all_utils.to_categorical(label_resp, num_classes=6)
        label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)


        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_resp))[0])>0:
            print('error')
        
        return Xdata, [label_resp,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator_breathing(keras.utils.all_utils.Sequence):

    
    'Generates data for Keras'

    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = hparams['build']['batch_shape']
        self.num_30s_seg = hparams['build']['batch_shape'][1]
        self.batch_size = self.dim[0]
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['batch_shape'][-1]
        self.chan_type = hparams['channels']
        self.n_classes_b1 = hparams['build']['n_classes_b1']
        self.n_classes_b2 = hparams['build']['n_classes_b2']
        self.data_per_prediction_b1 = hparams['build']['data_per_prediction_b1']
        self.data_per_prediction_b2 = hparams['build']['data_per_prediction_b2']
        self.label_type = hparams['data']['label_type']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (self.dim[0],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (self.dim[0],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.val_label_shape = (-1,int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']) )
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        'Generate one batch of data'
        Xdata, [label_resp,label_arousal] = load_patients_val_breathing(self,index,self.dataset_params['dataset_path_list'],num_30s_seg=35)


        label_resp = keras.utils.all_utils.to_categorical(label_resp, num_classes=6)
        label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_resp))[0])>0:
            print('error')
        
        return Xdata, [label_resp,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list']) 


            


class UTime_DataGenerator_no_EEG_EOG_restricion(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,batches_per_epoch,batch_size=64, n_channels=2,chan_type=None,n_classes=5,data_per_prediction=3840):
        'Initialization'
        self.dataset_params = dataset_params
        self.dim = [batch_size,35,3840,n_channels]
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.n_channels = n_channels
        self.chan_type = chan_type
        self.n_classes = n_classes
        self.data_per_prediction = data_per_prediction
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        
        def get_patients(data_list_per_set, chance_list, batch_size=self.batch_size):

            pat_list_idx = np.random.choice(len(data_list_per_set),batch_size+10,replace=False,p=chance_list)
            pat_list = [data_list_per_set[int(x)] for x in pat_list_idx]
            return pat_list

        def create_hypno(ids,fs=128):

            #length last line 
            lst =  [pos for pos, char in enumerate(ids[-1]) if char == ',']
            #add start + length of last line
            len_ids = int(ids[-1][:lst[0]])+ int(ids[-1][lst[0]+1:lst[1]])
            lab = np.empty((len_ids*fs,))

            #create hypno
            for i in range(len(ids)):
                ids_ = ids[i]
                arr = ids_.split(',')
                if arr[2] == 'W' or arr[2] == 'W\n' or arr[2] == 'Sleep stage W\n' or arr[2] == 'Sleep stage W':
                    arr[2] = 5
                if arr[2] == 'REM' or arr[2] == 'REM\n' or arr[2] == 'Sleep stage R\n' or arr[2] == 'Sleep stage R':
                    arr[2] = 4
                if arr[2] == 'N1' or arr[2] == 'N1\n' or arr[2] == 'Sleep stage 1\n' or arr[2] == 'Sleep stage 1':
                    arr[2] = 3
                if arr[2] == 'N2' or arr[2] == 'N2\n' or arr[2] == 'Sleep stage 2\n' or arr[2] == 'Sleep stage 2' or arr[2] =='uncertain' or arr[2] =='uncertain\n':
                    arr[2] = 2
                if arr[2] == 'N3' or arr[2] == 'N3\n' or arr[2] == 'Sleep stage 3\n' or arr[2] == 'Sleep stage 4\n' or arr[2] == 'Sleep stage 3' or arr[2] == 'Sleep stage 4' or arr[2] == 'arousal' or arr[2] == 'arousal\n':
                    arr[2] = 1
                if arr[2] == 'UNKNOWN' or arr[2] == 'UNKNOWN\n' or arr[2] == '?\n' or arr[2] == '?' or arr[2] == 'Movement time\n' or arr[2] == 'Sleep stage ?\n':
                    arr[2] = 0
                
                lab[int(arr[0])*fs:int(arr[0])*fs+int(arr[1])*fs] = arr[2]
            return lab

        def read_label(path):
            with open(path,'r') as f:
                ids = f.readlines()
                hypno = create_hypno(ids,fs=128)
            return hypno

        def load_h5_data(f,channels,start,end,fs=128):
            EEG = np.zeros((int((end-start)/35/3840),35,fs*30,len(channels)))
            for i,chan in enumerate(channels):
                EEG[:,:,:,i] = np.array(f['channels'][chan][start:end]).reshape((-1,35,3840))
            return EEG
            
        def find_start_idx_label(hypno,start_label):
            hypno = np.mean(hypno.reshape(-1,3840),axis=1)
            loc = np.where(hypno==start_label)[0]
            loc = [n for n in loc if n < len(hypno)-35]
            np.random.shuffle(loc)
            
            try:
                return loc[0]*128*30
            except:
                loc = np.arange(len(hypno)-36)
                np.random.shuffle(loc)
                return loc[0]*128*30
        
        def load_channels(all_channels,eeg_eog_list):
            all_eeg = [x for x in all_channels if x in eeg_eog_list['eeg']]
            all_eog = [x for x in all_channels if x in eeg_eog_list['eog']]
            eeg_to_load = np.random.choice(all_eeg,size=1,replace=False)
            all_second = [x for x in all_eeg+all_eog if x not in eeg_to_load[0]]
            eog_to_load = np.random.choice(all_second,size=1,replace=False)
            return [eeg_to_load[0],eog_to_load[0]]
                
        def load_patients(self,pat_list,start_label,batch_size=self.batch_size):
            
            #pre-allocate data
            EEG = np.zeros((batch_size,35,128*30,2))
            label = np.zeros((batch_size,35))

            #loop though patients
            i = 0
            success = 0
            while success<self.batch_size:
                

                pat =pat_list[i]
                try:
                    #read hypno
                    hypno = read_label(glob(os.path.join(pat,'*.ids'))[0])
                    
                    #find suiting 17,5 minutes of data
                    start_idx = find_start_idx_label(hypno,start_label[np.min((i,len(start_label)-1))])
                    end_idx = start_idx+(35*128*30)

                    with h5.File(glob(os.path.join(pat,'*.h5'))[0],'r') as f:
                        
                        channels = [x for x in f['channels'].keys()]
                        channels_to_load = load_channels(channels,self.chan_type)
                        
                        #load data
                        if len(f['channels'][channels_to_load[0]]) == len(hypno):
                            hypnogram = hypno[start_idx:end_idx]
                            hypnogram = np.mean(hypnogram.reshape(-1,3840),axis=1)
                            eeg = load_h5_data(f,channels_to_load,start_idx,end_idx)
                            
                            EEG[success,:,:,:] = eeg
                        label[success,:]= hypnogram#raises error when len hypno and eeg are not similar

                        i+=1
                        success+=1
                except:
                    #f = h5.File(glob(os.path.join(pat,'*.h5'))[0],'r') 
                    i+=1
                  
            EEG=EEG[:success,:,:,:]
            label=label[:success,:]

                
            return EEG, label  
          
        'Generate one batch of data'
        pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
        
        Xdata, label = load_patients(self,pat_list,self.dataset_params['dataset_label_list'])
        
        if self.data_per_prediction!=3840:
            label = np.expand_dims(label,axis=2)
            label = np.repeat(label,3840/self.data_per_prediction,axis=2)
         
        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label))[0])>0:
            print('error')
        
        return Xdata, keras.utils.all_utils.to_categorical(label)[:,:,1:]#label
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

class UTime_Val_DataGenerator_no_EEG_EOG_restricion(keras.utils.all_utils.Sequence):
    
    'Generates data for Keras'
    def __init__(self,dataset_params, n_channels=2,chan_type=None,n_classes=5,data_per_prediction=3840):
        'Initialization'
        self.dataset_params = dataset_params
        self.chan_type = chan_type
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.data_per_prediction = data_per_prediction
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.dataset_params['dataset_path_list'])

    def __getitem__(self, index):

        def create_hypno(ids,fs=128):

            #length last line 
            lst =  [pos for pos, char in enumerate(ids[-1]) if char == ',']
            #add start + length of last line
            len_ids = int(ids[-1][:lst[0]])+ int(ids[-1][lst[0]+1:lst[1]])
            lab = np.empty((len_ids*fs,))

            #create hypno
            for i in range(len(ids)):
                ids_ = ids[i]
                arr = ids_.split(',')
                if arr[2] == 'W' or arr[2] == 'W\n' or arr[2] == 'Sleep stage W\n' or arr[2] == 'Sleep stage W':
                    arr[2] = 5
                if arr[2] == 'REM' or arr[2] == 'REM\n' or arr[2] == 'Sleep stage R\n' or arr[2] == 'Sleep stage R':
                    arr[2] = 4
                if arr[2] == 'N1' or arr[2] == 'N1\n' or arr[2] == 'Sleep stage 1\n' or arr[2] == 'Sleep stage 1':
                    arr[2] = 3
                if arr[2] == 'N2' or arr[2] == 'N2\n' or arr[2] == 'Sleep stage 2\n' or arr[2] == 'Sleep stage 2' or arr[2] =='uncertain' or arr[2] =='uncertain\n':
                    arr[2] = 2
                if arr[2] == 'N3' or arr[2] == 'N3\n' or arr[2] == 'Sleep stage 3\n' or arr[2] == 'Sleep stage 4\n' or arr[2] == 'Sleep stage 3' or arr[2] == 'Sleep stage 4' or arr[2] == 'arousal' or arr[2] == 'arousal\n':
                    arr[2] = 1
                if arr[2] == 'UNKNOWN' or arr[2] == 'UNKNOWN\n' or arr[2] == '?\n' or arr[2] == '?' or arr[2] == 'Movement time\n' or arr[2] == 'Sleep stage ?\n':
                    arr[2] = 0
                
                lab[int(arr[0])*fs:int(arr[0])*fs+int(arr[1])*fs] = arr[2]
            return lab

        def read_label(path):
            with open(path,'r') as f:
                ids = f.readlines()
                hypno = create_hypno(ids,fs=128)
            return hypno

        def load_h5_data(f,channels,start,end):
            EEG = np.zeros((int((end-start)/35/3840),35,128*30,len(channels)))
            for i,chan in enumerate(channels):
                EEG[:,:,:,i] = np.array(f['channels'][chan][start:end]).reshape((-1,35,3840))
            return EEG
        
        def load_channels(all_channels,eeg_eog_list):
            all_eeg = [x for x in all_channels if x in eeg_eog_list['eeg']]
            all_eog = [x for x in all_channels if x in eeg_eog_list['eog']]
            eeg_to_load = np.random.choice(all_eeg,size=1,replace=False)
            all_second = [x for x in all_eeg+all_eog if x not in eeg_to_load[0]]
            eog_to_load = np.random.choice(all_second,size=1,replace=False)
            return [eeg_to_load[0],eog_to_load[0]]
        
        def load_patients(self,index,pat_list):
            
            pat = pat_list[index]
            try:
                #read hypno
                hypno = read_label(glob(os.path.join(pat,'*.ids'))[0])
                
                #find suiting 17,5 minutes of data
                start_idx = 0
                end_idx = (len(hypno)//(3840*35))*(3840*35)
                
                with h5.File(glob(os.path.join(pat,'*.h5'))[0],'r') as f:
                    channels = [x for x in f['channels'].keys()]
                    channels_to_load = load_channels(channels,self.chan_type)
                    
                    #load data
                    hypnogram = hypno[start_idx:end_idx]
                    hypnogram = np.mean(hypnogram.reshape(-1,3840),axis=1).reshape(-1,35)
                    hypnogram = keras.utils.all_utils.to_categorical(hypnogram)[:,:,1:]
                    eeg = load_h5_data(f,channels_to_load,start_idx,end_idx)
            except:
                eeg = np.zeros((64,35,3840,2))
                hypnogram = np.zeros((64,35,5))
                

            if hypnogram.shape[2]==0:
                eeg = np.zeros((64,35,3840,2))
                hypnogram = np.zeros((64,35,5))
                
            if eeg.shape[0]>64:
                eeg=eeg[:64,:,:,:]
                hypnogram =hypnogram[:64,:,:]
            
            if eeg.shape[0]<64:
                rows_to_fill = 64-eeg.shape[0]
                EEG=np.zeros((64,35,3840,2))
                HYPNO = np.zeros((64,35,5))
                EEG[:eeg.shape[0],:,:,:]=eeg
                start = eeg.shape[0]
                while rows_to_fill>0:
                    end = start+np.min((rows_to_fill,eeg.shape[0]))
                    EEG[start:end,:,:,:]=eeg[:end-start,:,:,:]
                    HYPNO[start:end,:,:]=hypnogram[:end-start,:,:]
                    rows_to_fill=rows_to_fill+start-end
                    start=end
                eeg=EEG
                hypnogram=HYPNO
                
            return eeg, hypnogram  
          
        'Generate one batch of data'
        Xdata, label = load_patients(self,index,self.dataset_params['dataset_path_list'])
        
        if self.data_per_prediction!=3840:
            label = np.expand_dims(label,axis=2)
            label = np.repeat(label,3840/self.data_per_prediction,axis=2)
         
        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label))[0])>0:
            print('error')
        
        return Xdata, label
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])  
    
class UTime_DataGenerator_arousal_hem(keras.utils.all_utils.Sequence):
    'Generates data for Keras'
    def __init__(self,dataset_params,hparams):
        self.dataset_params = dataset_params
        self.dim = [hparams['fit']['batch_size'],35,30*hparams['data']['data_Hz'],hparams['build']['n_channels']]
        self.batch_size = hparams['fit']['batch_size']
        self.batches_per_epoch = hparams['fit']['n_batches_epoch']
        self.n_channels = hparams['build']['n_channels']
        self.chan_type = hparams['channels']
        self.n_classes = hparams['build']['n_classes']
        self.data_per_prediction = hparams['build']['data_per_prediction']
        self.label_type = hparams['data']['label_type']
        self.label_hz = hparams['data']['label_Hz']
        self.hypno_hz = 1/int(hparams['data']['hypno_Hz'])
        self.load_hypno = hparams['fit']['load_hypno']
        self.load_arousal = hparams['fit']['load_arousal']
        self.data_hz = hparams['data']['data_Hz']
        self.label_shape_b1 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b1']))
        self.label_shape_b2 = (hparams['fit']['batch_size'],int(30*35*self.data_hz/hparams['build']['data_per_prediction_b2']))
        self.pred_path = hparams['hard_example_mining']['pred_path']
        
        self.on_epoch_end()
        
        
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.batches_per_epoch

    def __getitem__(self, index):
        succes = False
        while succes==False:
            try:
                'Generate one batch of data'
                pat_list =  get_patients(self.dataset_params['dataset_path_list'], self.dataset_params['dataset_chance_list'],batch_size=self.batch_size)
                
                Xdata, [label_hypno,label_arousal]  = load_patients_arousal_hem(self,pat_list,self.dataset_params['dataset_label_list'])
                succes = True
            except:
                pass
        
        
        label_hypno = keras.utils.all_utils.to_categorical(label_hypno,num_classes=6)[:,:,1:]
        if self.label_type=='arousal':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=2)
        elif self.label_type == 'arousal_enhanced' or 'arousal_enhanced_V2':
            label_arousal = keras.utils.all_utils.to_categorical(label_arousal, num_classes=3)

        
        if len(np.where(np.isnan(Xdata))[0])>0:
            print('error')
        if len(np.where(np.isnan(label_hypno))[0])>0:
            print('error')
        
        return Xdata, [label_hypno,label_arousal]
    

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        np.random.shuffle(self.dataset_params['dataset_label_list'])

