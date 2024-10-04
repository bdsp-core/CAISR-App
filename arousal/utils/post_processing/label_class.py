import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from arousal.utils.post_processing.smoothing_arousal import post_process_after_smoothing

def mask_data_breathing(input_dict, fs_annotation):

    #create masked labels 
    masked_resp = CAISER_labels(input_dict, fs=fs_annotation)
    masked_resp.set_mask(0)
    masked_resp.set_wake_label(4)
    masked_resp = masked_resp.get_masked_resp_label()
    
    return masked_resp


def mask_data_arousal(input_dict, fs_annotation):

    #create masked labels
    masked_arousal = CAISER_labels(input_dict,fs=fs_annotation)
    masked_arousal.set_mask(0)
    masked_arousal.set_add_before_wake(0)
    masked_arousal.set_add_after_wake(10)
    masked_arousal.set_wake_label(4)
    masked_arousal = masked_arousal.get_masked_arousal_label_eval()
    masked_arousal = post_process_after_smoothing(masked_arousal,[1],N_remove_length=2,N_max_len=30,Hz=fs_annotation)

    return masked_arousal
    

def mask_data_limb(input_dict, fs_annotation):
    
    #create masked labels
    masked_limb = CAISER_labels(input_dict,fs=fs_annotation)
    masked_limb.set_mask(0)
    masked_limb.set_wake_label(4)
    masked_limb = masked_limb.get_masked_limp_label()

    return masked_limb

def mask_data_hypno(hypno):
    return hypno

class CAISER_labels:
    #class that will load given labels from patients
    #input: load dictionary {'hypno':np.array,'arousal':np.array,'limp':np.array and 'resp':np.array} 
    #       not all inputs are required class functions work with the provided data 
    #output: unstable sleep vectors, labels with uncertaincy and post processed labels based on available data
    
    
    #CLASS VARIABELS
    #varables that might change later on 
    #can be changed by set_(variable name)
    wake_label = 4 #unitless
    nan_label_hypno = -7 #label
    add_prior_event_arousal = 2 #sec
    add_after_event_arousal = 10 #sec
    add_prior_event_limp = 2 #sec
    add_after_event_limp = 10 #sec
    add_prior_event_resp = 2 #sec
    add_after_event_resp = 2 #sec
    add_after_wake = 10 #sec
    add_before_wake = 0 #sec
    uncertainty_label = -9 #unitless
    unstable_sleep_label = -8 #unitless
    
        
    # load labels per patient
    def __init__(self,label_dict_,fs,remove_nan=None):
        
        
        #check if all inputs are 
        array_check = [isinstance(label_dict_[x], type(np.array(1))) for x in label_dict_.keys()]
        if not all(array_check):
            raise Exception("All variables in input dictionary must be arrays...")
        
        #check for names
        for x in label_dict_.keys():
            if x not in ['hypno','arousal','limp','resp']:
                raise Exception("Variable names in the input dictionary are limited to: 'hypno','arousal','limp' and 'resp' ")
        
        #check for same lengths
        lens = [len(label_dict_[x]) for x in label_dict_.keys()]   
        if len(np.unique(lens))>1:
            raise Exception("Variables lengths must match...")
            
        #get self statements    
        self.fs = fs
        self.label_dict = label_dict_.copy()
        self.remove_nan = remove_nan

        
    
    ###############################################
    ############## Hidden definitions #############
    ###############################################
    def __grab_label_if_available(self,str_,s,e):
        proceed = self.__check_availability(str_,continue_if_missing=True)
        if proceed:
            usv = self.__cut_label_with_padding(self.label_dict[str_],s,e,padding=0)
            return usv
        else:
            return []
        

    def __check_availability(self,str_,continue_if_missing=False):
        #Check if str is present in Label_dict. 
        #If not in dict raise error except continue_if_missing=False
        #output: variable containing the go/no-go statement for further analysis
        
        proceed = True
        if continue_if_missing==False and str_ not in self.label_dict.keys():
            raise Exception("No {} data available... ".format(str_))
        elif continue_if_missing==True and str_ not in self.label_dict.keys():
            print('###########################################')
            print("No {} data available, continuing without...  ".format(str_))
            print('###########################################')
            proceed = False
        
            
        return proceed
    
    def __cut_label_with_padding(self,label,s,e,padding=15):
        #if None, start at beginning
        if s == None:
            s=0
        #if start is smaller than padding size, start at beginning (partly padded)
        elif s<=padding*self.fs and s>=0:
            s=0
        #if start is bigger than label
        elif s>=len(label):
            raise Exception("Start idx to large...")
        #start is start idx - padding
        else:
            s=s-padding*self.fs
        
        #if None end at end
        if e== None:
            e=len(label)
        #if end is smaller than len(label)-padding end at end (partly padded)   
        elif e>len(label)-(padding*self.fs) and e<len(label):
            e=len(label)
        #rais errors
        elif e<=s:
            raise Exception("End idx is smaller/equal to start ...")
        elif e>=len(label):
            raise Exception("End idx to large...")
        elif e<0 or s<0:
            raise Exception("Negative indexing is not excepted...")
        #end is end idx + padding
        else:
            e=e+padding*self.fs
            
        return label[int(s):int(e)]
    
    def __remove_padding(self,label,len_orig_label,s,e,padding=15):
        #if None start at beginning
        if s == None:
            s=0
        #if s < padding calculate difference    
        elif s<padding*self.fs:
            s = (padding*self.fs)-s
        #remove full padding
        else: 
            s = (padding*self.fs)
        #if None end at end
        if e== None:
            e=len(label)
        #if e > len(original label)-padding calculate difference    
        elif e>len_orig_label-padding*self.fs:
            e = e-(len_orig_label-padding*self.fs)
        #remove full padding
        else: 
            e = -(padding*self.fs)
        #if not none, original start end end are used    
        return label[int(s):int(e)]
        
    def __find_consecutive_numbers(self,nums):
        #returns nx2 array with starting and ending points of consecutive numbers
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s+1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return np.array(list(zip(edges, edges)))   
    
    def __find_events(self,event_array,label_list):
        """
        pre-allocate array
        """
        events = np.array([0,0])
        events_labels = 0
        #loop over positive classes and add to arrays
        for lab in label_list:
            #find positive label
            events_temp = self.__find_consecutive_numbers(np.where(event_array==lab)[0])
            #add to array
            if events_temp.shape[0]!=0:
                events = np.vstack((events,events_temp))
                events_labels = np.append(events_labels,np.ones(len(events_temp))*lab)
        
        #remove 1st index since it was used for pre-allocation,
        if len(events.shape)>1:
            events = events[1:,:]
            events_labels = events_labels[1:]
        else:
            events = []
            events_labels = []

        return events,events_labels
    
    def __add_before_event(self,lab,diff,sec=2):
        #adds 2 seconds of 1's before all non-zero events 
        #if different events occur consecuativly, it is seen as one event
        start_events = np.where(diff>0)[0]
        for idx in start_events:
            lab[np.max((0,idx-(sec*self.fs)+1)):idx+1]=1 
        return lab
            
    def __add_after_event(self,lab,diff,sec=10):
        #adds 10 seconds of 1's after all non-zero events 
        #if different events occur consecuativly, it is seen as one event
        end_events = np.where(diff<0)[0]
        for idx in end_events:
            lab[idx+1:np.min((idx+(sec*self.fs)+1,len(lab)))]=1 
        return lab
    
    def __add_before_event_type(self,lab,IDX,sec=15):
        #adds 15 seconds of 1's before all specific events 
        #NOTE: this function is calibrated on the __find_consecutive_numbers function.
        #if used otherwise, 1 sample offset might occur
        for idx in IDX:
            lab[np.max((0,idx-(sec*self.fs))):idx]=1 
        return lab
    
    def __add_after_event_type(self,lab,IDX,sec=15):
        #adds 15 seconds of 1's after all specific events
        #NOTE: this function is calibrated on the __find_consecutive_numbers function.
        #if used otherwise, 1 sample offset might occur
        for idx in IDX:
            lab[idx+1:np.min((idx+(sec*self.fs)+1,len(lab)))]=1 
        return lab
    
    def __unstable_sleep_vector_event(self,label,sec_end=10):

        #create unstable sleep vector based on event (event included)
        usv_x = label.copy()
        #binarize 1=event, 0=no event
        usv_x[usv_x>1]=1
        usv_x[usv_x<1]=0
        #find start and end of event
        diff_label = np.diff(usv_x)
        usv_x = self.__add_after_event(usv_x,diff_label,sec=sec_end)
        return usv_x
           
    def __uncertainty_vector_event(self,label,sec_start=2,sec_end=10):
        
        #create unstable sleep vector based on event
        usv_x = np.zeros(len(label))
        #find start and end of event
        diff_label = np.diff(label)
        usv_x = self.__add_before_event(usv_x,diff_label,sec=sec_start)
        usv_x = self.__add_after_event(usv_x,diff_label,sec=sec_end)
        return usv_x
        
    def __build_final_pp_label(self,hypno,Label,list_usv,list_lwu):
        
        #copy label, otherwise orig label is changed
        label = Label.copy()
        #check if data is there
        if not np.array(label).size==0 and np.array(label).shape == np.array(hypno).shape:
            
            
            #first create uncertainty labels
            for lwu in list_lwu:
               if not np.array(lwu).size==0:
                    label[np.where(lwu!=0)]=self.uncertainty_label
            #secondly, create unstable sleep labels. 
            #since unstable sleep labels needs to overwrite uncertainty labeling
            
            for usv in list_usv:
                if not np.array(usv).size==0:
                    #get uncertainty periods as events
                    usv_events = self.__find_events(usv,[1])[0]
                    lab_events = self.__find_events(Label,[1])[0]
                    overlap_events=[]
                    
                    if len(lab_events)>0:
                        for S,E in usv_events:
                            #look for overlab
                            overlap_idx = np.where(((lab_events[:,0]-S<0) & (lab_events[:,1]-S>0)) | ((lab_events[:,0]-E<0) & (lab_events[:,1]-E>0)) | ((lab_events[:,0]-S>=0) & (lab_events[:,1]-E<=0)))[0]
                            if len(overlap_idx) != 0:
                                for o in overlap_idx:
                                    overlap_events.append(o)
                        
                        overlap_events = np.unique(overlap_events)

                        #only remove if overlap is bigger than x%
                        to_rem = []
                        for i,o in enumerate(overlap_events):
                            if sum(usv[lab_events[o,0]:lab_events[o,1]]) > (lab_events[o,1]-lab_events[o,0])*0.90:
                                to_rem.append(i)
                            else:
                                pass

                        #remove events with overlap with uncertainty
                        for overlap_event in overlap_events[to_rem]:
                            label[lab_events[overlap_event,0]:lab_events[overlap_event,1]]=self.uncertainty_label

                        #make sure, full uncertainty period is masked
                        label[np.where(usv!=0)]=self.unstable_sleep_label
            
            #find wake
            wakefullness_idx = np.where(hypno==self.wake_label)[0]
            nan_idx = np.where(np.isnan(hypno))[0]
            nan_idx2 = np.where(hypno==self.nan_label_hypno)[0]
            #remove events while in wake
            label[wakefullness_idx]=self.uncertainty_label
            label[nan_idx]=self.uncertainty_label
            label[nan_idx2]=self.uncertainty_label
            return label
             
        else:
            return []

    ##################################################
    ############## Callable Definitions ##############
    ##################################################
    
    @classmethod
    def set_mask(cls,int):
        cls.uncertainty_label = int
        cls.unstable_sleep_label = int

    @classmethod
    def set_wake_label(cls,label):
        #to chance wake label if needed
        cls.wake_label=label

    @classmethod
    def set_nan_label_hypno(cls,label):
        #change additional nan/unknown label in hypnogram
        cls.nan_label_hypno =label
    
    @classmethod
    def set_add_after_wake(cls,int):
        #to chance wake label if needed
        cls.add_after_wake=int

    @classmethod
    def set_add_before_wake(cls,int):
        #to chance wake label if needed
        cls.add_before_wake=int
    
    @classmethod
    def set_add_prior_to_event_arousal(cls,int):
        #to chance uncertainty seconds prior to event if needed
        cls.add_prior_event_arousal=int

    @classmethod
    def set_add_prior_to_event_limp(cls,int):
        #to chance uncertainty seconds prior to event if needed
        cls.add_prior_event_limp=int

    @classmethod
    def set_add_prior_to_event_resp(cls,int):
        #to chance uncertainty seconds prior to event if needed
        cls.add_prior_event_resp=int
        
    @classmethod
    def set_add_after_event_arousal(cls,int):
        #to chance uncertainty seconds after to event if needed
        cls.add_after_event_arousal=int

    @classmethod
    def set_add_after_event_limp(cls,int):
        #to chance uncertainty seconds after to event if needed
        cls.add_after_event_limp=int

    @classmethod
    def set_add_after_event_resp(cls,int):
        #to chance uncertainty seconds after to event if needed
        cls.add_after_event_resp=int
    
    @classmethod
    def set_uncertainty_label(cls,int):
        #to chance uncertainty seconds after event if needed
        cls.uncertainty_label=int
        
    @classmethod
    def set_unstable_sleep_label(cls,int):
        #to chance uncertainty seconds after event if needed
        cls.unstable_sleep_label=int
    

    def get_unstable_sleep_vector_hypno(self,start_idx=None,end_idx=None,continue_if_missing=False):
        str_ = 'hypno'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)

        if proceed:
            #create unstable sleep vector based on event
            label = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            usv_h = np.zeros(len(label))
            #find wake edges
            consecutive_list = self.__find_consecutive_numbers(np.where(label==self.wake_label)[0])
            if not consecutive_list.size==0:
                usv_h = self.__add_before_event_type(usv_h,consecutive_list[:,0],sec=self.add_before_wake)
                usv_h = self.__add_after_event_type(usv_h,consecutive_list[:,1],sec=self.add_after_wake)
            else:    
                print('#####################################')
                print('Wakefulness not found in given segment')
                print('#####################################')
            usv_h = self.__remove_padding(usv_h,len(self.label_dict[str_]),start_idx,end_idx)
            return usv_h
        else:
            return []
        
    def get_unstable_sleep_vector_arousal(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # add 10 seconds of unstable sleep adjecent to event. 
        # event is included in the unstable sleep
        str_ = 'arousal'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            usv = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            usv = self.__unstable_sleep_vector_event(usv,sec_end=10)
            usv = self.__remove_padding(usv,len(self.label_dict[str_]),start_idx,end_idx)
            return usv
        else:
            return []
        
    def get_unstable_sleep_vector_limp(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # add 10 seconds of unstable sleep adjecent to event. 
        # event is included in the unstable sleep
        str_ = 'limp'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            usv = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            usv = self.__unstable_sleep_vector_event(usv,sec_end=10)
            usv = self.__remove_padding(usv,len(self.label_dict[str_]),start_idx,end_idx)
            return usv
        else:
            return []
    
    def get_unstable_sleep_vector_resp(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # add 10 seconds of unstable sleep adjecent to event. 
        # event is included in the unstable sleep
        str_ = 'resp'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            usv = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            usv = self.__unstable_sleep_vector_event(usv,sec_end=10)
            usv = self.__remove_padding(usv,len(self.label_dict[str_]),start_idx,end_idx)
            return usv
        else:
            return []
    
    def get_arousal_uncertainty_label(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # 2 seconds prior to the event and 10 after event are classified as uncertain. 
        # event is NOT included in the uncertainty
        str_ = 'arousal'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            masked_label = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            masked_label = self.__uncertainty_vector_event(masked_label,sec_start=self.add_prior_event_arousal,sec_end=self.add_after_event_arousal)
            masked_label = self.__remove_padding(masked_label,len(self.label_dict[str_]),start_idx,end_idx)
            return masked_label
        else:
            return []
    
    def get_limp_uncertainty_label(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # 2 seconds prior to the event and 10 after event are classified as uncertain. 
        # event is NOT included in the uncertainty
        str_ = 'limp'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            masked_label = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            masked_label = self.__uncertainty_vector_event(masked_label,sec_start=self.add_prior_event_limp,sec_end=self.add_after_event_limp)
            masked_label = self.__remove_padding(masked_label,len(self.label_dict[str_]),start_idx,end_idx)
            return masked_label
        else:
            return []
    
    def get_resp_uncertainty_label(self,start_idx=None,end_idx=None,continue_if_missing=False):
        # 2 seconds prior to the event and 10 after event are classified as uncertain. 
        # event is NOT included in the uncertainty
        str_ = 'resp'
        proceed = self.__check_availability(str_,continue_if_missing=continue_if_missing)
        if proceed:
            masked_label = self.__cut_label_with_padding(self.label_dict[str_],start_idx,end_idx,padding=15)
            masked_label = self.__uncertainty_vector_event(masked_label,sec_start=self.add_prior_event_resp,sec_end=self.add_after_event_resp)
            masked_label = self.__remove_padding(masked_label,len(self.label_dict[str_]),start_idx,end_idx)
            return masked_label
        else:
            return []
    
    def get_hypno(self,start_idx=None,end_idx=None):
        #retuns unalterd hypnogram
        if 'hypno' not in self.label_dict:
            raise Exception("No hypno data available... ")
        else:
            return self.__cut_label_with_padding(self.label_dict['hypno'],start_idx,end_idx,padding=0)
        
    def get_arousal_label(self,start_idx=None,end_idx=None):
        #retuns unalterd hypnogram
        if 'arousal' not in self.label_dict:
            raise Exception("No arousal data available... ")
        else:
            return self.__cut_label_with_padding(self.label_dict['arousal'],start_idx,end_idx,padding=0)
        
    def get_limp_label(self,start_idx=None,end_idx=None):
        #retuns unalterd hypnogram
        if 'limp' not in self.label_dict:
            raise Exception("No limp data available... ")
        else:
            return self.__cut_label_with_padding(self.label_dict['limp'],start_idx,end_idx,padding=0)
        
    def get_resp_label(self,start_idx=None,end_idx=None):
        #retuns unalterd hypnogram
        if 'resp' not in self.label_dict:
            raise Exception("No resp data available... ")
        else:
            return self.__cut_label_with_padding(self.label_dict['resp'],start_idx,end_idx,padding=0)
        
    def get_masked_arousal_label(self,start_idx=None,end_idx=None,continue_if_missing=True):
        
        #get labels
        hypno = self.__grab_label_if_available('hypno',start_idx,end_idx)
        arousal =self.__grab_label_if_available('arousal',start_idx,end_idx)

        
        #get individual unstable sleep vectors
        usv_h = self.get_unstable_sleep_vector_hypno(start_idx,end_idx,continue_if_missing=continue_if_missing)
        usv_l = self.get_unstable_sleep_vector_limp(start_idx,end_idx,continue_if_missing=continue_if_missing)
        usv_r = self.get_unstable_sleep_vector_resp(start_idx,end_idx,continue_if_missing=continue_if_missing)
        
        #get individual uncertainty labels
        lwu_a = self.get_arousal_uncertainty_label(start_idx,end_idx,continue_if_missing=continue_if_missing)*self.uncertainty_label

        #build final masked labeling
        masked_label_arousal = self.__build_final_pp_label(hypno,arousal,[usv_h,usv_l,usv_r],[lwu_a])

        return masked_label_arousal

    def get_masked_arousal_label_eval(self,start_idx=None,end_idx=None,continue_if_missing=True):
        
        #get labels
        hypno = self.__grab_label_if_available('hypno',start_idx,end_idx)
        arousal =self.__grab_label_if_available('arousal',start_idx,end_idx)

        
        #get individual unstable sleep vectors
        usv_h = self.get_unstable_sleep_vector_hypno(start_idx,end_idx,continue_if_missing=continue_if_missing)
        
        #build final masked labeling
        masked_label_arousal = self.__build_final_pp_label(hypno,arousal,[usv_h],[])

        return masked_label_arousal


    def get_masked_resp_label(self,start_idx=None,end_idx=None,continue_if_missing=True):
        
        #get labels
        hypno = self.__grab_label_if_available('hypno',start_idx,end_idx)
        resp =self.__grab_label_if_available('resp',start_idx,end_idx)

        #get individual uncertainty labels
        lwu_r = self.get_resp_uncertainty_label(start_idx,end_idx,continue_if_missing=continue_if_missing)*self.uncertainty_label
        
        #build final label
        masked_label_resp = self.__build_final_pp_label(hypno,resp,[],[])
    
        return masked_label_resp

    def get_masked_limp_label(self,start_idx=None,end_idx=None,continue_if_missing=True):
        
        #get labels
        hypno = self.__grab_label_if_available('hypno',start_idx,end_idx)
        limp =self.__grab_label_if_available('limp',start_idx,end_idx)
        
        #get individual unstable sleep vectors
        usv_h = self.get_unstable_sleep_vector_hypno(start_idx,end_idx,continue_if_missing=continue_if_missing)
        usv_r = self.get_unstable_sleep_vector_resp(start_idx,end_idx,continue_if_missing=continue_if_missing)

        #build final masked label 
        masked_label_limp = self.__build_final_pp_label(hypno,limp,[usv_h,usv_r],[])
  
        return masked_label_limp


    def get_masked_labels(self,start_idx=None,end_idx=None,continue_if_missing=True):
        
        #get labels
        hypno = self.__grab_label_if_available('hypno',start_idx,end_idx)
        arousal =self.__grab_label_if_available('arousal',start_idx,end_idx)
        limp =self.__grab_label_if_available('limp',start_idx,end_idx)
        resp =self.__grab_label_if_available('resp',start_idx,end_idx)
        
        #get individual unstable sleep vectors
        usv_h = self.get_unstable_sleep_vector_hypno(start_idx,end_idx,continue_if_missing=continue_if_missing)
        usv_l = self.get_unstable_sleep_vector_limp(start_idx,end_idx,continue_if_missing=continue_if_missing)
        usv_r = self.get_unstable_sleep_vector_resp(start_idx,end_idx,continue_if_missing=continue_if_missing)
        
        #get individual uncertainty labels
        lwu_a = self.get_arousal_uncertainty_label(start_idx,end_idx,continue_if_missing=continue_if_missing)*self.uncertainty_label
        lwu_r = self.get_resp_uncertainty_label(start_idx,end_idx,continue_if_missing=continue_if_missing)*self.uncertainty_label
        
        #build final masked label 
        masked_label_arousal = self.__build_final_pp_label(hypno,arousal,[usv_h,usv_l,usv_r],[lwu_a])
        masked_label_limp = self.__build_final_pp_label(hypno,limp,[usv_h,usv_r],[])
        masked_label_resp = self.__build_final_pp_label(hypno,resp,[],[lwu_r])
        
        return masked_label_arousal,masked_label_limp,masked_label_resp

       
                