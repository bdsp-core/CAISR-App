import numpy as np
import h5py as h5 
from arousal.utils.post_processing.label_class import *

##########
#load data 
##########
path = '/media/mad3/Projects_New/zz_SLEEP/CAISR1/data_prepared/robert/Study 97 JD051021.h5'

f = h5.File(path, 'r')
hypno = np.array(f.get('Xy')[17,:((f.get('Xy').shape[1]//6000)*6000)])
arousal = np.array(f.get('Xy')[18,:((f.get('Xy').shape[1]//6000)*6000)])
resp = np.array(f.get('Xy')[19,:((f.get('Xy').shape[1]//6000)*6000)])
limp = np.array(f.get('Xy')[20,:((f.get('Xy').shape[1]//6000)*6000)])





##############################
######### PART THIJS #########
##############################
input_dict_thijs = {'hypno':hypno,
              'resp':resp
              }
pat_label_thijs= CAISER_labels(input_dict_thijs,fs=200)
pat_label_thijs.set_mask(7)
masked_resp = pat_label_thijs.get_masked_resp_label()

##############################
######### PART SAMANEH #########
##############################
#with all labels
input_dict = {'hypno':hypno,
              'arousal':arousal,
              'limp':limp,
              'resp':resp
              }
pat_label = CAISER_labels(input_dict,fs=200)
pat_label.set_mask(2)
masked_resp = pat_label.get_masked_arousal_label()
resp_ = pat_label.get_arousal_label()#to check




##########
#callables
##########

#RETURN UNSTABLE SLEEP VECTOR
# this is a vector that indicates unstable sleep based on 1 event type. 
# rules can be altered sepereately per event 
# rules are by default: 
# unstable sleep is: 15 second before and after wakefulness, & during and 10 after event
# unstable sleep is returned as 1, stable sleep AND WAKEFULNESS are returned as 0.
usv_a = pat_label.get_unstable_sleep_vector_arousal()
usv_l = pat_label.get_unstable_sleep_vector_limp()
usv_h = pat_label.get_unstable_sleep_vector_hypno()
usv_r = pat_label.get_unstable_sleep_vector_resp()

#Callables within UNSTABLE SLEEP VECTOR
#continue_if_missing=True
#if resp signal is not given in dict, function will not raise an error, but will notice you in terminal
usv_r_empty = pat_label_thijs.get_unstable_sleep_vector_resp(continue_if_missing=True) 
#start_idx and or end_idx
#will create indexing. NOTE that if you index yourself, you might get different results, 
# since this indexing uses a buffer of 15 seconds before and after indexing 
# This to make sure that events close to the border are also detected and taken into account in this unstable sleep vector
usv_r_indexed = pat_label.get_unstable_sleep_vector_resp(start_idx=600000,end_idx=600000*2)

#NOTE that changing times before/after event is not event specific. 
# This will be available in later version

#to change 15second wake buffer 
pat_label.set_add_to_wake(15) #seconds
#to change wake label
pat_label.set_wake_label(5) #change to 5
#to change uncertainty prior to event   
pat_label.set_add_prior_to_event_arousal(2)#seconds
#to change uncertainty AND unstable sleep after event 
pat_label.set_add_after_event_arousal(10)#seconds
#to change uncertainty label
pat_label.set_uncertainty_label(-9)#randomly chosen
#to change unstable sleep label
pat_label.set_unstable_sleep_label(-8)#randomly chosen

#Callables within UNCERTAINTY LABEL
# this is a vector that indicates uncertaity of labels. 
# rules can be altered sepereately per event 
# rules are by default: 
# uncertainty: 2 second before event and 10 seconds after
# consecutive events are handled as one.
ul_a = pat_label.get_arousal_uncertainty_label()
ul_r = pat_label.get_resp_uncertainty_label()
ul_l = pat_label.get_limp_uncertainty_label()

#Callables within UNCERTAINTY LABEL
# see Callables within UNSTABLE SLEEP VECTOR
ul_a = pat_label.get_arousal_uncertainty_label(start_idx=600000,end_idx=600000*2,continue_if_missing=True)

#you can also return unaltered labels 
hypno = pat_label.get_hypno()
arousal = pat_label.get_arousal_label()
limp = pat_label.get_limp_label()
resp = pat_label.get_resp_label()

##############
# Post_process labels
##############
# main task!
# post process labels based on given labels. 
# if labels are not present they will not raise an error and will be returned as an empty list
pat_label.set_mask(7)

Pp_label_arousal,Pp_label_limp,Pp_label_resp = pat_label.get_masked_labels()


#okay a small explenation before you watch this complicated figure. 
#first subplot: Blue: returned hypno, Orange: unstable sleep based on hypnogram
#second subplot: Blue: returned arousal label, Orange: post processed arousal label 
#third subplot: Blue: returned arousal limp, Orange: post processed arousal limp 
#fourth subplot: Blue: returned arousal resp, Orange: post processed arousal resp

#for the second third and fourth subplot, multiple scatter plots are added. 
# scatter orange = event in post processed labels
# scatter red = uncertainty event in post processed labels
# scatter green = unstable sleep in post processed labels

# if you zoom in, you will see that unstable sleep events in one channel are caused by events+uncertaincy after event in another channel
# NOTE figure is slow due to scatter plots
 
fig, ax = plt.subplots(4,1,sharex=True)

ax[0].plot(hypno)
ax[0].plot((usv_h*4)+1) 
ax[0].set_title('hypnogram (blue) with unstable sleep based on hypno (orange)')

ax[1].plot(arousal)
ax[1].plot(Pp_label_arousal)
# idx = np.where(Pp_label_arousal==1)[0]
# ax[1].scatter(idx,Pp_label_arousal[idx],color='orange')
# idx = np.where(Pp_label_arousal==-9)[0]
# ax[1].scatter(idx,Pp_label_arousal[idx],color='red')
# idx = np.where(Pp_label_arousal==-8)[0]
# ax[1].scatter(idx,Pp_label_arousal[idx],color='green')
ax[1].set_title('arousal (blue) with post processed arousal (orange), for scatterplot description see script')

ax[2].plot(limp,label='orig label')
ax[2].plot(Pp_label_limp,label='new label')
# idx = np.where(Pp_label_limp==1)[0]
# ax[2].scatter(idx,Pp_label_limp[idx],color='orange',label='event')
# idx = np.where(Pp_label_limp==-9)[0]
# ax[2].scatter(idx,Pp_label_limp[idx],color='red',label='uncertainty')
# idx = np.where(Pp_label_limp==-8)[0]
# ax[2].scatter(idx,Pp_label_limp[idx],color='green',label='unstable sleep')
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[2].set_title('limp (blue) with post processed limp (orange), for scatterplot description see script')

ax[3].plot(resp)
ax[3].plot(Pp_label_resp)
# idx = np.where(Pp_label_resp>0)[0]
# ax[3].scatter(idx,Pp_label_resp[idx],color='orange')
# idx = np.where(Pp_label_resp==-9)[0]
# ax[3].scatter(idx,Pp_label_resp[idx],color='red')
# idx = np.where(Pp_label_resp==-8)[0]
# ax[3].scatter(idx,Pp_label_resp[idx],color='green')
ax[3].set_title('resp (blue) with post processed resp (orange), for scatterplot description see script')

plt.show()
