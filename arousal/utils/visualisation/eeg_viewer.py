import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(data,label,y_val= 0, color='black',marker='s',alpha='normal',legend_label='_no_show'):
    loc = np.where(data==label)[0]
    loc_ = [np.arange(int(x),int(x+1)) for x in loc]
    loc_ = np.reshape(loc_,-1)
    loc_ = np.append(-100,loc_)
    if alpha == 'normal':
        plt.hlines(-y_val,0,len(data),'black',alpha=0.2,linewidth=0.5)
        plt.scatter(loc_/128,np.zeros(len(loc_))-y_val,color=color,marker=marker,label=legend_label)
    elif alpha == 'masked':
        plt.scatter(loc_/128,np.zeros(len(loc_))-y_val,color=color,marker=marker,alpha=0.005,label=legend_label)
    elif alpha == 'resp':
        plt.scatter(loc_/128,np.zeros(len(loc_))-y_val,color=color,marker=marker,label=legend_label)