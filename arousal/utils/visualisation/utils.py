import matplotlib.pyplot as plt

def pan_nav(event):

    ax_tmp = plt.gca()
    if event.key == 'left':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 0.5
        ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
        plt.draw()
    elif event.key == 'right':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 0.5
        ax_tmp.set_xlim((lims[0] + adjust, lims[1] + adjust))
        plt.draw()
    elif event.key == 'down':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 5
        ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
        plt.draw()
    elif event.key == 'up':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 5
        ax_tmp.set_xlim((lims[0] + adjust, lims[1] + adjust))
        plt.draw()

def next_event(event):

    ax_tmp = plt.gca()
    global start_idx_arousals_for_pan
    global arousal_idx_index

    if event.key == 'n':
 
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) 
        s = np.max((start_idx_arousals_for_pan[arousal_idx_index]/128-15,0))
        e = s+adjust
        ax_tmp.set_xlim((s,e))
        arousal_idx_index +=1
        plt.draw()
    
    elif event.key == 'b':

        arousal_idx_index -=2
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) 
        s = np.max((start_idx_arousals_for_pan[arousal_idx_index]/128-15,0))
        e = s+adjust
        ax_tmp.set_xlim((s,e))
        arousal_idx_index +=1
        plt.draw()

    elif event.key == 'left':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 0.5
        ax_tmp.set_xlim((lims[0] - adjust, lims[1] - adjust))
        plt.draw()

    elif event.key == 'right':
        lims = ax_tmp.get_xlim()
        adjust = (lims[1] - lims[0]) * 0.5
        ax_tmp.set_xlim((lims[0] + adjust, lims[1] + adjust))
        plt.draw()