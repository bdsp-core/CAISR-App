import tensorflow as tf

def scheduler(epoch, lr):
    # if epoch == 1:
    #     return lr * 0.1
    # else:
    #     return lr
    return lr*0.95
