from sklearn.metrics import confusion_matrix
import sklearn.metrics as metrics
import numpy as np
import itertools
import matplotlib.pyplot as plt


def evaluations_matrix(gs, pred):
    fpr, tpr, thresholds = metrics.roc_curve(gs, pred, pos_label=1)
    auroc = metrics.auc(fpr, tpr)
    auprc = metrics.average_precision_score(gs, pred)
    return auroc, auprc

def evaluations_matrix_binary(gs, pred):
    """
    Params
    ------
    gs: gold standards
    pred: predicted binary labels 
    Yields
    ------
    sensitivity
    specificity
    precision
    F1-score
    auprc_baseline: P/N in all samples
    """
    # kappa = metrics.cohen_kappa_score(gs, pred)
    conf_mat = confusion_matrix(gs, pred)
    tn, fp, fn, tp = conf_mat.ravel()
    auprc_baseline = (fn+tp)/(tn+fp+fn+tp)
    sensitivity = tp/(tp+fn) # recall
    specificity =tn/(tn+fp)
    precision = tp/(tp+fp)
    accuracy = (tn+tp)/(tn+fp+fn+tp)
    F1= 2*(sensitivity*precision)/(sensitivity+precision)
    return conf_mat, sensitivity, specificity, precision, accuracy, F1, auprc_baseline

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    


    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))