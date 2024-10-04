#from tensorflow_addons import optimizers as addon_optimizers
from tensorflow_addons import activations as addon_activations
from tensorflow.keras import optimizers, losses, metrics, activations

def ensure_list_or_tuple(obj):
    return [obj] if not isinstance(obj, (list, tuple)) else obj

def _init(string_list, tf_funcs, custom_funcs, logger=None, **kwargs):
    """
    Helper for 'init_losses' or 'init_metrics'.
    Please refer to their docstrings.

    Args:
        string_list:  (list)   List of strings, each giving a name of a metric
                               or loss to use for training. The name should
                               refer to a function or class in either tf_funcs
                               or custom_funcs modules.
        tf_funcs:     (module) A Tensorflow.keras module of losses or metrics,
                               or a list of various modules to look through.
        custom_funcs: (module) A custom module or losses or metrics
        logger:       (Logger) A Logger object
        **kwargs:     (dict)   Parameters passed to all losses or metrics which
                               are represented by a class (i.e. not a function)

    Returns:
        A list of len(string_list) of initialized classes of losses or metrics
        or references to loss or metric functions.
    """
    initialized = []
    tf_funcs = ensure_list_or_tuple(tf_funcs)
    for func_or_class in ensure_list_or_tuple(string_list):
        modules_found = list(filter(None, [getattr(m, func_or_class, None)
                                           for m in tf_funcs]))
        if modules_found:
            initialized.append(modules_found[0])  # return the first found
        else:
            # Fall back to look in custom module
            initialized.append(getattr(custom_funcs, func_or_class))
    return initialized

def init_activation(activation_string, logger=None, **kwargs):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    activation = _init(
        activation_string,
        tf_funcs=[activations, addon_activations],#tf_funcs=[activations, addon_activations]
        custom_funcs=None,
        logger=logger
    )[0]
    return activation

