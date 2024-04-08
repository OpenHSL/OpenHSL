from collections import OrderedDict
from typing import List


def get_mean_weights(weights_list: List[OrderedDict]) -> OrderedDict:
    """
    mean_weights(weights_list)

    Returns calculated mean weights of models weights list

    Arguments
    ---------
        weights_list - list of models weights
    """
    mean_weight = dict()

    for key in weights_list[0].keys():
        mean_weight[key] = sum([w[key] for w in weights_list]) / len(weights_list)

    return OrderedDict(mean_weight)
# ----------------------------------------------------------------------------------------------------------------------
