import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math

import load_data

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,resDict):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    #Simple Accuracy
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()/(pred.numel())

    if "auxPred" in resDict.keys():
        auxPred = resDict["auxPred"].argmax(dim=-1)
        aux_acc = (auxPred == target).float().sum()/(auxPred.numel())

    metDict = {"Accuracy":acc,"Accuracy_aux":aux_acc}

    return metDict
