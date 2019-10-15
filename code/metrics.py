import utils
import numpy as np
import torch
import scipy as sp

def binaryToMetrics(pred,target):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - pred (list): the batch of predicted class
    - target (list): the batch of ground truth class

    '''

    acc = (pred == target).sum()/(pred.numel())

    return acc
