import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math
# Code taken from https://gist.github.com/PetrochukM/afaa3613a99a8e7213d2efdd02ae4762#file-top_k_viterbi-py-L5
# Credits to AllenNLP for the base implementation and base tests:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L174

# Modified AllenNLP `viterbi_decode` to support `top_k` sequences efficiently.
def viterbi_decode(tag_sequence,transition_matrix,top_k=1):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """

    transition_matrix = transition_matrix.to(tag_sequence.device)

    sequence_length, num_tags = list(tag_sequence.size())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores


def regressionPred2Confidence(regresPred,nbClass):
    #Converting each element of output into a list of confidence for each class
    #Therefore the shape will go from a matrix to a 3D tensor
    output = torch.abs(regresPred.unsqueeze(2)-torch.arange(nbClass).unsqueeze(0).unsqueeze(0).to(regresPred.device).float())
    conf = F.softmax(-output,dim=-1)

    return conf


def emptyMetrDict(uncertainty=False):
    if not uncertainty:
        return {"Loss":0,"Accuracy":0,"Accuracy (Viterbi)":0}
    else:
        return {"Loss":0,"Accuracy":0,"Accuracy (Viterbi)":0,"Entropy (Correct)":None,"Entropy (Incorrect)":None}

def updateMetrDict(metrDict,metrDictSample):

    for metric in metrDict.keys():

        if metric in list(metrDictSample.keys()):
            if metric.find("Entropy") == -1:
                metrDict[metric] += metrDictSample[metric]
            else:
                if metrDict[metric] is None:
                    metrDict[metric] = metrDictSample[metric]
                else:
                    metrDict[metric] = torch.cat((metrDict[metric],metrDictSample[metric]),dim=0)

    return metrDict

def binaryToMetrics(output,target,transition_matrix,regression,uncertainty,videoNames=None):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    #Simple Accuracy
    if regression:
        pred = torch.round(output).long()
        output = regressionPred2Confidence(output,transition_matrix.size(0))
    else:
        pred = output.argmax(dim=-1)

    acc = (pred == target).float().sum()/(pred.numel())

    if torch.isnan(transition_matrix).sum() == 0:

        #Accuracy with viterbi
        pred = []
        for outSeq in output:

            if uncertainty:
                outSeq = F.softplus(outSeq)+1
                outSeq = outSeq/outSeq.sum(dim=-1,keepdim=True)
            else:
                outSeq = torch.nn.functional.softmax(outSeq, dim=-1)

            predSeqs,_ = viterbi_decode(torch.log(outSeq),torch.log(transition_matrix),top_k=1)

            pred.append(torch.tensor(predSeqs[0]).unsqueeze(0))

        pred = torch.cat(pred,dim=0).to(target.device)
        accViterb = (pred == target).float().sum()/(pred.numel())

    else:
        accViterb = 0

    metDict = {"Accuracy":acc,'Accuracy (Viterbi)':accViterb}

    if uncertainty:

        #output,target = output.view(output.size(0)*output.size(1),-1),target.view(-1)

        probs = output/output.sum(dim=-1,keepdim=True)
        classNb = probs.size(-1)
        entropies = -(probs*torch.log(probs)).sum(dim=-1)
        entropies_norm = entropies/math.log(classNb)

        metDict["Entropy (Correct)"] = entropies_norm[pred == target]
        metDict["Entropy (Incorrect)"] = entropies_norm[pred != target]

    if not videoNames is None:
        metDict["Correlation"] = correlation(output,target,videoNames)

    return metDict


def correlation(output,target,videoNames):
    ''' Computes the times at which the model predicts the developpement phase is changing and
    compare it to the real times where the phase is changing. Computes a correlation between those
    two list of numbers.

    '''

    for i,outSet in enumerate(output):

        dataset = load_data.getDataset(videoNames[i])
        timeElapsedTensor = np.genfromtxt("../data/{}/annotations/{}_timeElapsed.csv".format(dataset,videoNames[i]))

        pred = output.argmax(dim=-1)
        phasesPredDict = phaseToTime(pred,timeElapsedTensor)

        phasesTargDict = phaseToTime(target,timeElapsedTensor)

        commonPhases = list(set(list(phasesPredDict.keys())).intersection(set(list(phasesTargDict.keys()))))

        timePairs = []
        for phase in commonPhases:
            timePairs.append((phasesPredDict[phase],phasesTargDict[phase]))

        return timePairs

def phaseToTime(phaseList,timeElapsedTensor):

    changingPhaseFrame = np.cat(([1],(phaseList[1:]-phaseList[:-1]) == 1),axis=0)
    phases = phaseList[changingPhaseFrame]

    changingPhaseFrame = np.argwhere(changingPhaseFrame)[:,0]
    changingPhaseTime = timeElapsedTensor[changingPhaseFrame]

    phaseToFrameDict = {phases[i]:changingPhaseTime[i] for i in range(len(phases))}

    return phaseToFrameDict
