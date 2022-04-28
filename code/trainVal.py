from multiprocessing import Value
from models import inter_by_parts
import os
import sys
import glob

import args
from args import ArgReader
from args import str2bool

import numpy as np
import torch
from torch.nn import functional as F

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import matplotlib.pyplot as plt

plt.switch_backend('agg')

import modelBuilder
import load_data
import metrics
import utils
import update
from gradcam import GradCAMpp
from score_map import ScoreCam
from rise import RISE
from xgradcam import XGradCAM,AblationCAM

import time

import configparser

import optuna
import sqlite3

from shutil import copyfile

import gc

import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
from models import protopnet 

import captum
from captum.attr import (IntegratedGradients,NoiseTunnel)

OPTIM_LIST = ["Adam", "AMSGrad", "SGD"]

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def epochSeqTr(model, optim, log_interval, loader, epoch, args, **kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network
    '''

    start_time = time.time() if args.debug or args.benchmark else None

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0
    gpu = kwargs["gpu"]

    if args.grad_exp:
        allGrads = None

    acc_size = 0
    acc_nb = 0

    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if batch_idx % log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]

        if acc_size + data.size(0) > args.batch_size:

            if args.batch_size-acc_size < 2*torch.cuda.device_count():
                data = data[:2*torch.cuda.device_count()]
                target = target[:2*torch.cuda.device_count()]
            else:
                data = data[:args.batch_size-acc_size]
                target = target[:args.batch_size-acc_size]
            acc_size = args.batch_size
        else:
            acc_size += data.size(0)
        acc_nb += 1

        if args.with_seg:
            seg = batch[2]
        else:
            seg = None

        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            if args.with_seg:
                seg = seg.cuda(non_blocking=True)

        resDict = model(data)
        output = resDict["pred"]

        if args.master_net:
            with torch.no_grad():
                mastDict = kwargs["master_net"](data)
                resDict["master_net_pred"] = mastDict["pred"]
                resDict["master_net_attMaps"] = mastDict["attMaps"]
                resDict["master_net_features"] = mastDict["features"]

        loss = kwargs["lossFunc"](output, target, resDict, data).mean()
        loss.backward()

        if acc_size == args.batch_size:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= acc_nb
            optim.step()
            optim.zero_grad()
            acc_size = 0
            acc_nb = 0

        loss = loss.detach().data.item()

        if args.grad_exp:
            allGrads = updateGradExp(model,allGrads)

        optim.step()

        # Metrics
        with torch.no_grad():
            metDictSample = metrics.binaryToMetrics(output, target, seg,resDict)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch > 3 and args.debug:
            break

    if args.grad_exp and gpu == 0:
        updateGradExp(model,allGrads,True,epoch,args.exp_id,args.model_id,args.grad_exp)

    # If the training set is empty (which we might want to just evaluate the model), then allOut and allGT will still be None
    if validBatch > 0 and gpu==0:

        if not args.optuna:
            torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
            writeSummaries(metrDict, totalImgNb, epoch, "train", args.model_id, args.exp_id)

        if args.debug or args.benchmark:
            totalTime = time.time() - start_time
            update.updateTimeCSV(epoch, "train", args.exp_id, args.model_id, totalTime, batch_idx)

def updateGradExp(model,allGrads,end=False,epoch=None,exp_id=None,model_id=None,grad_exp=None,conv=True):

    if not end:
        if conv:
            newGrads = model.firstModel.featMod.layer4[-1].conv1.weight.grad.data
            norm_conv1 = torch.sqrt(torch.pow(newGrads.view(-1),2).sum()).unsqueeze(0)

            newGrads = model.firstModel.featMod.layer4[-1].conv2.weight.grad.data
            norm_conv2 = torch.sqrt(torch.pow(newGrads.view(-1),2).sum()).unsqueeze(0)

            newGrads = model.firstModel.featMod.layer4[-1].conv3.weight.grad.data
            norm_conv3 = torch.sqrt(torch.pow(newGrads.view(-1),2).sum()).unsqueeze(0)

            if allGrads is None:
                allGrads = {"conv1":norm_conv1,"conv2":norm_conv2,"conv3":norm_conv3}
            else:
                allGrads["conv1"] = torch.cat((allGrads["conv1"],norm_conv1),dim=0)
                allGrads["conv2"] = torch.cat((allGrads["conv2"],norm_conv2),dim=0)
                allGrads["conv3"] = torch.cat((allGrads["conv3"],norm_conv3),dim=0)

        else:
            newGrads = model.secondModel.linLay.weight.grad.data.unsqueeze(0)

            if allGrads is None:
                allGrads = newGrads
            else:
                allGrads = torch.cat((allGrads,newGrads),dim=0)

        return allGrads

    else:
        if conv:
            torch.save(allGrads["conv1"].float(),"../results/{}/{}_allGradsConv1_{}HypParams_epoch{}.th".format(exp_id,model_id,grad_exp,epoch))
            torch.save(allGrads["conv2"].float(),"../results/{}/{}_allGradsConv2_{}HypParams_epoch{}.th".format(exp_id,model_id,grad_exp,epoch))
            torch.save(allGrads["conv3"].float(),"../results/{}/{}_allGradsConv3_{}HypParams_epoch{}.th".format(exp_id,model_id,grad_exp,epoch))
        else:
            torch.save(allGrads.float(),"../results/{}/{}_allGrads_{}HypParams_epoch{}.th".format(exp_id,model_id,grad_exp,epoch))

class Loss(torch.nn.Module):

    def __init__(self,args,reduction="mean"):
        super(Loss, self).__init__()
        self.args = args
        self.reduction = reduction

    def forward(self,output,target,resDict,data):
        return computeLoss(self.args,output, target, resDict, data,reduction=self.reduction).unsqueeze(0)

def computeLoss(args, output, target, resDict, data,reduction="mean"):

    if not args.master_net:
        loss = args.nll_weight * F.cross_entropy(output, target,reduction=reduction)

        if args.inter_by_parts:
            loss += 0.5*inter_by_parts.shapingLoss(resDict["attMaps"],args.resnet_bil_nb_parts,args)

        if args.abn:
            loss += args.nll_weight*F.cross_entropy(resDict["att_outputs"], target,reduction=reduction)

    else:
        kl = F.kl_div(F.log_softmax(output/args.kl_temp, dim=1),F.softmax(resDict["master_net_pred"]/args.kl_temp, dim=1),reduction="batchmean")
        ce = F.cross_entropy(output, target)
        loss = args.nll_weight*(kl*args.kl_interp*args.kl_temp*args.kl_temp+ce*(1-args.kl_interp))

        if args.transfer_att_maps:
            loss += args.att_weights*computeAttDiff(args.att_term_included,args.att_term_reg,resDict["attMaps"],resDict["features"],resDict["master_net_attMaps"],resDict["master_net_features"])

    for key in resDict.keys():
        if key.find("pred_") != -1:
            loss += args.nll_weight * F.cross_entropy(resDict[key], target)

    loss = loss

    return loss

def computeAttDiff(att_term_included,att_term_reg,studMaps,studFeat,teachMaps,teachFeat,attPow=2):

    studNorm = torch.sqrt(torch.pow(studFeat,2).sum(dim=1,keepdim=True))
    teachNorm = torch.sqrt(torch.pow(teachFeat,2).sum(dim=1,keepdim=True))

    if att_term_included:
        studMaps = normMap(studMaps,minMax=True)*normMap(studNorm)
        teachMaps = normMap(teachMaps,minMax=True)*normMap(teachNorm)

        kerSize = studMaps.size(-1)//teachMaps.size(-1)
        studMaps = F.max_pool2d(studMaps,kernel_size=kerSize,stride=kerSize)
        term = torch.pow(torch.abs(teachMaps-studMaps),attPow)
        term *= (1-teachMaps)
        term = term.sum(dim=(2,3)).mean()

    elif att_term_reg:
        studMaps = normMap(studMaps,minMax=True)*normMap(studNorm,minMax=True)
        teachMaps = normMap(teachMaps,minMax=True)*normMap(teachNorm,minMax=True)

        teachX,teachY = softCoord(teachMaps)
        ratio = studMaps.size(-1)//teachMaps.size(-1)
        teachX,teachY = teachX*ratio,teachY*ratio
        studX,studY = softCoord(studMaps)

        term = torch.sqrt(torch.pow(studX-teachX,2)+torch.pow(studY-teachY,2)).mean()

    else:
        studMaps = normMap(studMaps,minMax=True)*normMap(studNorm)
        teachMaps = normMap(teachMaps,minMax=True)*normMap(teachNorm)

        teachMaps = F.interpolate(teachMaps,size=(studMaps.size(-2),studMaps.size(-1)),mode='bilinear',align_corners=True)
        term = torch.pow(torch.pow(torch.abs(teachMaps-studMaps),attPow).sum(dim=(2,3)),1.0/attPow).mean()

    return term

def softCoord(maps):

    x = torch.arange(maps.size(3)).unsqueeze(0).expand(maps.size(2),-1).to(maps.device)
    y = torch.arange(maps.size(2)).unsqueeze(1).expand(-1,maps.size(3)).to(maps.device)

    valX = (x.unsqueeze(0).unsqueeze(0)*maps).sum(dim=(2,3))
    valX /= maps.sum(dim=(2,3))

    valY = (y.unsqueeze(0).unsqueeze(0)*maps).sum(dim=(2,3))
    valY /= maps.sum(dim=(2,3))

    return valX,valY

def normMap(map,minMax=False):
    if not minMax:
        max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        map = map/max
    else:
        max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        min = map.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
        map = (map-min)/(max-min)

    return map

def epochImgEval(model, log_interval, loader, epoch, args, metricEarlyStop, mode="val",**kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network

    '''

    print("../results/{}/model{}_epoch{}_metrics_{}.csv".format(args.exp_id, args.model_id, epoch, mode))

    if args.debug or args.benchmark:
        start_time = time.time()

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    intermVarDict = {"fullAttMap": None, "fullFeatMapSeq": None, "fullNormSeq":None}
    gpu = kwargs["gpu"]

    compute_latency = args.compute_latency and mode == "test"

    if mode=="test" and args.grad_exp_test:
        allGrads = None

    if compute_latency:
        latency_list=[]
        batchSize_list = []
    else:
        latency_list,batchSize_list =None,None

    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        if args.with_seg:
            seg=batch[2]
            path_list = None
        elif args.dataset_test.find("emb") != -1:
            seg = None
            path_list = batch[2]
        else:
            seg=None
            path_list = None

        # Puting tensors on cuda
        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

            if args.with_seg:
                seg = seg.cuda(non_blocking=True)

        # Computing predictions
        if compute_latency:
            lat_start_time = time.time()
            resDict = model(data)
            latency_list.append(time.time()-lat_start_time)
            batchSize_list.append(data.size(0))
        else:
            resDict = model(data)

        output = resDict["pred"]

        if args.master_net:
            mastDict = kwargs["master_net"](data)
            resDict["master_net_pred"] = mastDict["pred"]
            resDict["master_net_attMaps"] = mastDict["attMaps"]
            resDict["master_net_features"] = mastDict["features"]

        # Loss
        if not (mode=="test" and args.grad_exp_test):
            loss = 0
        else:
            loss = kwargs["lossFunc"](output, target, resDict, data).mean()

        # Other variables produced by the net
        if mode == "test":
            resDict["norm"] = torch.sqrt(torch.pow(resDict["features"],2).sum(dim=1,keepdim=True))
            intermVarDict = update.catIntermediateVariables(resDict, intermVarDict, validBatch)

        # Harware occupation
        if gpu == 0:
            update.updateHardWareOccupation(args.debug, args.benchmark, args.cuda, epoch, mode, args.exp_id, args.model_id,
                                        batch_idx)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target, seg,resDict,comp_spars=(mode=="test") and args.with_seg)

        if mode=="test" and args.grad_exp_test:
            loss.backward()

            newGrads = model.secondModel.linLay.weight.grad.data.float().cpu().unsqueeze(0)

            if allGrads is None:
                allGrads = newGrads
            else:
                allGrads = torch.cat((allGrads,newGrads),dim=0)

            model.zero_grad()

        if (mode=="test" and args.grad_exp_test):
            metDictSample["Loss"] = loss.detach().data.item()
        else:
            metDictSample["Loss"] = loss

        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        if mode == "test" and args.dataset_test.find("emb") != -1:
            writePreds(output, target, epoch, args.exp_id, args.model_id, args.class_nb, batch_idx,mode,path_list)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch  >= 4*(50.0/args.val_batch_size) and args.debug:
            break

    if mode == "test":

        if args.att_metrics_post_hoc:
            suff = "_"+args.att_metrics_post_hoc
        else:
            suff = ""

        intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id+suff, epoch, mode)

    writeSummaries(metrDict, totalImgNb, epoch, mode, args.model_id, args.exp_id)

    if mode == "test" and args.grad_exp_test and gpu == 0:
        allGrads = allGrads.view(allGrads.size(0),-1)
        mean = allGrads.mean(dim=0)
        std = allGrads.std(dim=0)
        var = std*std
        snr = (mean/var).mean(dim=0)
        with open("../results/{}/snr_{}.csv".format(args.exp_id,args.model_id),"a") as text_file:
            print("{},{},{}".format(args.trial_id,snr,metrDict["Accuracy"]),file=text_file)

    if compute_latency and gpu == 0:
        latency_list = np.array(latency_list)[:,np.newaxis]
        batchSize_list = np.array(batchSize_list)[:,np.newaxis]
        latency_list = np.concatenate((latency_list,batchSize_list),axis=1)
        np.savetxt("../results/{}/latency_{}_epoch{}.csv".format(args.exp_id,args.model_id,epoch),latency_list,header="latency,batch_size",delimiter=",")

    if (args.debug or args.benchmark) and gpu == 0:
        totalTime = time.time() - start_time
        update.updateTimeCSV(epoch, mode, args.exp_id, args.model_id, totalTime, batch_idx)

    return metrDict[metricEarlyStop]

def writePreds(predBatch, targBatch, epoch, exp_id, model_id, class_nb, batch_idx,mode,path_list):
    csvPath = "../results/{}/{}_epoch{}_{}.csv".format(exp_id, model_id, epoch,mode)

    if (batch_idx == 0 and (epoch == 1 or mode == "test")) or not os.path.exists(csvPath):
        with open(csvPath, "w") as text_file:
            print("file,targ," + ",".join(np.arange(class_nb).astype(str)), file=text_file)

    with open(csvPath, "a") as text_file:
        for i in range(len(predBatch)):
            print(path_list[i]+","+str(targBatch[i].cpu().detach().numpy()) + "," + ",".join(
                predBatch[i][:class_nb].cpu().detach().numpy().astype(str)), file=text_file)

def writeSummaries(metrDict, totalImgNb, epoch, mode, model_id, exp_id):
    ''' Write the metric computed during an evaluation in a csv file

    Args:
    - metrDict (dict): the dictionary containing the value of metrics (not divided by the number of batch)
    - totalImgNb (int): the total number of images during the epoch
    - mode (str): either 'train', 'val' or 'test' to indicate if the epoch was a training epoch or a validation epoch
    - model_id (str): the id of the model
    - exp_id (str): the experience id
    - nbVideos (int): During validation the metrics are computed over whole videos and not batches, therefore the number of videos should be indicated \
        with this argument during validation

    Returns:
    - metricDict (dict): a dictionnary containing the metrics value

    '''

    for metric in metrDict.keys():
        metrDict[metric] /= totalImgNb

    header = ",".join([metric.lower().replace(" ", "_") for metric in metrDict.keys()])

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id, model_id, epoch, mode), "a") as text_file:
        print(header, file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def getOptim_and_Scheduler(optimStr, lr,momentum,weightDecay,useScheduler,maxEpoch,lastEpoch,net):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim, optimStr)
        if optimStr == "SGD":
            kwargs = {'lr':lr,'momentum': momentum,"weight_decay":weightDecay}
        elif optimStr == "Adam":
            kwargs = {'lr':lr,"weight_decay":weightDecay}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':lr,'amsgrad': True,"weight_decay":weightDecay}

    optim = optimConst(net.parameters(), **kwargs)

    if useScheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=2, gamma=0.9)
        print("Sched",scheduler.get_last_lr())
        for _ in range(lastEpoch):
            scheduler.step()
        print("Sched",scheduler.get_last_lr())
    else:
        scheduler = None

    return optim, scheduler

def initialize_Net_And_EpochNumber(net, exp_id, model_id, cuda, start_mode, init_path, strict):
    '''Initialize a network

    If init is None, the network will be left unmodified. Its initial parameters will be saved.

    Args:
        net (CNN): the net to be initialised
        exp_id (string): the name of the experience
        model_id (int): the id of the network
        cuda (bool): whether to use cuda or not
        start_mode (str): a string indicating the start mode. Can be \'scratch\' or \'fine_tune\'.
        init_path (str): the path to the weight file to use to initialise. Ignored is start_mode is \'scratch\'.

    Returns: the start epoch number
    '''

    if start_mode == "auto":
        if len(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id))) > 0:
            start_mode = "fine_tune"
        else:
            start_mode = "scratch"
        print("Autodetected mode", start_mode)

    if start_mode == "scratch":

        # Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id, model_id))
        startEpoch = 1

    elif start_mode == "fine_tune":

        if init_path == "None":
            init_path = sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id)), key=utils.findLastNumbers)[-1]

        net = preprocessAndLoadParams(init_path,cuda,net,strict)

        startEpoch = utils.findLastNumbers(init_path)+1

    return startEpoch

def preprocessAndLoadParams(init_path,cuda,net,strict):
    print("Init from",init_path)
    params = torch.load(init_path, map_location="cpu" if not cuda else None)

    params = addOrRemoveModule(params,net)
    paramCount = len(params.keys())
    params = removeBadSizedParams(params,net)
    if paramCount != len(params.keys()):
        strict=False
    params = addFeatModZoom(params,net)
    params = changeOldNames(params,net)
    params = removeConvDense(params)

    res = net.load_state_dict(params, False)

    # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
    if not res is None:
        missingKeys, unexpectedKeys = res
        if len(missingKeys) > 0:
            print("missing keys")
            for key in missingKeys:
                print(key)
        if len(unexpectedKeys) > 0:
            print("unexpected keys")
            for key in unexpectedKeys:
                print(key)

    return net

def removeConvDense(params):

    keyToRemove = []

    for key in params:
        if key.find("featMod.fc") != -1:
            keyToRemove.append(key)
    for key in keyToRemove:
        params.pop(key,None)

    return params

def addOrRemoveModule(params,net):
    # Checking if the key of the model start with "module."
    startsWithModule = (list(net.state_dict().keys())[0].find("module.") == 0)

    if startsWithModule:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = "module." + key if key.find("module") == -1 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    else:
        paramsFormated = {}
        for key in params.keys():
            keyFormat = key.split('.')
            if keyFormat[0] == 'module':
                keyFormat = '.'.join(keyFormat[1:])
            else:
                keyFormat = '.'.join(keyFormat)
            # keyFormat = key.replace("module.", "") if key.find("module.") == 0 else key
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
    return params

def addFeatModZoom(params,net):

    shouldAddFeatModZoom = False
    for key in net.state_dict().keys():
        if key.find("featMod_zoom") != -1:
            shouldAddFeatModZoom = True

    if shouldAddFeatModZoom:
        #Adding keys in case model was created before the zoom feature was implemented
        keyValsToAdd = {}
        for key in params.keys():
            if key.find(".featMod.") != -1:
                keyToAdd = key.replace(".featMod.",".featMod_zoom.")
                valToAdd = params[key]
            keyValsToAdd.update({keyToAdd:valToAdd})
        params.update(keyValsToAdd)
    return params

def removeBadSizedParams(params,net):
    # Removing keys corresponding to parameter which shape are different in the checkpoint and in the current model
    # For example, this is necessary to load a model trained on n classes to bootstrap a model with m != n classes.
    keysToRemove = []
    for key in params.keys():
        if key in net.state_dict().keys():
            if net.state_dict()[key].size() != params[key].size():
                keysToRemove.append(key)
    for key in keysToRemove:
        params.pop(key)
    return params

def changeOldNames(params,net):
    # This is necessary to start with weights created when the model attributes were "visualModel" and "tempModel".
    paramsWithNewNames = {}
    for key in params.keys():
        paramsWithNewNames[key.replace("visualModel", "firstModel").replace("tempModel", "secondModel")] = params[
            key]
    params = paramsWithNewNames

    if hasattr(net, "secondModel"):
        if not hasattr(net.secondModel, "linLay"):
            def checkAndReplace(key):
                if key.find("secondModel.linLay") != -1:
                    key = key.replace("secondModel.linLay", "secondModel.linTempMod.linLay")
                return key

            params = {checkAndReplace(k): params[k] for k in params.keys()}
    return params

def getBestEpochInd_and_WorseEpochNb(start_mode, exp_id, model_id, epoch):
    if start_mode == "scratch":
        bestEpoch = epoch
        worseEpochNb = 0
    else:
        bestModelPaths = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id, model_id))
        if len(bestModelPaths) == 0:
            bestEpoch = epoch
            worseEpochNb = 0
        elif len(bestModelPaths) == 1:
            bestModelPath = bestModelPaths[0]
            bestEpoch = int(os.path.basename(bestModelPath).split("epoch")[1])
            worseEpochNb = epoch - bestEpoch
        else:
            raise ValueError("Wrong number of best model weight file : ", len(bestModelPaths))

    return bestEpoch, worseEpochNb


def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str, metavar='SM',
                                  help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')
    argreader.parser.add_argument('--init_path', type=str, metavar='SM',
                                  help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--strict_init', type=str2bool, metavar='SM',
                                  help='Set to True to make torch.load_state_dict throw an error when not all keys match (to use with --init_path)')

    return argreader


def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=float, metavar='LR',
                                  help='learning rate')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                                  help='SGD momentum')
    argreader.parser.add_argument('--weight_decay', type=float, metavar='M',
                                  help='Weight decay')
    argreader.parser.add_argument('--use_scheduler', type=args.str2bool, metavar='M',
                                  help='To use a learning rate scheduler')
    argreader.parser.add_argument('--always_sched', type=args.str2bool, metavar='M',
                                  help='To always use a learning rate scheduler when optimizing hyper params')

    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                                  help='the optimizer to use (default: \'SGD\')')

    argreader.parser.add_argument('--bil_clus_soft_sched', type=args.str2bool, metavar='BOOL',
                                  help='Added schedule to increase temperature of the softmax of the bilinear cluster model.')

    return argreader


def addValArgs(argreader):

    argreader.parser.add_argument('--metric_early_stop', type=str, metavar='METR',
                                  help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=args.str2bool, metavar='BOOL',
                                  help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=int, metavar='NB',
                                  help='The number of epochs to wait if the validation performance does not improve.')
    argreader.parser.add_argument('--run_test', type=args.str2bool, metavar='NB',
                                  help='Evaluate the model on the test set')



    return argreader


def addLossTermArgs(argreader):
    argreader.parser.add_argument('--nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function.')
    argreader.parser.add_argument('--aux_mod_nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function for the aux model (when using pointnet).')
    argreader.parser.add_argument('--zoom_nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function for the zoom model (when using a model that generates points).')
    argreader.parser.add_argument('--bil_backgr_weight', type=float, metavar='FLOAT',
                                  help='The weight of the background term when using bilinear model.')
    argreader.parser.add_argument('--bil_backgr_thres', type=float, metavar='FLOAT',
                                  help='The threshold between 0 and 1 for the background term when using bilinear model.')

    argreader.parser.add_argument('--crop_nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function for the crop term.')
    argreader.parser.add_argument('--drop_nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function for the drop term.')

    argreader.parser.add_argument('--center_loss_weight', type=float, metavar='FLOAT',
                                  help='The weight of the center loss term in the loss function when using bilinear model.')

    argreader.parser.add_argument('--supervised_segm_weight', type=float, metavar='FLOAT',
                                  help='The weight of the supervised segmentation term.')

    return argreader

def initMasterNet(args,gpu=None):
    config = configparser.ConfigParser()

    config.read("../models/{}/{}.ini".format(args.exp_id,args.m_model_id))
    args_master = Bunch(config["default"])

    args_master.multi_gpu = args.multi_gpu
    args_master.distributed = args.distributed

    argDic = args.__dict__
    mastDic = args_master.__dict__

    for arg in mastDic:
        if arg in argDic:
            if not argDic[arg] is None:
                if not type(argDic[arg]) is bool:
                    if mastDic[arg] != "None":
                        mastDic[arg] = type(argDic[arg])(mastDic[arg])
                    else:
                        mastDic[arg] = None
                else:
                    if arg != "multi_gpu" and arg != "distributed":
                        mastDic[arg] = str2bool(mastDic[arg]) if mastDic[arg] != "None" else False
            else:
                mastDic[arg] = None

    for arg in argDic:
        if not arg in mastDic:
            mastDic[arg] = argDic[arg]

    master_net = modelBuilder.netBuilder(args_master,gpu=gpu)

    best_paths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.m_model_id))

    if len(best_paths) > 1:
        raise ValueError("Too many best path for master")
    if len(best_paths) == 0:
        print("Missing best path for master")
    else:
        bestPath = best_paths[0]
        params = torch.load(bestPath, map_location="cpu" if not args.cuda else None)

        for key in params:
            if key.find("firstModel.attention.1.weight") != -1:

                if params[key].shape[0] < master_net.state_dict()[key].shape[0]:
                    padd = torch.zeros(1,params[key].size(1),params[key].size(2),params[key].size(3)).to(params[key].device)
                    params[key] = torch.cat((params[key],padd),dim=0)
                elif params[key].shape[0] > master_net.state_dict()[key].shape[0]:
                    params[key] = params[key][:master_net.state_dict()[key].shape[0]]

        params = addOrRemoveModule(params,master_net)
        params = removeConvDense(params)
        master_net.load_state_dict(params, strict=True)

    master_net.eval()

    return master_net

def run(args,trial=None):

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not trial is None:
        args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        args.optim = trial.suggest_categorical("optim", OPTIM_LIST)

        if args.distributed:
            if args.max_batch_size <= 12//torch.cuda.device_count():
                minBS = 1
            else:
                minBS = 12//torch.cuda.device_count()
        else:
            if args.max_batch_size <= 12:
                minBS = 4
            else:
                minBS = 12
        print(minBS,args.distributed)
        args.batch_size = trial.suggest_int("batch_size", minBS, args.max_batch_size, log=True)
        print("Batch size is ",args.batch_size)
        args.dropout = trial.suggest_float("dropout", 0, 0.6,step=0.2)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        if args.optim == "SGD":
            args.momentum = trial.suggest_float("momentum", 0., 0.9,step=0.1)
            if not args.always_sched:
                args.use_scheduler = trial.suggest_categorical("use_scheduler",[True,False])
    
        if args.always_sched:
            args.use_scheduler = True

        if args.opt_data_aug:
            args.brightness = trial.suggest_float("brightness", 0, 0.5, step=0.05)
            args.saturation = trial.suggest_float("saturation", 0, 0.9, step=0.1)
            args.crop_ratio = trial.suggest_float("crop_ratio", 0.8, 1, step=0.05)

        if args.master_net:
            args.kl_temp = trial.suggest_float("kl_temp", 1, 21, step=5)
            args.kl_interp = trial.suggest_float("kl_interp", 0.1, 1, step=0.1)

            if args.transfer_att_maps:
                args.att_weights = trial.suggest_float("att_weights",0.001,0.5,log=True)

        if args.opt_att_maps_nb:
            args.resnet_bil_nb_parts = trial.suggest_int("resnet_bil_nb_parts", 3, 64, log=True)

    if not args.distributed:
        args.world_size = 1
        value = train(0,args,trial)
        return value
    else:
        if args.distributed:
            args.world_size = torch.cuda.device_count()
            os.environ['MASTER_ADDR'] = 'localhost'              #
            os.environ['MASTER_PORT'] = '8889'
            mp.spawn(train, nprocs=args.world_size, args=(args,trial))
            value = np.genfromtxt("../results/{}/{}_{}_valRet.csv".format(args.exp_id,args.model_id,trial.number))
            return value

def train(gpu,args,trial):

    if args.distributed:
        dist.init_process_group(backend='nccl',init_method='env://',world_size=args.world_size,rank=gpu)

    if not trial is None:
        args.trial_id = trial.number

    if not args.only_test:
        trainLoader,_ = load_data.buildTrainLoader(args,withSeg=args.with_seg,reprVec=args.repr_vec,gpu=gpu)
    else:
        trainLoader = None
    valLoader,_ = load_data.buildTestLoader(args,"val",withSeg=args.with_seg,reprVec=args.repr_vec,gpu=gpu)

    # Building the net
    net = modelBuilder.netBuilder(args,gpu=gpu)

    trainFunc = epochSeqTr
    valFunc = epochImgEval

    kwargsTr = {'log_interval': args.log_interval, 'loader': trainLoader, 'args': args,"gpu":gpu}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    startEpoch = initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, args.start_mode,
                                                args.init_path, args.strict_init)

    kwargsTr["optim"],scheduler = getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,args.epochs,startEpoch,net)

    epoch = startEpoch
    bestEpoch, worseEpochNb = getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    if args.maximise_val_metric:
        bestMetricVal = -np.inf
        isBetter = lambda x, y: x > y
    else:
        bestMetricVal = np.inf
        isBetter = lambda x, y: x < y

    if args.master_net:
        kwargsTr["master_net"] = initMasterNet(args,gpu=gpu)
        kwargsVal["master_net"] = kwargsTr["master_net"]

    lossFunc = Loss(args,reduction="mean")

    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc,device_ids=[gpu])

    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc

    if not args.only_test and not args.grad_cam:

        actual_bs = args.batch_size if args.batch_size < args.max_batch_size_single_pass else args.max_batch_size_single_pass
        args.batch_per_epoch = len(trainLoader.dataset)//actual_bs if len(trainLoader.dataset) > actual_bs else 1

        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            if args.protonet:
                if epoch - startEpoch > args.protonet_warm:
                    print("Joint")
                    protopnet.joint(net.module.firstModel.protopnet)
                else:
                    print("Warmup")
                    protopnet.warm_only(net.module.firstModel.protopnet)
            elif args.prototree:
                if  epoch-startEpoch < args.protonet_warm:
                    net.module.firstModel.featMod.requires_grad= False 
                else:
                    net.module.firstModel.featMod.requires_grad= True

            trainFunc(**kwargsTr)
            if not scheduler is None:
                scheduler.step()

            if not (args.no_val or args.grad_exp):
                with torch.no_grad():
                    metricVal = valFunc(**kwargsVal)
                if gpu == 0:
                    bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                                args.model_id, bestEpoch, epoch, net,
                                                                                isBetter, worseEpochNb)
                if trial is not None and gpu==0:
                    trial.report(metricVal, epoch)

            epoch += 1

    if trial is None:
        if args.run_test or args.only_test:

            if os.path.exists("../results/{}/test_done.txt".format(args.exp_id)):
                test_done = np.genfromtxt("../results/{}/test_done.txt".format(args.exp_id),delimiter=",",dtype=str)

                if len(test_done.shape) == 1:
                    test_done = test_done[np.newaxis]
            else:
                test_done = None

            alreadyDone = (test_done==np.array([args.model_id,str(bestEpoch)])).any()

            if (test_done is None) or (alreadyDone and args.do_test_again) or (not alreadyDone):

                testFunc = valFunc

                kwargsTest = kwargsVal
                kwargsTest["mode"] = "test"

                testLoader,_ = load_data.buildTestLoader(args, "test",withSeg=args.with_seg,reprVec=args.repr_vec,shuffle=args.shuffle_test_set)

                kwargsTest['loader'] = testLoader

                net = preprocessAndLoadParams("../models/{}/model{}_best_epoch{}".format(args.exp_id, args.model_id, bestEpoch),args.cuda,net,args.strict_init)

                kwargsTest["model"] = net
                kwargsTest["epoch"] = bestEpoch

                if not args.grad_exp_test:    
                    with torch.no_grad():
                        testFunc(**kwargsTest)
                else:
                    testFunc(**kwargsTest)

                with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                    print("{},{}".format(args.model_id,bestEpoch),file=text_file)

    else:
        if gpu == 0:
            oldPath = "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, bestEpoch)
            os.rename(oldPath, oldPath.replace("best_epoch","trial{}_best_epoch".format(trial.number)))

            with open("../results/{}/{}_{}_valRet.csv".format(args.exp_id,args.model_id,trial.number),"w") as text:
                print(metricVal,file=text)

        return metricVal

def updateSeedAndNote(args):
    if args.start_mode == "auto" and len(
            glob.glob("../models/{}/model{}_epoch*".format(args.exp_id, args.model_id))) > 0:
        args.seed += 1
        init_path = args.init_path
        if init_path == "None" and args.strict_init:
            init_path = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id, args.model_id)),
                               key=utils.findLastNumbers)[-1]
        startEpoch = utils.findLastNumbers(init_path)
        args.note += ";s{} at {}".format(args.seed, startEpoch)
    return args

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--no_train', type=str2bool, help='To use to re-evaluate a model at each epoch after training. At each epoch, the model is not trained but \
                                                                            the weights of the corresponding epoch are loaded and then the model is evaluated.\
                                                                            The arguments --exp_id_no_train and the --model_id_no_train must be set')
    argreader.parser.add_argument('--exp_id_no_train', type=str,
                                  help="To use when --no_train is set to True. This is the exp_id of the model to get the weights from.")
    argreader.parser.add_argument('--model_id_no_train', type=str,
                                  help="To use when --no_train is set to True. This is the model_id of the model to get the weights from.")

    argreader.parser.add_argument('--no_val', type=str2bool, help='To not compute the validation')
    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')
    argreader.parser.add_argument('--compute_latency', type=str2bool, help='To write in a file the latency at each forward pass.')
    argreader.parser.add_argument('--grad_cam', type=int, help='To compute grad cam instead of training or testing.',nargs="*")
    argreader.parser.add_argument('--rise', type=str2bool, help='To compute rise instead or gradcam')
    argreader.parser.add_argument('--score_map', type=str2bool, help='To compute score_map instead or gradcam')
    argreader.parser.add_argument('--noise_tunnel', type=str2bool, help='To compute the methods based on noise tunnel instead or gradcam')

    argreader.parser.add_argument('--viz_id', type=str, help='The visualization id.',default="")

    argreader.parser.add_argument('--attention_metrics', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
    
    argreader.parser.add_argument('--att_metrics_post_hoc', type=str, help='The post-hoc method to use instead of the model ')
    argreader.parser.add_argument('--att_metrics_max_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_onlyfirst_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_few_steps', type=str2bool, help='To do as much step for high res than for low res')
    argreader.parser.add_argument('--att_metr_do_again', type=str2bool, help='To run computation if already done',default=True)

    argreader.parser.add_argument('--att_metr_img_bckgr', type=str2bool, help='To replace the image by another image instead of a black patch',default=False)
    argreader.parser.add_argument('--att_metr_save_feat', type=str2bool, help='',default=False)

    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader.parser.add_argument('--grad_exp', type=str, help='To store the gradients of the feature matrix.')
    argreader.parser.add_argument('--grad_exp_test', type=str2bool, help='To store the gradients of the feature matrix during test.')
    argreader.parser.add_argument('--trial_id', type=int, help='The trial ID. Useful for grad exp during test')

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    # The folders where the experience file will be written
    if not os.path.exists("../vis/{}".format(args.exp_id)):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not os.path.exists("../results/{}".format(args.exp_id)):
        os.makedirs("../results/{}".format(args.exp_id))
    if not os.path.exists("../models/{}".format(args.exp_id)):
        os.makedirs("../models/{}".format(args.exp_id))

    args = updateSeedAndNote(args)
    # Update the config args
    argreader.args = args
    # Write the arguments in a config file so the experiment can be re-run

    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id, args.model_id))
    print("Model :", args.model_id, "Experience :", args.exp_id)

    if args.optuna:
        def objective(trial):
            return run(args,trial=trial)

        study = optuna.create_study(direction="maximize" if args.maximise_val_metric else "minimize",\
                                    storage="sqlite:///../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id), \
                                    study_name=args.model_id,load_if_exists=True)

        con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
        curr = con.cursor()

        failedTrials = 0
        for elem in curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall():
            if elem[1] is None:
                failedTrials += 1

        trialsAlreadyDone = len(curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall())

        if trialsAlreadyDone-failedTrials < args.optuna_trial_nb:

            studyDone = False
            while not studyDone:
                try:
                    print("N trials",args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                    study.optimize(objective,n_trials=args.optuna_trial_nb-trialsAlreadyDone+failedTrials)
                    studyDone = True
                except Exception as e:
                    if str(e).find("CUDA out of memory.") != -1:
                        gc.collect()
                        torch.cuda.empty_cache()
                        args.max_batch_size -= 5
                        print("New max batch size",args.max_batch_size)
                    else:
                        raise RuntimeError(e)

        curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
        query_res = curr.fetchall()

        query_res = list(filter(lambda x:not x[1] is None,query_res))

        trialIds = [id_value[0] for id_value in query_res]
        values = [id_value[1] for id_value in query_res]

        trialIds = trialIds[:args.optuna_trial_nb]
        values = values[:args.optuna_trial_nb]

        bestTrialId = trialIds[np.array(values).argmax()]

        curr.execute('SELECT param_name,param_value from trial_params WHERE trial_id == {}'.format(bestTrialId))
        query_res = curr.fetchall()

        bestParamDict = {key:value for key,value in query_res}

        #args.lr,args.batch_size = bestParamDict["lr"],int(bestParamDict["batch_size"])
        #args.optim = OPTIM_LIST[int(bestParamDict["optim"])]
        args.only_test = True

        print("bestTrialId-1",bestTrialId-1)
        bestPath = glob.glob("../models/{}/model{}_trial{}_best_epoch*".format(args.exp_id,args.model_id,bestTrialId-1))[0]
        print(bestPath)

        copyfile(bestPath, bestPath.replace("_trial{}".format(bestTrialId-1),""))


        args.distributed=False

        train(0,args,None)

    elif args.grad_exp:

        if len(glob.glob("../results/{}/{}_allGrads_{}HypParams_epoch*.th".format(args.exp_id,args.model_id,args.grad_exp))) < args.epochs:

            con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
            curr = con.cursor()

            if args.grad_exp == "best":
                print("Grad exp : Best run")
                trialId = getBestTrial(curr,args.optuna_trial_nb)
            elif args.grad_exp == "worst":
                print("Grad exp : worst run")
                trialId = getWorstTrial(curr,args.optuna_trial_nb)
            else:
                print("Grad exp : median run")
                trialId = getMedianTrial(curr,args.optuna_trial_nb)

            curr.execute('SELECT param_name,param_value from trial_params WHERE trial_id == {}'.format(trialId))
            query_res = curr.fetchall()

            paramDict = {key:value for key,value in query_res}

            args.lr,args.batch_size = paramDict["lr"],int(paramDict["batch_size"])
            args.optim = OPTIM_LIST[int(paramDict["optim"])]

            if "dropout" in paramDict:
                args.dropout = paramDict["dropout"]
                args.weight_decay = paramDict["weight_decay"]

                if args.optim == "SGD":
                    args.momentum = paramDict["momentum"]
                    args.use_scheduler = (paramDict["use_scheduler"]==1.0)

            if args.opt_data_aug:
                args.brightness = paramDict["brightness"]
                args.saturation = paramDict["saturation"]
                args.crop_ratio = paramDict["crop_ratio"]

            args.run_test = False
            args.distributed = False
            args.optuna = False
            train(0,args,None)
        else:
            print("Already done")
    elif args.grad_exp_test:

        args.distributed = False

        con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(args.exp_id,args.model_id))
        curr = con.cursor()
        trialIds,values = getTrialList(curr,args.optuna_trial_nb)
        valDic = {id:val for id,val in zip(trialIds,values)}

        bestOfAllPaths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id))
        if len(bestOfAllPaths) >1:
            raise ValueError("Too many best path ({}) : {}".format(len(bestOfAllPaths),bestOfAllPaths))
        elif len(bestOfAllPaths) == 1:
            bestOfAllPath = bestOfAllPaths[0]
            copyfile(bestOfAllPath, bestOfAllPath.replace("best","bestOfAll"))
            os.remove(bestOfAllPath)
        else:
            if len(glob.glob("../models/{}/model{}_bestOfAll_epoch*".format(args.exp_id,args.model_id))) == 0:
                raise ValueError("No best of bestOfAll weight file.")

        bestPaths = glob.glob("../models/{}/model{}_trial*_best*".format(args.exp_id,args.model_id))
        bestPaths = sorted(bestPaths,key=lambda x:int(os.path.basename(x).split("trial")[1].split("_")[0]))
        bestPaths = bestPaths[:args.optuna_trial_nb]

        snrPath = "../results/{}/snr_{}.csv".format(args.exp_id,args.model_id)
        if not os.path.exists(snrPath):
            with open(snrPath,"w") as text_file:
                print("trial_id,snr,accuracy",file=text_file)

        snr_csv = np.genfromtxt(snrPath,delimiter=",",dtype=str)
        if len(snr_csv.shape) == 1:
            snr_csv = snr_csv[np.newaxis]
            trial_ids_done = []
        else:
            trial_ids_done = snr_csv[1:,0]

        for path in bestPaths:
            trialId = int(os.path.basename(path).split("trial")[1].split("_")[0])+1
            args.trial_id = trialId

            print("Trial id",trialId,"Accuracy :",valDic[trialId],"Path",path)

            exists=False
            for doneTrial in trial_ids_done:
                if str(trialId) == doneTrial:
                    exists=True
            args.only_test = True
            if not exists:
                copyfile(path, path.replace("trial{}_best".format(trialId-1),"best"))
                train(0,args,None)

                os.remove(path.replace("trial{}_best".format(trialId-1),"best"))

        copyfile(bestOfAllPath.replace("best","bestOfAll"),bestOfAllPath)

    elif args.grad_cam:

        args.val_batch_size = 1
        testLoader,testDataset = load_data.buildTestLoader(args, "test",withSeg=args.with_seg)

        bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
        bestEpoch = int(os.path.basename(bestPath).split("epoch")[1])

        net = modelBuilder.netBuilder(args,gpu=0)
        net_raw = preprocessAndLoadParams(bestPath,args.cuda,net,args.strict_init)

        if not args.rise and not args.score_map and not args.noise_tunnel:
            net = modelBuilder.GradCamMod(net_raw)
            model_dict = dict(type=args.first_mod, arch=net, layer_name='layer4', input_size=(448, 448))
            grad_cam = captum.attr.LayerGradCam(net.forward,net.layer4)
            grad_cam_pp = GradCAMpp(model_dict,True)
            guided_backprop_mod = captum.attr.GuidedBackprop(net)
            
            allMask = None
            allMask_pp = None
            allMaps = None 
        elif args.score_map:
            score_mod = ScoreCam(net_raw)
            allScore = None
        elif args.noise_tunnel:
            torch.backends.cudnn.benchmark = False

            net_raw.eval()
            net = modelBuilder.GradCamMod(net_raw)

            ig = IntegratedGradients(net)
            nt = NoiseTunnel(ig)

            batch = testDataset.__getitem__(0)
            data_base = torch.zeros_like(batch[0].unsqueeze(0))
            if args.cuda:
                data_base = data_base.cuda()
            
            allSq = None 
            allVar = None

        else:
            rise_mod = RISE(net_raw)
            allRise = None

        if args.grad_cam == [-1]:
            args.grad_cam = np.arange(len(testDataset))
        
        for i in args.grad_cam:
            batch = testDataset.__getitem__(i)
            data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)

            if args.cuda:
                data = data.cuda()
                targ = targ.cuda()

            if not args.rise and not args.score_map and not args.noise_tunnel:

                mask = grad_cam.attribute(data,targ).detach().cpu()
                mask_pp = grad_cam_pp(data,targ).detach().cpu()
                map = guided_backprop_mod.attribute(data,targ).detach().cpu()

                if allMask is None:
                    allMask = mask
                    allMask_pp = mask_pp
                    allMaps = map
                else:
                    allMask = torch.cat((allMask,mask),dim=0)
                    allMask_pp = torch.cat((allMask_pp,mask_pp),dim=0)
                    allMaps = torch.cat((allMaps,map),dim=0)
            elif args.score_map:
                score_map = score_mod.generate_cam(data).detach().cpu()

                if allScore is None:
                    allScore = torch.tensor(score_map)
                else:
                    allScore = torch.cat((allScore,torch.tensor(score_map)),dim=0)       
            elif args.noise_tunnel:    

                attr_sq = nt.attribute(data, nt_type='smoothgrad_sq', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3,baselines=data_base, target=targ)
                attr_var = nt.attribute(data, nt_type='vargrad', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3,baselines=data_base, target=targ)

                attr_sq,attr_var = attr_sq.detach().cpu(),attr_var.detach().cpu()

                if allSq is None:
                    allSq,allVar = attr_sq,attr_var
                else:
                    allSq,allVar = torch.cat((allSq,attr_sq),dim=0),torch.cat((allVar,attr_var),dim=0)

            else:
                with torch.no_grad():
                    rise_map = rise_mod(data).detach().cpu()

                if allRise is None:
                    allRise = rise_map
                else:
                    allRise = torch.cat((allRise,rise_map),dim=0)

        suff = "" if args.viz_id == "" else "{}_".format(args.viz_id)
        if not args.rise and not args.score_map and not args.noise_tunnel:
            np.save("../results/{}/gradcam_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMask.numpy())
            np.save("../results/{}/gradcam_pp_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMask_pp.numpy())
            np.save("../results/{}/gradcam_maps_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMaps.numpy())
        elif args.score_map:
            np.save("../results/{}/score_maps_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allScore.numpy())
        elif args.noise_tunnel:
            np.save("../results/{}/smoothgrad_sq_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allSq.numpy())
            np.save("../results/{}/vargrad_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allVar.numpy())
        else:
            np.save("../results/{}/rise_maps_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allRise.numpy())

    elif args.attention_metrics:

        path_suff = args.attention_metrics
        path_suff += "-IB" if args.att_metr_img_bckgr else ""
        model_id_suff = "-"+args.att_metrics_post_hoc if args.att_metrics_post_hoc else ""

        if args.att_metr_img_bckgr:
            print("\tImg bckgr")

        if args.att_metr_do_again or not os.path.exists("../results/{}/attMetr{}_{}{}.npy".format(args.exp_id,path_suff,args.model_id,model_id_suff)):

            args.val_batch_size = 1
            testLoader,testDataset = load_data.buildTestLoader(args, "test",withSeg=args.with_seg)

            bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
            bestEpoch = int(os.path.basename(bestPath).split("epoch")[1])

            if args.prototree:
                net = torch.load(glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id))[0])
            else:
                net = modelBuilder.netBuilder(args,gpu=0)
                net = preprocessAndLoadParams(bestPath,args.cuda,net,args.strict_init)
            net.eval()
            
            if args.att_metrics_post_hoc:
                attrFunc,kwargs = getAttMetrMod(net,testDataset,args)
            else:
                attMaps_dataset,norm_dataset = loadAttMaps(args.exp_id,args.model_id)

                if not args.resnet_bilinear or (args.resnet_bilinear and args.bil_cluster):
                    attrFunc = lambda i:(attMaps_dataset[i,0:1]*norm_dataset[i]).unsqueeze(0)
                else:
                    attrFunc = lambda i:(attMaps_dataset[i].float().mean(dim=0,keepdim=True).byte()*norm_dataset[i]).unsqueeze(0)
            
            if args.att_metrics_post_hoc != "gradcam_pp":
                torch.set_grad_enabled(False)

            nbImgs = args.att_metrics_img_nb
            print("\tnbImgs",nbImgs)

            if args.attention_metrics in ["Del","Add"]:
                allScoreList = []
                allPreds = []
                allTarg = []
                allFeat = []
            elif args.attention_metrics == "AttScore":
                allAttScor = []
            elif args.attention_metrics == "Time":
                allTimeList = []
            elif args.attention_metrics == "Lift":
                allScoreList = []
                allScoreMaskList = []
                allScoreInvMaskList = []
                allFeat = []
            else:
                allSpars = []

            torch.manual_seed(0)
            inds = torch.randint(len(testDataset),size=(nbImgs,))

            if args.att_metr_img_bckgr:
                inds_bckgr = (inds + len(testDataset)//2) % len(testDataset)

                labels = np.array([testDataset[ind][1] for ind in inds])
                labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

                while (labels==labels_bckgr).any():
                    for i in range(len(inds)):
                        if inds[i] == inds_bckgr[i]:
                            inds_bckgr[i] = torch.randint(len(testDataset),size=(1,))[0]

                    inds_bckgr = torch.randint(len(testDataset),size=(nbImgs,))
                    labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

            blurWeight = torch.ones(121,121)
            blurWeight = blurWeight/blurWeight.numel()
            blurWeight = blurWeight.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
            blurWeight = blurWeight.cuda() if args.cuda else blurWeight

            for imgInd,i in enumerate(inds):
                if imgInd % 20 == 0 :
                    print("Img",i.item(),"(",imgInd,"/",len(inds),")")

                data,targ = getBatch(testDataset,i,args)
                allData = data.clone().cpu()

                if args.att_metr_img_bckgr:
                    data_bckgr,_ = getBatch(testDataset,inds_bckgr[imgInd],args)
                else:
                    data_bckgr = None

                startTime = time.time()
                if not args.prototree:
                    resDic = net(data)
                    scores = torch.softmax(resDic["pred"],dim=-1)
                else:
                    resDic = None
                    scores = net(data)[0]
                inf_time = time.time() - startTime

                if args.attention_metrics in ["Add","Del"]:
                    predClassInd = scores.argmax(dim=-1)
                    allPreds.append(predClassInd.item())
                    allTarg.append(targ.item())

                if args.att_metrics_post_hoc:
                    startTime = time.time()
                    attMaps = applyPostHoc(attrFunc,data,targ,kwargs,args)
                    totalTime = inf_time + time.time() - startTime
                else:
                    attMaps = attrFunc(i)
                    totalTime = inf_time

                if args.attention_metrics=="Add":
                    origData = data.clone()
                    if args.att_metr_img_bckgr:
                        data = data_bckgr.clone()
                    else:
                        data = F.conv2d(data,blurWeight,padding=blurWeight.size(-1)//2,groups=blurWeight.size(0))

                attMaps = (attMaps-attMaps.min())/(attMaps.max()-attMaps.min())

                if args.attention_metrics=="Spars":
                    sparsity= computeSpars(data.size(),attMaps,args,resDic)
                    allSpars.append(sparsity)
                elif args.attention_metrics == "AttScore":
                    allAttScor.append(attMaps.view(-1).sort()[0].detach().cpu().numpy())
                elif args.attention_metrics == "Time":
                    allTimeList.append(totalTime)
                elif args.attention_metrics == "Lift":
                    predClassInd = scores.argmax(dim=-1)
                    score = scores[:,predClassInd[0]].cpu().detach().numpy()
                    allScoreList.append(score)
                    attMaps_interp = torch.nn.functional.interpolate(attMaps,size=(data.shape[-1]),mode="bicubic",align_corners=False).to(data.device)                    
                    
                    feat = resDic["x"].detach().cpu()

                    data_masked =  maskData(data,attMaps_interp,args,data_bckgr)
                    allData = torch.cat((allData,data_masked.cpu()),dim=0)
                    score_mask,feat_mask = inference(net,data_masked,predClassInd,args)
                    feat_mask = feat_mask.detach().cpu()
                    allScoreMaskList.append(score_mask.cpu().detach().numpy())
                    
                    data_invmasked =  maskData(data,1-attMaps_interp,args,data_bckgr)
                    allData = torch.cat((allData,data_invmasked.cpu()),dim=0)
                    score_invmask,feat_invmask = inference(net,data_invmasked,predClassInd,args)
                    feat_invmask = feat_invmask.detach().cpu()
                    allScoreInvMaskList.append(score_invmask.cpu().detach().numpy())

                    allFeat.append(torch.cat((feat,feat_mask,feat_invmask),dim=0).unsqueeze(0))

                elif args.attention_metrics in ["Del","Add"]:
                    allAttMaps = attMaps.clone().cpu()
                    statsList = []

                    totalPxlNb = attMaps.size(2)*attMaps.size(3)
                    leftPxlNb = totalPxlNb

                    if args.prototree or args.protonet:
                        stepNb = 49
                    elif args.att_metrics_few_steps:
                        stepNb = 196 
                    else:
                        stepNb = totalPxlNb

                    score_prop_list = []

                    ratio = data.size(-1)//attMaps.size(-1)

                    stepCount = 0

                    allFeatIter = []
                    while leftPxlNb > 0:

                        attMin,attMean,attMax = attMaps.min().item(),attMaps.mean().item(),attMaps.max().item()
                        statsList.append((attMin,attMean,attMax))

                        _,ind_max = (attMaps)[0,0].view(-1).topk(k=totalPxlNb//stepNb)
                        ind_max = ind_max[:leftPxlNb]

                        x_max,y_max = ind_max % attMaps.shape[3],ind_max // attMaps.shape[3]
                        
                        ratio = data.size(-1)//attMaps.size(-1)

                        for i in range(len(x_max)):
                            
                            x1,y1 = x_max[i]*ratio,y_max[i]*ratio,
                            x2,y2 = x1+ratio,y1+ratio

                            if args.attention_metrics=="Add":
                                data[0,:,y1:y2,x1:x2] = origData[0,:,y1:y2,x1:x2]
                            elif args.attention_metrics=="Del":
                                if args.att_metr_img_bckgr:
                                    data[0,:,y1:y2,x1:x2] = data_bckgr[0,:,y1:y2,x1:x2]
                                else:
                                    data[0,:,y1:y2,x1:x2] = 0
                            else:
                                raise ValueError("Unkown attention metric",args.attention_metrics)

                            attMaps[0,:,y_max[i],x_max[i]] = -1                       

                        leftPxlNb -= totalPxlNb//stepNb
                        if stepCount % 30 == 0:
                            allAttMaps = torch.cat((allAttMaps,torch.clamp(attMaps,0,attMaps.max().item()).cpu()),dim=0)
                            allData = torch.cat((allData,data.cpu()),dim=0)
                        stepCount += 1

                        score,feat = inference(net,data,predClassInd,args)

                        allFeatIter.append(feat.detach().cpu())

                        score_prop_list.append((leftPxlNb,score.item()))
                    
                    allFeat.append(torch.cat(allFeatIter,dim=0).unsqueeze(0))

                    allScoreList.append(score_prop_list)
                else:
                    raise ValueError("Unkown attention metric",args.attention_metrics)

            if args.att_metrics_post_hoc:
                args.model_id = args.model_id + "-"+args.att_metrics_post_hoc
            
            if args.attention_metrics == "Spars":
                np.save("../results/{}/attMetrSpars_{}.npy".format(args.exp_id,args.model_id),np.array(allSpars,dtype=object))
            elif args.attention_metrics == "AttScore":
                np.save("../results/{}/attMetrAttScore_{}.npy".format(args.exp_id,args.model_id),np.array(allAttScor,dtype=object))
            elif args.attention_metrics == "Time":
                np.save("../results/{}/attMetrTime_{}.npy".format(args.exp_id,args.model_id),np.array(allTimeList,dtype=object))
            elif args.attention_metrics == "Lift":
                suff = path_suff.replace("Lift","")
                np.save("../results/{}/attMetrLift{}_{}.npy".format(args.exp_id,suff,args.model_id),np.array(allScoreList,dtype=object))
                np.save("../results/{}/attMetrLiftMask{}_{}.npy".format(args.exp_id,suff,args.model_id),np.array(allScoreMaskList,dtype=object))
                np.save("../results/{}/attMetrLiftInvMask{}_{}.npy".format(args.exp_id,suff,args.model_id),np.array(allScoreInvMaskList,dtype=object))
            else:
                np.save("../results/{}/attMetr{}_{}.npy".format(args.exp_id,path_suff,args.model_id),np.array(allScoreList,dtype=object))
                np.save("../results/{}/attMetrPreds{}_{}.npy".format(args.exp_id,path_suff,args.model_id),np.array(allPreds,dtype=object))
    
            if args.attention_metrics in ["Lift","Del","Add"] and args.att_metr_save_feat:
                allFeat = torch.cat(allFeat,dim=0)
                np.save("../results/{}/attMetrFeat{}_{}.npy".format(args.exp_id,path_suff,args.model_id),allFeat.numpy())
    
            if len(allData) > 1:
                torchvision.utils.save_image(allData,"../vis/{}/attMetrData{}_{}.png".format(args.exp_id,path_suff,args.model_id))
    
    else:
        train(0,args,None)

def maskData(data,attMaps_interp,args,data_bckgr):
    if args.att_metr_img_bckgr:
        data_masked = data*attMaps_interp+data_bckgr*(1-attMaps_interp)
    else:
        data_masked = data*attMaps_interp
    return data_masked 

def getBatch(testDataset,i,args):
    batch = testDataset.__getitem__(i)
    data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)
    data = data.cuda() if args.cuda else data
    targ = targ.cuda() if args.cuda else targ
    return data,targ 

def computeSpars(data_shape,attMaps,args,resDic):
    if args.att_metrics_post_hoc:
        features = None 
    else:
        features = resDic["features"]
        if "attMaps" in resDic:
            attMaps = resDic["attMaps"]
        else:
            attMaps = torch.ones(data_shape[0],1,features.size(2),features.size(3)).to(features.device)

    sparsity = metrics.compAttMapSparsity(attMaps,features)
    sparsity = sparsity/data_shape[0]
    return sparsity 

def inference(net,data,predClassInd,args):
    if not args.prototree:
        resDic = net(data)
        score = torch.softmax(resDic["pred"],dim=-1)[:,predClassInd[0]]
        feat = resDic["x"]
    else:
        score = net(data)[0][:,predClassInd[0]]
        feat = None
    return score,feat

def applyPostHoc(attrFunc,data,targ,kwargs,args):

    attMaps = []
    for i in range(len(data)):

        if args.att_metrics_post_hoc.find("var") == -1 and args.att_metrics_post_hoc.find("smooth") == -1:
            argList = [data[i:i+1],targ[i:i+1]]
        else:
            argList = [data[i:i+1]]
            kwargs["target"] = targ[i:i+1]

        attMap = torch.tensor(attrFunc(*argList,**kwargs)).to(data.device)

        if len(attMap.size()) == 2:
            attMap = attMap.unsqueeze(0).unsqueeze(0)
        elif len(attMap.size()) == 3:
            attMap = attMap.unsqueeze(0)
        
        attMaps.append(attMap)
    
    return torch.cat(attMaps,dim=0)
    
def getAttMetrMod(net,testDataset,args):
    if args.att_metrics_post_hoc == "gradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = captum.attr.LayerGradCam(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "gradcam_pp":
        netGradMod = modelBuilder.GradCamMod(net)
        model_dict = dict(type=args.first_mod, arch=netGradMod, layer_name='layer4', input_size=(448, 448))
        attrMod = GradCAMpp(model_dict,True)
        attrFunc = attrMod.__call__
        kwargs = {}
    elif args.att_metrics_post_hoc == "guided":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = captum.attr.GuidedBackprop(netGradMod)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "xgradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = XGradCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "ablation_cam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "score_map":
        attrMod = ScoreCam(net)
        attrFunc = attrMod.generate_cam
        kwargs = {}
    elif args.att_metrics_post_hoc == "varGrad" or args.att_metrics_post_hoc == "smoothGrad":
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)
        
        net.eval()
        netGradMod = modelBuilder.GradCamMod(net)

        ig = IntegratedGradients(netGradMod)
        attrMod = NoiseTunnel(ig)
        attrFunc = attrMod.attribute

        batch = testDataset.__getitem__(0)
        data_base = torch.zeros_like(batch[0].unsqueeze(0))

        if args.cuda:
            data_base = data_base.cuda()
        kwargs = {"nt_type":'smoothgrad_sq' if args.att_metrics_post_hoc == "smoothGrad" else "vargrad", \
                        "stdevs":0.02, "nt_samples":3,"nt_samples_batch_size":3}
    elif args.att_metrics_post_hoc == "rise":
        torch.set_grad_enabled(False)
        attrMod = RISE(net)
        attrFunc = attrMod.__call__
        kwargs = {}
    else:
        raise ValueError("Unknown post-hoc method",args.att_metrics_post_hoc)
    return attrFunc,kwargs

def loadAttMaps(exp_id,model_id):

    paths = glob.glob("../results/{}/attMaps_{}_epoch*.npy".format(exp_id,model_id))

    if len(paths) >1 or len(paths) == 0:
        raise ValueError("Wrong path number for exp {} model {}",exp_id,model_id)

    attMaps,norm = np.load(paths[0],mmap_mode="r"),np.load(paths[0].replace("attMaps","norm"),mmap_mode="r")

    return torch.tensor(attMaps),torch.tensor(norm)

def getBestTrial(curr,optuna_trial_nb):
    trialIds,values = getTrialList(curr,optuna_trial_nb)
    bestTrialId = trialIds[np.array(values).argmax()]
    return bestTrialId

def getWorstTrial(curr,optuna_trial_nb):
    trialIds,values = getTrialList(curr,optuna_trial_nb)
    trialIds,values = zip(*list(filter(lambda x:x[1]>0.1,zip(trialIds,values))))
    worstTrialId = trialIds[np.array(values).argmin()]
    return worstTrialId

def getMedianTrial(curr,optuna_trial_nb):
    trialIds,values = getTrialList(curr,optuna_trial_nb)
    median = np.median(np.array(values))
    medianTrialId = np.array(trialIds)[values==median][0]
    return medianTrialId

def getTrialList(curr,optuna_trial_nb):
    curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
    query_res = curr.fetchall()

    query_res = list(filter(lambda x:not x[1] is None,query_res))

    trialIds = [id_value[0] for id_value in query_res]
    values = [id_value[1] for id_value in query_res]

    trialIds = trialIds[:optuna_trial_nb]
    values = values[:optuna_trial_nb]
    return trialIds,values

if __name__ == "__main__":
    main()
