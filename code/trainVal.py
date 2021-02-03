import os
import sys
import glob

import args
from args import ArgReader
from args import str2bool

import numpy as np
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import matplotlib.pyplot as plt

plt.switch_backend('agg')

import modelBuilder
import load_data
import metrics
import utils
import update

import torch.distributed as dist
from torch.multiprocessing import Process
import time

import configparser

import optuna
import sqlite3

from shutil import copyfile

import gc

OPTIM_LIST = ["Adam", "AMSGrad", "SGD"]

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def epochSeqTr(model, optim, log_interval, loader, epoch, args, writer, **kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network
    - writer (tensorboardX.SummaryWriter): the writer to use to log metrics evolution to tensorboardX
    '''

    start_time = time.time() if args.debug or args.benchmark else None

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0

    if args.grad_exp:
        allGrads = None

    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if (batch_idx % log_interval == 0):
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]

        if args.with_seg:
            seg = batch[2]
        else:
            seg = None

        if args.cuda:
            data, target = data.cuda(), target.cuda()

            if args.with_seg:
                seg = seg.cuda()

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
        loss = loss.detach().data.item()

        if args.distributed:
            average_gradients(model)

        if args.grad_exp:
            allGrads = updateGradExp(model,allGrads)

        optim.step()
        update.updateHardWareOccupation(args.debug, args.benchmark, args.cuda, epoch, "train", args.exp_id,
                                        args.model_id, batch_idx)

        # Metrics
        with torch.no_grad():
            metDictSample = metrics.binaryToMetrics(output, target, seg,resDict)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch > 3 and args.debug:
            break

    if args.grad_exp:
        updateGradExp(model,allGrads,True,epoch,args.exp_id,args.model_id,args.grad_exp)

    # If the training set is empty (which we might want to just evaluate the model), then allOut and allGT will still be None
    if validBatch > 0:

        if not args.optuna:
            torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
            writeSummaries(metrDict, totalImgNb, writer, epoch, "train", args.model_id, args.exp_id)

        if args.debug or args.benchmark:
            totalTime = time.time() - start_time
            update.updateTimeCSV(epoch, "train", args.exp_id, args.model_id, totalTime, batch_idx)

def updateGradExp(model,allGrads,end=False,epoch=None,exp_id=None,model_id=None,grad_exp=None):

    if not end:
        newGrads = model.secondModel.linLay.weight.grad.data.unsqueeze(0)

        if allGrads is None:
            allGrads = newGrads
        else:
            allGrads = torch.cat((allGrads,newGrads),dim=0)

        return allGrads

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
    else:
        kl = F.kl_div(F.log_softmax(output/args.kl_temp, dim=1),F.softmax(resDict["master_net_pred"]/args.kl_temp, dim=1),reduction="batchmean")
        ce = F.cross_entropy(output, target)
        loss = args.nll_weight*(kl*args.kl_interp*args.kl_temp*args.kl_temp+ce*(1-args.kl_interp))

        if args.transfer_att_maps:
            loss += args.att_weights*computeAttDiff(args.att_term_included,resDict["attMaps"],resDict["features"],resDict["master_net_attMaps"],resDict["master_net_features"],args.att_pow)

    nbTerms = 1
    for key in resDict.keys():
        if key.find("pred_") != -1:
            loss += args.nll_weight * F.cross_entropy(resDict[key], target)
            nbTerms += 1

    loss = loss/nbTerms

    return loss


def computeAttDiff(att_term_included,studMaps,studFeat,teachMaps,teachFeat,attPow):

    studNorm = torch.sqrt(torch.pow(studFeat,2).sum(dim=1,keepdim=True))
    teachNorm = torch.sqrt(torch.pow(teachFeat,2).sum(dim=1,keepdim=True))

    studMaps = normMap(studMaps,minMax=True)*normMap(studNorm)
    teachMaps = normMap(teachMaps,minMax=True)*normMap(teachNorm)

    teachMaps = F.interpolate(teachMaps,size=(studMaps.size(-2),studMaps.size(-1)),mode='bilinear',align_corners=True)

    if att_term_included:
        stud_pow = torch.pow(studMaps,attPow)
        teach_pow = torch.pow(1-teachMaps,attPow)
        term = torch.pow((stud_pow*teach_pow).sum(dim=(2,3)),1.0/attPow).mean()
    else:
        term = torch.pow(torch.pow(torch.abs(teachMaps-studMaps),attPow).sum(dim=(2,3)),1.0/attPow).mean()
    return term

def normMap(map,minMax=False):
    if not minMax:
        max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        map = map/max
    else:
        max = map.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        min = map.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
        map = (map-min)/(max-min)

    return map

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if not param.grad is None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def epochImgEval(model, log_interval, loader, epoch, args, writer, metricEarlyStop, mode="val",**kwargs):
    ''' Train a model during one epoch

    Args:
    - model (torch.nn.Module): the model to be trained
    - optim (torch.optim): the optimiser
    - log_interval (int): the number of epochs to wait before printing a log
    - loader (load_data.TrainLoader): the train data loader
    - epoch (int): the current epoch
    - args (Namespace): the namespace containing all the arguments required for training and building the network
    - writer (tensorboardX.SummaryWriter): the writer to use to log metrics evolution to tensorboardX

    '''

    if args.debug or args.benchmark:
        start_time = time.time()

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    intermVarDict = {"fullAttMap": None, "fullFeatMapSeq": None, "fullNormSeq":None}

    compute_latency = args.compute_latency and mode == "test"

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
        else:
            seg=None

        # Puting tensors on cuda
        if args.cuda:
            data, target = data.cuda(), target.cuda()

            if args.with_seg:
                seg = seg.cuda()

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
        loss = kwargs["lossFunc"](output, target, resDict, data).mean()

        # Other variables produced by the net
        if mode == "test":
            resDict["norm"] = torch.sqrt(torch.pow(resDict["features"],2).sum(dim=1,keepdim=True))
            intermVarDict = update.catIntermediateVariables(resDict, intermVarDict, validBatch)

        # Harware occupation
        update.updateHardWareOccupation(args.debug, args.benchmark, args.cuda, epoch, mode, args.exp_id, args.model_id,
                                        batch_idx)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target, seg,resDict,comp_spars=(mode=="test") and args.with_seg)
        metDictSample["Loss"] = loss.detach().data.item()
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        writePreds(output, target, epoch, args.exp_id, args.model_id, args.class_nb, batch_idx,mode)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch  >= 4*(50.0/args.val_batch_size) and args.debug:
            break

    if mode == "test":
        intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id, epoch, mode)

    writeSummaries(metrDict, totalImgNb, writer, epoch, mode, args.model_id, args.exp_id)

    if compute_latency:
        latency_list = np.array(latency_list)[:,np.newaxis]
        batchSize_list = np.array(batchSize_list)[:,np.newaxis]
        latency_list = np.concatenate((latency_list,batchSize_list),axis=1)
        np.savetxt("../results/{}/latency_{}_epoch{}.csv".format(args.exp_id,args.model_id,epoch),latency_list,header="latency,batch_size",delimiter=",")

    if args.debug or args.benchmark:
        totalTime = time.time() - start_time
        update.updateTimeCSV(epoch, mode, args.exp_id, args.model_id, totalTime, batch_idx)

    return metrDict[metricEarlyStop]

def writePreds(predBatch, targBatch, epoch, exp_id, model_id, class_nb, batch_idx,mode):
    csvPath = "../results/{}/{}_epoch{}_{}.csv".format(exp_id, model_id, epoch,mode)

    if (batch_idx == 0 and epoch == 1) or not os.path.exists(csvPath):
        with open(csvPath, "w") as text_file:
            print("targ," + ",".join(np.arange(class_nb).astype(str)), file=text_file)

    with open(csvPath, "a") as text_file:
        for i in range(len(predBatch)):
            print(str(targBatch[i].cpu().detach().numpy()) + "," + ",".join(
                predBatch[i].cpu().detach().numpy().astype(str)), file=text_file)

def writeSummaries(metrDict, totalImgNb, writer, epoch, mode, model_id, exp_id):
    ''' Write the metric computed during an evaluation in a tf writer and in a csv file

    Args:
    - metrDict (dict): the dictionary containing the value of metrics (not divided by the number of batch)
    - totalImgNb (int): the total number of images during the epoch
    - writer (tensorboardX.SummaryWriter): the writer to use to write the metrics to tensorboardX
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

    for metric in metrDict:
        if metric.find("Accuracy_") != -1:
            suffix = metric[metric.find("_"):]
            writer.add_scalars("Accuracy", {model_id + suffix + "_" + mode: metrDict[metric]}, epoch)
        else:
            writer.add_scalars(metric, {model_id + "_" + mode: metrDict[metric]}, epoch)

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

        # Start epoch is 1 if strict if false because strict=False means that it is another model which is being trained
        if strict:
            startEpoch = utils.findLastNumbers(init_path)+1
        else:
            startEpoch = 1

    return startEpoch

def preprocessAndLoadParams(init_path,cuda,net,strict):
    params = torch.load(init_path, map_location="cpu" if not cuda else None)

    params = addOrRemoveModule(params,net)
    paramCount = len(params.keys())
    params = removeBadSizedParams(params,net)
    if paramCount != len(params.keys()):
        strict=False
    params = addFeatModZoom(params,net)
    params = changeOldNames(params,net)

    res = net.load_state_dict(params, strict)

    # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
    if not res is None:
        missingKeys, unexpectedKeys = res
        if len(missingKeys) > 0:
            print("missing keys", missingKeys)
        if len(unexpectedKeys) > 0:
            print("unexpected keys", unexpectedKeys)

    return net

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


def init_process(args, rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args)

def initMasterNet(args):
    config = configparser.ConfigParser()

    config.read("../models/{}/{}.ini".format(args.exp_id,args.m_model_id))
    args_master = Bunch(config["default"])

    args_master.multi_gpu = args.multi_gpu

    argDic = args.__dict__
    mastDic = args_master.__dict__

    for arg in mastDic:
        if arg in argDic:
            if not argDic[arg] is None:
                if not type(argDic[arg]) is bool:
                    mastDic[arg] = type(argDic[arg])(mastDic[arg])
                else:
                    mastDic[arg] = str2bool(mastDic[arg]) if arg != "multi_gpu" else mastDic[arg]
            else:
                mastDic[arg] = None

    for arg in argDic:
        if not arg in mastDic:
            mastDic[arg] = argDic[arg]

    master_net = modelBuilder.netBuilder(args_master)

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
        master_net.load_state_dict(params, strict=True)

    master_net.eval()

    return master_net

def run(args,trial=None):
    writer = SummaryWriter("../results/{}".format(args.exp_id))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    if not trial is None:
        args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        args.optim = trial.suggest_categorical("optim", OPTIM_LIST)
        args.batch_size = trial.suggest_int("batch_size", 10, args.max_batch_size, log=True)
        args.dropout = trial.suggest_float("dropout", 0, 0.6,step=0.2)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

        if args.optim == "SGD":
            args.momentum = trial.suggest_float("momentum", 0., 0.9,step=0.1)
            args.use_scheduler = trial.suggest_categorical("use_scheduler",[True,False])

        if args.opt_data_aug:
            args.brightness = trial.suggest_float("brightness", 0, 0.5, step=0.05)
            args.saturation = trial.suggest_float("saturation", 0, 0.9, step=0.1)
            args.crop_ratio = trial.suggest_float("crop_ratio", 0.8, 1, step=0.05)

        if args.master_net:
            args.kl_temp = trial.suggest_float("kl_temp", 1, 21, step=5)
            args.kl_interp = trial.suggest_float("kl_interp", 0.1, 1, step=0.1)

            if args.transfer_att_maps:
                args.att_weights = trial.suggest_float("att_weights",0.001,4,log=True)
                args.att_pow = trial.suggest_int("att_pow",1,3,step=1)

        if args.opt_att_maps_nb:
            args.resnet_bil_nb_parts = trial.suggest_int("resnet_bil_nb_parts", 3, 64, log=True)

    trainLoader,_ = load_data.buildTrainLoader(args,withSeg=args.with_seg,reprVec=args.repr_vec)
    valLoader,_ = load_data.buildTestLoader(args,"val",withSeg=args.with_seg,reprVec=args.repr_vec)

    # Building the net
    net = modelBuilder.netBuilder(args)

    trainFunc = epochSeqTr
    valFunc = epochImgEval

    kwargsTr = {'log_interval': args.log_interval, 'loader': trainLoader, 'args': args, 'writer': writer}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    startEpoch = initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, args.start_mode,
                                                args.init_path, args.strict_init)

    kwargsTr["optim"],scheduler = getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,args.epochs,-1,net)

    epoch = startEpoch
    bestEpoch, worseEpochNb = getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    if args.maximise_val_metric:
        bestMetricVal = -np.inf
        isBetter = lambda x, y: x > y
    else:
        bestMetricVal = np.inf
        isBetter = lambda x, y: x < y

    if args.master_net:
        kwargsTr["master_net"] = initMasterNet(args)
        kwargsVal["master_net"] = kwargsTr["master_net"]

    lossFunc = Loss(args,reduction="mean")
    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc)
    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc

    if not args.only_test and not args.grad_cam:
        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            if not args.no_train:

                trainFunc(**kwargsTr)
                if not scheduler is None:
                    writer.add_scalars("LR", {args.model_id: scheduler.get_last_lr()}, epoch)
                    scheduler.step()
            else:
                if not args.no_val:
                    if args.model_id_no_train == "":
                        args.model_id_no_train = args.model_id
                    if args.exp_id_no_train == "":
                        args.exp_id_no_train = args.exp_id

                    net = preprocessAndLoadParams("../models/{}/model{}_epoch{}".format(args.exp_id_no_train, args.model_id_no_train, epoch),args.cuda,net,args.strict_init)

            if not (args.no_val or args.grad_exp):
                with torch.no_grad():
                    metricVal = valFunc(**kwargsVal)

                bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                                args.model_id, bestEpoch, epoch, net,
                                                                                isBetter, worseEpochNb)
                if trial is not None:
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

                with torch.no_grad():
                    testFunc(**kwargsTest)

                with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                    print("{},{}".format(args.model_id,bestEpoch),file=text_file)

    else:
        oldPath = "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, bestEpoch)
        os.rename(oldPath, oldPath.replace("best_epoch","trial{}_best_epoch".format(trial.number)))

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
    argreader.parser.add_argument('--grad_cam', type=str2bool, help='To compute grad cam instead of training or testing.')
    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader.parser.add_argument('--grad_exp', type=str, help='To store the gradients of the feature matrix.')

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
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    args = updateSeedAndNote(args)
    # Update the config args
    argreader.args = args
    # Write the arguments in a config file so the experiment can be re-run

    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id, args.model_id))
    print("Model :", args.model_id, "Experience :", args.exp_id)

    if args.distributed:
        size = args.distrib_size
        processes = []
        for rank in range(size):
            p = Process(target=init_process, args=(args, rank, size, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:

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
                    except RuntimeError as e:
                        if str(e).find("CUDA out of memory.") != -1:
                            gc.collect()
                            torch.cuda.empty_cache()
                            args.max_batch_size -= 5
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

            args.lr,args.batch_size = bestParamDict["lr"],int(bestParamDict["batch_size"])
            args.optim = OPTIM_LIST[int(bestParamDict["optim"])]
            args.only_test = True

            bestPath = glob.glob("../models/{}/model{}_trial{}_best_epoch*".format(args.exp_id,args.model_id,bestTrialId-1))[0]

            copyfile(bestPath, bestPath.replace("_trial{}".format(bestTrialId-1),""))

            run(args)


        if args.grad_exp:

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
                run(args)
            else:
                print("Already done")
        else:
            run(args)

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
