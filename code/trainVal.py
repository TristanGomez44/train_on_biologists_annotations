import os
import sys
import glob

import args
from args import ArgReader
from args import str2bool
from args import str2StrList

import numpy as np
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

import matplotlib.cm as cm
import matplotlib.pyplot as plt

plt.switch_backend('agg')

import modelBuilder
import load_data
import metrics
import utils
import update
import warnings

import torch.distributed as dist
from torch.multiprocessing import Process

import time




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
    - width (int): the width of the triangular window (i.e. the number of steps over which the window is spreading)

    '''

    start_time = time.time() if args.debug or args.benchmark else None

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    allOut, allGT = None, None

    for batch_idx, batch in enumerate(loader):

        if (batch_idx % log_interval == 0):
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        resDict = model(data)

        output = resDict["pred"]

        loss = computeLoss(args, output, target, resDict, data)

        loss.backward()

        if args.distributed:
            average_gradients(model)

        optim.step()
        update.updateHardWareOccupation(args.debug, args.benchmark, args.cuda, epoch, "train", args.exp_id,
                                        args.model_id, batch_idx)
        optim.zero_grad()

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target, resDict)
        metDictSample["Loss"] = loss.detach().data.item()
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1

        if validBatch > 15 and args.debug:
            break

    # If the training set is empty (which we might want to just evaluate the model), then allOut and allGT will still be None
    if validBatch > 0:

        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
        if (not args.save_all) and os.path.exists(
                "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch - 1)):
            os.remove("../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch - 1))
        writeSummaries(metrDict, validBatch, writer, epoch, "train", args.model_id, args.exp_id)

    if args.debug or args.benchmark:
        totalTime = time.time() - start_time
        update.updateTimeCSV(epoch, "train", args.exp_id, args.model_id, totalTime, batch_idx)

def computeLoss(args, output, target, resDict, data):

    loss = args.nll_weight * F.cross_entropy(output, target)
    if args.pn_reinf_weight > 0:
        loss += pn_reinf_term(args.pn_reinf_weight, resDict, target, args.pn_reinf_weight_baseline, args.score_reward)
    if args.aux_mod_nll_weight > 0:
        loss += aux_model_loss_term(args.aux_mod_nll_weight, resDict, data, target)
    if args.zoom_nll_weight > 0:
        loss += zoom_loss_term(args.zoom_nll_weight, resDict, data, target)
    return loss

def pn_reinf_term(pn_reinf_weight, resDict, target, pn_reinf_weight_baseline, score_reward):
    flatInds = resDict['flatInds']
    pi = resDict['probs'][torch.arange(flatInds.size(0), dtype=torch.long).unsqueeze(1), flatInds]
    acc = (resDict["pred"].detach().argmax(dim=-1) == target)

    if score_reward:
        _, topk = torch.topk(resDict["pred"], 200)
        ranks = (topk == target.unsqueeze(1)).nonzero()[:, 1].type(torch.cuda.FloatTensor)
        reward = ((200 - ranks) / 200).unsqueeze(1)
    else:
        reward = (acc * 1.0).unsqueeze(1)

    if pn_reinf_weight_baseline > 0.0:
        baseline = pn_reinf_weight_baseline * F.mse_loss(resDict['baseline'], reward)
        reward = baseline.detach() - reward

    loss_reinforce = torch.mean(torch.mean(-torch.log(pi) * reward, dim=1))
    return pn_reinf_weight * loss_reinforce

def aux_model_loss_term(aux_model_weight, resDict, data, target):
    return aux_model_weight * F.cross_entropy(resDict["auxPred"], target)

def zoom_loss_term(zoom_nll_weight, resDict, data, target):
    return zoom_nll_weight * F.cross_entropy(resDict["pred_zoom"], target)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if not param.grad is None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size


def epochImgEval(model, log_interval, loader, epoch, args, writer, metricEarlyStop, mode="val"):
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

    allOut = None
    allGT = None
    intermVarDict = {"fullAttMap": None, "fullFeatMapSeq": None, "fullAffTransSeq": None, "fullPointsSeq": None,"fullPointsWeightSeq":None,
                     "fullPointsSeq_pureText": None,"fullPointsWeightSeq_pureText":None,"fullPointsSeqDropped":None,"fullPNReconstSeq": None,"fullProbMap":None,\
                     "fullReconstSeq":None}

    compute_latency = args.compute_latency and mode == "test"

    if compute_latency:
        latency_list=[]
        batchSize_list = []
    else:
        latency_list,batchSize_list =None,None

    for batch_idx, (data, target) in enumerate(loader):

        if (batch_idx % log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        # Puting tensors on cuda
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        # Computing predictions
        if compute_latency:
            lat_start_time = time.time()
            resDict = model(data)
            latency_list.append(time.time()-lat_start_time)
            batchSize_list.append(data.size(0))
        else:
            resDict = model(data)

        output = resDict["pred"]

        # Loss
        loss = computeLoss(args, output, target, resDict, data)

        # Other variables produced by the net
        intermVarDict = update.catIntermediateVariables(resDict, intermVarDict, validBatch, args.save_all)

        # Harware occupation
        update.updateHardWareOccupation(args.debug, args.benchmark, args.cuda, epoch, mode, args.exp_id, args.model_id,
                                        batch_idx)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target, resDict)
        metDictSample["Loss"] = loss.detach().data.item()
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        writePreds(output, target, epoch, args.exp_id, args.model_id, args.class_nb, batch_idx)

        validBatch += 1

        if validBatch > 15 and args.debug:
            break


    intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id, epoch, mode,
                                                     args.save_all)

    writeSummaries(metrDict, validBatch, writer, epoch, mode, args.model_id, args.exp_id)

    if compute_latency:
        latency_list = np.array(latency_list)[:,np.newaxis]
        batchSize_list = np.array(batchSize_list)[:,np.newaxis]
        latency_list = np.concatenate((latency_list,batchSize_list),axis=1)
        np.savetxt("../results/{}/latency_{}_epoch{}.csv".format(args.exp_id,args.model_id,epoch),latency_list,header="latency,batch_size",delimiter=",")

    if args.debug or args.benchmark:
        totalTime = time.time() - start_time
        update.updateTimeCSV(epoch, mode, args.exp_id, args.model_id, totalTime, batch_idx)

    return metrDict[metricEarlyStop]


def writePreds(predBatch, targBatch, epoch, exp_id, model_id, class_nb, batch_idx):
    csvPath = "../results/{}/{}_epoch{}.csv".format(exp_id, model_id, epoch)

    if (batch_idx == 0 and epoch == 1) or not os.path.exists(csvPath):
        with open(csvPath, "w") as text_file:
            print("targ," + ",".join(np.arange(class_nb).astype(str)), file=text_file)

    with open(csvPath, "a") as text_file:
        for i in range(len(predBatch)):
            print(str(targBatch[i].cpu().detach().numpy()) + "," + ",".join(
                predBatch[i].cpu().detach().numpy().astype(str)), file=text_file)


def writeSummaries(metrDict, sampleNb, writer, epoch, mode, model_id, exp_id):
    ''' Write the metric computed during an evaluation in a tf writer and in a csv file

    Args:
    - metrDict (dict): the dictionary containing the value of metrics (not divided by the number of batch)
    - batchNb (int): the total number of batches during the epoch
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
        metrDict[metric] /= sampleNb

    for metric in metrDict:
        if metric.find("Accuracy_") != -1:
            suffix = metric[metric.find("_"):]
            writer.add_scalars("Accuracy", {model_id + suffix + "_" + mode: metrDict[metric]}, epoch)
        else:
            writer.add_scalars(metric, {model_id + "_" + mode: metrDict[metric]}, epoch)

    if not os.path.exists("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id, model_id, epoch, mode)):
        header = [metric.lower().replace(" ", "_") for metric in metrDict.keys()]
    else:
        header = ""

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
            kwargs = {'lr':lr}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'lr':lr,'amsgrad': True}

    optim = optimConst(net.parameters(), **kwargs)

    if useScheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, maxEpoch,last_epoch=lastEpoch)
    else:
        scheduler = None

    return optim, scheduler

def initialize_Net_And_EpochNumber(net, exp_id, model_id, cuda, start_mode, init_path, strict, init_pn_path):
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

        if init_pn_path != "None":
            pn_params = torch.load(init_pn_path, map_location="cpu" if not cuda else None)
            pn_params_filtered = {}
            for key in pn_params.keys():
                if key.find("sa1_module.conv.local_nn.0.0.weight") == -1 and key.find("lin3") == -1:
                    pn_params_filtered[key] = pn_params[key]

            res = net.secondModel.pn2.load_state_dict(pn_params_filtered, False)

        # Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id, model_id))
        startEpoch = 1

    elif start_mode == "fine_tune":

        if init_path == "None":
            init_path = \
                sorted(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id)), key=utils.findLastNumbers)[-1]

        print(init_path)
        params = torch.load(init_path, map_location="cpu" if not cuda else None)

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

        # else:

        # Removing keys corresponding to parameter which shape are different in the checkpoint and in the current model
        # For example, this is necessary to load a model trained on n classes to bootstrap a model with m != n classes.
        keysToRemove = []
        for key in params.keys():
            if key in net.state_dict().keys():
                if net.state_dict()[key].size() != params[key].size():
                    keysToRemove.append(key)
        for key in keysToRemove:
            params.pop(key)

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

        res = net.load_state_dict(params, strict)

        # Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
        if not res is None:
            missingKeys, unexpectedKeys = res
            if len(missingKeys) > 0:
                print("missing keys", missingKeys)
            if len(unexpectedKeys) > 0:
                print("unexpected keys", unexpectedKeys)

        # Start epoch is 1 if strict if false because strict=False means that it is another model which is being trained
        if strict:
            startEpoch = utils.findLastNumbers(init_path)
        else:
            startEpoch = 1

    return startEpoch


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
    argreader.parser.add_argument('--init_pn_path', type=str, metavar='SM',
                                  help='The path to the weight file of a pn model.')
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

    argreader.parser.add_argument('--pn_reinf_weight', type=float, metavar='FLOAT',
                                  help='The weight of the reinforcement term in the loss function when using a reinforcement learning.')
    argreader.parser.add_argument('--pn_reinf_weight_baseline', type=float, metavar='FLOAT',
                                  help='The weight of the reinforcement baseline term in the loss function when using a reinforcement learning.')
    argreader.parser.add_argument('--pn_reinf_score_reward', type=args.str2bool, metavar='BOOL',
                                  help='Whether to calculate the reinforcement learning reward with topk score')

    return argreader


def init_process(args, rank, size, fn, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(args)


def run(args):
    writer = SummaryWriter("../results/{}".format(args.exp_id))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    trainLoader, trainDataset = load_data.buildTrainLoader(args)
    valLoader = load_data.buildTestLoader(args, "val")

    # Building the net
    net = modelBuilder.netBuilder(args)

    trainFunc = epochSeqTr
    valFunc = epochImgEval

    kwargsTr = {'log_interval': args.log_interval, 'loader': trainLoader, 'args': args, 'writer': writer}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    startEpoch = initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, args.start_mode,
                                                args.init_path, args.strict_init, args.init_pn_path)

    kwargsTr["optim"],scheduler = getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,args.epochs,-1,net)

    epoch = startEpoch
    bestEpoch, worseEpochNb = getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    if args.maximise_val_metric:
        bestMetricVal = -np.inf
        isBetter = lambda x, y: x > y
    else:
        bestMetricVal = np.inf
        isBetter = lambda x, y: x < y

    if not args.only_test:
        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            if not args.no_train:
                if args.smooth_features:
                    smoothKerSize = update.updateSmoothKer(net,epoch,args.smooth_features_sched_step,args.smooth_features_start_size,startEpoch)
                    writer.add_scalars("SmoothKerSize", {args.model_id:smoothKerSize}, epoch)

                trainFunc(**kwargsTr)
                if not scheduler is None:
                    writer.add_scalars("LR", {args.model_id: scheduler.get_lr()}, epoch)
                    scheduler.step()
            else:
                if not args.no_val:
                    net.load_state_dict(torch.load(
                        "../models/{}/model{}_epoch{}".format(args.exp_id_no_train, args.model_id_no_train, epoch),
                        map_location="cpu" if not args.cuda else None))

            if not args.no_val:
                with torch.no_grad():
                    metricVal = valFunc(**kwargsVal)

                bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                                args.model_id, bestEpoch, epoch, net,
                                                                                isBetter, worseEpochNb)

            epoch += 1

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

            testLoader = load_data.buildTestLoader(args, "test")

            kwargsTest['loader'] = testLoader

            net.load_state_dict(torch.load("../models/{}/model{}_best_epoch{}".format(args.exp_id, args.model_id, bestEpoch),
                                           map_location="cpu" if not args.cuda else None))
            kwargsTest["model"] = net
            kwargsTest["epoch"] = bestEpoch

            with torch.no_grad():
                testFunc(**kwargsTest)

            with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                print("{},{}".format(args.model_id,bestEpoch),file=text_file)

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
    argreader.parser.add_argument('--save_all', type=str2bool,
                                  help="Whether to save network weights at each epoch.")

    argreader.parser.add_argument('--no_val', type=str2bool, help='To not compute the validation')
    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')
    argreader.parser.add_argument('--compute_latency', type=str2bool, help='To write in a file the latency at each forward pass.')

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
        run(args)


if __name__ == "__main__":
    main()
