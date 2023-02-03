import os
import sys
import glob
import time
import configparser
from shutil import copyfile
import gc
import subprocess

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import optuna
import sqlite3

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
import sal_metr_data_aug
import utils
import update
from models import inter_by_parts,protopnet

def remove_excess_examples(data,target,accumulated_size,batch_size):
    if accumulated_size + data.size(0) > batch_size:

        if batch_size-accumulated_size < 2*torch.cuda.device_count():
            data = data[:2*torch.cuda.device_count()]
            target = target[:2*torch.cuda.device_count()]
        else:
            data = data[:batch_size-accumulated_size]
            target = target[:batch_size-accumulated_size]
        accumulated_size = batch_size
    else:
        accumulated_size += data.size(0)

    return data,target,accumulated_size

def master_net_inference(data,kwargs,resDict):
    with torch.no_grad():
        mastDict = kwargs["master_net"](data)
        resDict["master_net_pred"] = mastDict["pred"]
        resDict["master_net_features"] = mastDict["features"]
        if "attMaps" in mastDict:
            resDict["master_net_attMaps"] = mastDict["attMaps"]
    return resDict

def optim_step(model,optim,acc_nb):
    #Scale gradients (required when using gradient accumulation)
    for p in model.parameters():
        if p.grad is not None:
            p.grad /= acc_nb
    optim.step()
    optim.zero_grad()
    accumulated_size = 0
    acc_nb = 0

    return model,optim,accumulated_size,acc_nb

class Loss(torch.nn.Module):

    def __init__(self,args,reduction="mean"):
        super(Loss, self).__init__()
        self.reduction = reduction
        self.args= args
    def forward(self,output,target,resDict):
        return computeLoss(self.args,output, target, resDict,reduction=self.reduction).unsqueeze(0)

def computeLoss(args, output, target, resDict,reduction="mean"):

    if args.master_net and ("master_net_pred" in resDict):
        kl = F.kl_div(F.log_softmax(output/args.kl_temp, dim=1),F.softmax(resDict["master_net_pred"]/args.kl_temp, dim=1),reduction="batchmean")
        ce = F.cross_entropy(output, target)
        loss = args.nll_weight*(kl*args.kl_interp*args.kl_temp*args.kl_temp+ce*(1-args.kl_interp))
    else:
        loss = args.nll_weight * F.cross_entropy(output, target,reduction=reduction)
        if args.inter_by_parts:
            loss += 0.5*inter_by_parts.shapingLoss(resDict["attMaps"],args.resnet_bil_nb_parts,args)
        if args.abn:
            loss += args.nll_weight*F.cross_entropy(resDict["att_outputs"], target,reduction=reduction)

    return loss

def epochSeqTr(model, optim, loader, epoch, args, **kwargs):

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0

    accumulated_size = 0
    acc_nb = 0

    for batch_idx, batch in enumerate(loader):
        optim.zero_grad()

        if batch_idx % args.log_interval == 0:
            processedImgNb = batch_idx * len(batch[0])
            print("\t", processedImgNb, "/", len(loader.dataset))

        data, target = batch[0], batch[1]

        #Removing excess samples (if training is done by accumulating gradients)
        data,target,accumulated_size = remove_excess_examples(data,target,accumulated_size,args.batch_size)

        acc_nb += 1

        if args.cuda:
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        resDict = {}

        if args.master_net:
            resDict = master_net_inference(data,kwargs,resDict)

        if args.att_metr_mask and epoch >= args.att_metr_mask_start_epoch:
            data = sal_metr_data_aug.apply_att_metr_masks(model,data)

        resDict_model = model(data)
        resDict.update(resDict_model)

        output = resDict["pred"]

        loss = kwargs["lossFunc"](output, target, resDict).mean()
        loss.backward()

        if accumulated_size == args.batch_size:
            model,optim,accumulated_size,acc_nb = optim_step(model,optim,acc_nb)

        loss = loss.detach().data.item()

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target,resDict)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch > 3 and args.debug:
            break

    if not args.optuna:
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
        writeSummaries(metrDict, totalImgNb, epoch, "train", args.model_id, args.exp_id)

def epochImgEval(model, loader, epoch, args, mode="val",**kwargs):

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    intermVarDict = {"fullAttMap": None, "fullFeatMapSeq": None, "fullNormSeq":None}

    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % args.log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        # Puting tensors on cuda
        if args.cuda: data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        # Computing predictions
        resDict = model(data)

        output = resDict["pred"]

        # Loss
        loss = kwargs["lossFunc"](output, target, resDict).mean()

        # Other variables produced by the net
        if mode == "test":
            resDict["norm"] = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
            intermVarDict = update.catIntermediateVariables(resDict, intermVarDict, validBatch)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target,resDict,comp_spars=(mode=="test"))

        metDictSample["Loss"] = loss

        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        validBatch += 1
        totalImgNb += target.size(0)

        if validBatch  >= 4*(50.0/args.val_batch_size) and args.debug:
            break

    if mode == "test":
        intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id, epoch, mode)

    writeSummaries(metrDict, totalImgNb, epoch, mode, args.model_id, args.exp_id)

    return metrDict["Accuracy"]

def writeSummaries(metrDict, totalImgNb, epoch, mode, model_id, exp_id):
 
    for metric in metrDict.keys():
        metrDict[metric] /= totalImgNb

    header = ",".join([metric.lower().replace(" ", "_") for metric in metrDict.keys()])

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id, model_id, epoch, mode), "a") as text_file:
        print(header, file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def getOptim_and_Scheduler(optimStr, lr,momentum,weightDecay,useScheduler,lastEpoch,net):

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
        for _ in range(lastEpoch-1):
            scheduler.step()
    else:
        scheduler = None

    return optim, scheduler

def initialize_Net_And_EpochNumber(net, exp_id, model_id, cuda, start_mode, init_path,optuna):

    if start_mode == "auto":
        if (not optuna) and len(glob.glob("../models/{}/model{}_epoch*".format(exp_id, model_id))) > 0:
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

        net = preprocessAndLoadParams(init_path,cuda,net)

        startEpoch = utils.findLastNumbers(init_path)+1

    return startEpoch

def preprocessAndLoadParams(init_path,cuda,net):
    print("Init from",init_path)
    params = torch.load(init_path, map_location="cpu" if not cuda else None)
    params = addOrRemoveModule(params,net)
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
            paramsFormated[keyFormat] = params[key]
        params = paramsFormated
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

    return argreader

def initMasterNet(args):
    config = configparser.ConfigParser()

    config.read("../models/{}/{}.ini".format(args.exp_id,args.m_model_id))
    args_master = utils.Bunch(config["default"])

    args_master.multi_gpu = args.multi_gpu

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

def run(args,trial):

    args.lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    args.optim = trial.suggest_categorical("optim", ["Adam", "AMSGrad", "SGD"])

    if args.max_batch_size <= 12:
        minBS = 4
    else:
        minBS = 12

    args.batch_size = trial.suggest_int("batch_size", minBS, args.max_batch_size, log=True)
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

    if args.opt_att_maps_nb:
        args.resnet_bil_nb_parts = trial.suggest_int("resnet_bil_nb_parts", 3, 64, log=True)

    value = train(args,trial)
    return value

def save_git_status(args):

    path_start = f"../models/{args.exp_id}/model{args.model_id}"

    if hasattr(args,"trial_id") and args.trial_id is not None:
        path_start += "_"+str(args.trial_id)

    cmd_list = ["git status","git diff","git rev-parse --short HEAD"]
    labels = ["status","diff","commit"]

    for labels,cmd in zip(labels,cmd_list):
        path = path_start + "_git_"+labels+".txt"
        output = subprocess.check_output(cmd.split(" "),text=True)
        with open(path,"w") as file:
            print(output,file=file)

def train(args,trial):

    if not trial is None:
        args.trial_id = trial.number

    save_git_status(args)

    if not args.only_test:
        trainLoader,_ = load_data.buildTrainLoader(args)
    else:
        trainLoader = None
    valLoader,_ = load_data.buildTestLoader(args,"val")

    # Building the net
    net = modelBuilder.netBuilder(args)

    trainFunc = epochSeqTr
    valFunc = epochImgEval

    kwargsTr = {'loader': trainLoader, 'args': args}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader

    startEpoch = initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, args.start_mode,
                                                args.init_path,args.optuna)

    kwargsTr["optim"],scheduler = getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,startEpoch,net)

    epoch = startEpoch
    bestEpoch, worseEpochNb = getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    bestMetricVal = -np.inf
    isBetter = lambda x, y: x > y

    if args.master_net:
        kwargsTr["master_net"] = initMasterNet(args)
        kwargsVal["master_net"] = kwargsTr["master_net"]

    lossFunc = Loss(args,reduction="mean")

    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc)

    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc

    if not args.only_test:

        actual_bs = args.batch_size if args.batch_size < args.max_batch_size_single_pass else args.max_batch_size_single_pass
        args.batch_per_epoch = len(trainLoader.dataset)//actual_bs if len(trainLoader.dataset) > actual_bs else 1

        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            if args.protonet:
                if epoch - startEpoch > args.protonet_warm:
                    protopnet.joint(net.module.firstModel.protopnet)
                else:
                    protopnet.warm_only(net.module.firstModel.protopnet)
            elif args.prototree:
                if  epoch-startEpoch < args.protonet_warm:
                    net.module.firstModel.featMod.requires_grad= False 
                else:
                    net.module.firstModel.featMod.requires_grad= True

            trainFunc(**kwargsTr)
            if not scheduler is None:
                scheduler.step()

            if not args.no_val:
                with torch.no_grad():
                    metricVal = valFunc(**kwargsVal)

                bestEpoch, bestMetricVal, worseEpochNb = update.updateBestModel(metricVal, bestMetricVal, args.exp_id,
                                                                            args.model_id, bestEpoch, epoch, net,
                                                                            isBetter, worseEpochNb)
                if trial is not None:
                    trial.report(metricVal, epoch)

            epoch += 1

    if trial is None:
        if args.run_test:

            testFunc = valFunc

            kwargsTest = kwargsVal
            kwargsTest["mode"] = "test"

            testLoader,_ = load_data.buildTestLoader(args, "test")

            kwargsTest['loader'] = testLoader

            net = preprocessAndLoadParams("../models/{}/model{}_best_epoch{}".format(args.exp_id, args.model_id, bestEpoch),args.cuda,net)

            kwargsTest["model"] = net
            kwargsTest["epoch"] = bestEpoch

            with torch.no_grad():
                testFunc(**kwargsTest)

            with open("../results/{}/test_done.txt".format(args.exp_id),"a") as text_file:
                print("{},{}".format(args.model_id,bestEpoch),file=text_file)

    else:

        oldPath = "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, bestEpoch)
        os.rename(oldPath, oldPath.replace("best_epoch","trial{}_best_epoch".format(trial.number)))

        with open("../results/{}/{}_{}_valRet.csv".format(args.exp_id,args.model_id,trial.number),"w") as text:
            print(metricVal,file=text)

        return metricVal

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--no_val', type=str2bool, help='To not compute the validation')
    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')

    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader.parser.add_argument('--trial_id', type=int, help='The trial ID. Useful for grad exp during test')

    argreader.parser.add_argument('--att_metr_mask', type=str2bool, help='To apply the masking of attention metrics during training.')
    argreader.parser.add_argument('--att_metr_mask_start_epoch', type=int, help='The epoch at which to start applying the masking.')

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    # The folders where the experience file will be written
    if not os.path.exists("../vis/{}".format(args.exp_id)):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not os.path.exists("../results/{}".format(args.exp_id)):
        os.makedirs("../results/{}".format(args.exp_id))
    if not os.path.exists("../models/{}".format(args.exp_id)):
        os.makedirs("../models/{}".format(args.exp_id))

    args = update.updateSeedAndNote(args)

    # Update the config args
    argreader.args = args
    # Write the arguments in a config file so the experiment can be re-run

    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id, args.model_id))
    print("Model :", args.model_id, "Experience :", args.exp_id)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

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

        args.only_test = True

        print("bestTrialId-1",bestTrialId-1)
        bestPath = glob.glob("../models/{}/model{}_trial{}_best_epoch*".format(args.exp_id,args.model_id,bestTrialId-1))[0]
        print(bestPath)

        copyfile(bestPath, bestPath.replace("_trial{}".format(bestTrialId-1),""))

        train(args,None)

    else:
        train(args,None)

if __name__ == "__main__":
    main()
