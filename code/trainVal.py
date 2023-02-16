import os
import sys
import glob
import time

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

from args import ArgReader,str2bool,addInitArgs,addValArgs,init_post_hoc_arg
import init_model
from loss import Loss,agregate_losses
import modelBuilder
import load_data
import metrics
import sal_metr_data_aug
import utils
import update

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

def epochSeqTr(model, optim, loader, epoch, args, **kwargs):

    model.train()

    print("Epoch", epoch, " : train")

    metrDict = None
    validBatch = 0
    totalImgNb = 0

    accumulated_size = 0
    acc_nb = 0

    featDic = {} if args.nce_weight > 0 else None
    target_list = []
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

        if args.sal_metr_mask:
            resDict,data = sal_metr_data_aug.apply_sal_metr_masks_and_update_dic(model,data,args.sal_metr_mask_prob,args.nce_weight,resDict)
                   
        resDict.update(model(data))
        output = resDict["pred"]
        loss_dic = kwargs["lossFunc"](output, target, resDict)
        loss_dic = agregate_losses(loss_dic)
        loss = loss_dic["loss"]/len(data)
        loss.backward()

        if accumulated_size == args.batch_size:
            model,optim,accumulated_size,acc_nb = optim_step(model,optim,acc_nb)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target,resDict)
        metDictSample = metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        if args.nce_weight > 0: 
            featDic = update.catFeat(resDict,featDic)
            target_list.append(target)

        validBatch += 1
        totalImgNb += len(data)

        if validBatch > 5 and args.debug:
            break
    
    if "nce_weight_updater" in kwargs:
        args.nce_weight = kwargs["nce_weight_updater"].update_nce_weight(metrDict)
        metrDict["nce_weight"] = args.nce_weight

    if not args.optuna:
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch))
        previous_epoch_model = "../models/{}/model{}_epoch{}".format(args.exp_id, args.model_id, epoch-1)
        if os.path.exists(previous_epoch_model):
            os.remove(previous_epoch_model)

        if args.nce_weight > 0: 
            target_list = torch.cat(target_list,dim=0)
            metrDict = metrics.separability_metric(featDic["feat_pooled"].detach().cpu(),featDic["feat_pooled_masked"].detach().cpu(),target_list,metrDict,args.seed,args.img_nb_per_class)
             
        writeSummaries(metrDict, totalImgNb, epoch, "train", args.model_id, args.exp_id)

    return metrDict

def epochImgEval(model, loader, epoch, args, mode="val",**kwargs):

    model.eval()

    print("Epoch", epoch, " : {}".format(mode))

    metrDict = None

    validBatch = 0
    totalImgNb = 0
    intermVarDict = {}
    
    featDic = {} if args.nce_weight > 0 else None
    target_list = []
    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % args.log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        # Puting tensors on cuda
        if args.cuda: data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        resDict = {}
        
        if args.sal_metr_mask:
            resDict,data = sal_metr_data_aug.apply_sal_metr_masks_and_update_dic(model,data,args.sal_metr_mask_prob,args.nce_weight,resDict)

        resDict.update(model(data))
        output = resDict["pred"]
        loss_dic = kwargs["lossFunc"](output, target, resDict)
        loss_dic = agregate_losses(loss_dic)

        # Metrics
        metDictSample = metrics.binaryToMetrics(output, target,resDict,comp_spars=(mode=="test"))
        metDictSample = metrics.add_losses_to_dic(metDictSample,loss_dic)
        metrDict = metrics.updateMetrDict(metrDict, metDictSample)

        # Other variables produced by the net
        if mode == "test":
            resDict["norm"] = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
            intermVarDict = update.catIntermediateVariables(resDict, intermVarDict)

        if args.nce_weight > 0: 
            target_list.append(target)
            featDic = update.catFeat(resDict,featDic)

        validBatch += 1
        totalImgNb += len(data)

        if validBatch  >= 2 and args.debug:
            break

    if mode == "test":
        intermVarDict = update.saveIntermediateVariables(intermVarDict, args.exp_id, args.model_id, epoch, mode)

    if args.nce_weight > 0: 
        target_list = torch.cat(target_list)
        metrDict = metrics.separability_metric(featDic["feat_pooled"].cpu(),featDic["feat_pooled_masked"].cpu(),target_list,metrDict,args.seed,args.img_nb_per_class)
    writeSummaries(metrDict, totalImgNb, epoch, mode, args.model_id, args.exp_id)

    return metrDict["Accuracy"]

def writeSummaries(metrDict, totalImgNb, epoch, mode, model_id, exp_id):
 
    for metric in metrDict.keys():
        if metric.find("Sep") == -1 and metric != "nce_weight":
            metrDict[metric] /= totalImgNb

    header_list = ["epoch"]
    header_list += [metric.lower().replace(" ", "_") for metric in metrDict.keys()]
    header = ",".join(header_list)

    csv_path = f"../results/{exp_id}/metrics_{model_id}_{mode}.csv"

    if not os.path.exists(csv_path):
        with open(csv_path, "w") as text_file:
           print(header, file=text_file) 

    with open(csv_path, "a") as text_file:
        print(epoch,file=text_file,end=",")
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]), file=text_file)

    return metrDict

def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=float, metavar='LR',
                                  help='learning rate')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                                  help='SGD momentum')
    argreader.parser.add_argument('--weight_decay', type=float, metavar='M',
                                  help='Weight decay')
    argreader.parser.add_argument('--use_scheduler', type=str2bool, metavar='M',
                                  help='To use a learning rate scheduler')
    argreader.parser.add_argument('--always_sched', type=str2bool, metavar='M',
                                  help='To always use a learning rate scheduler when optimizing hyper params')
    argreader.parser.add_argument('--sched_step_size', type=int, metavar='M',
                                  help='The number of epochs before reducing learning rate.')
    argreader.parser.add_argument('--sched_gamma', type=float, metavar='M',
                                  help='Multiplicative factor of learning rate decay')

    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                                  help='the optimizer to use (default: \'SGD\')')

    argreader.parser.add_argument('--bil_clus_soft_sched', type=str2bool, metavar='BOOL',
                                  help='Added schedule to increase temperature of the softmax of the bilinear cluster model.')

    return argreader

def addLossTermArgs(argreader):
    argreader.parser.add_argument('--nll_weight', type=float, metavar='FLOAT',
                                  help='The weight of the negative log-likelihood term in the loss function.')
    argreader.parser.add_argument('--nce_weight', type=str, metavar='FLOAT',
                                  help='The weight of the saliency metric mask in the loss function. Can be set to "scheduler".')
    argreader.parser.add_argument('--nce_sched_start', type=float, metavar='FLOAT',
                                  help='The initial value of nce_weight loss term.')
    return argreader


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

    startEpoch = init_model.initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, args.start_mode,
                                                args.init_path,args.optuna)

    kwargsTr["optim"],scheduler = init_model.getOptim_and_Scheduler(args.optim, args.lr,args.momentum,args.weight_decay,args.use_scheduler,startEpoch,net,args.sched_step_size,args.sched_gamma)

    if args.nce_weight == "scheduler":
        nce_weight_updater = update.NCEWeightUpdater(args)
        args.nce_weight = nce_weight_updater.init_nce_weight()
        kwargsTr["nce_weight_updater"] = nce_weight_updater
    else:
        args.nce_weight = float(args.nce_weight)

    epoch = startEpoch
    bestEpoch, worseEpochNb = init_model.getBestEpochInd_and_WorseEpochNb(args.start_mode, args.exp_id, args.model_id, epoch)

    bestMetricVal = -np.inf
    isBetter = lambda x, y: x > y

    if args.master_net:
        kwargsTr["master_net"] = init_model.initMasterNet(args)
        kwargsVal["master_net"] = kwargsTr["master_net"]

    lossFunc = Loss(args,reduction="sum")

    if args.multi_gpu:
        lossFunc = torch.nn.DataParallel(lossFunc)

    kwargsTr["lossFunc"],kwargsVal["lossFunc"] = lossFunc,lossFunc


    if not args.only_test:

        actual_bs = args.batch_size if args.batch_size < args.max_batch_size_single_pass else args.max_batch_size_single_pass
        args.batch_per_epoch = len(trainLoader.dataset)//actual_bs if len(trainLoader.dataset) > actual_bs else 1

        while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

            kwargsTr["epoch"], kwargsVal["epoch"] = epoch, epoch
            kwargsTr["model"], kwargsVal["model"] = net, net

            metrDict = trainFunc(**kwargsTr)
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

            net = init_model.preprocessAndLoadParams("../models/{}/model{}_best_epoch{}".format(args.exp_id, args.model_id, bestEpoch),args.cuda,net)

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

    argreader.parser.add_argument('--sal_metr_mask', type=str2bool, help='To apply the masking of attention metrics during training.')

    argreader.parser.add_argument('--sal_metr_mask_prob', type=float, help='The probability to apply saliency metrics masking.')
    argreader.parser.add_argument('--sal_metr_mask_remove_masked_obj',type=str2bool, help='Set to True to remove terms masked by the DAUC and ADD metrics.')

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)
    argreader = init_post_hoc_arg(argreader)

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
