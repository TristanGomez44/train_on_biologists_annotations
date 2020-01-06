import os
import sys

import args
from args import ArgReader
from args import str2bool
from args import str2StrList

import glob

import numpy as np
import torch
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import torch.backends.cudnn as cudnn
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from sklearn.metrics import roc_auc_score

import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import modelBuilder
import load_data
import metrics
import utils
import update
import warnings
import formatData
from attack import FGSM
import imageio

from skimage import transform,io
from skimage import img_as_ubyte

import cv2
import psutil
#warnings.simplefilter('error', UserWarning)

import torch.distributed as dist
from torch.multiprocessing import Process

import time
import subprocess

def epochSeqTr(model,optim,log_interval,loader, epoch, args,writer,**kwargs):
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

    if args.debug:
        start_time = time.time()

    model.train()

    print("Epoch",epoch," : train")

    metrDict = metrics.emptyMetrDict(args.uncertainty)

    validBatch = 0

    allOut = None
    allGT = None

    for batch_idx,(data,target,_,_,timeElapsedTensor) in enumerate(loader):

        if (batch_idx % log_interval == 0):
            print("\t",batch_idx*len(data)*len(data[0]),"/",len(loader.dataset))

        #Puting tensors on cuda
        if args.cuda:
            data, target,timeElapsedTensor = data.cuda(), target.cuda(),timeElapsedTensor.cuda()

        #Computing predictions
        output = model(data,timeElapsedTensor)["pred"]


        #Computing loss
        output = output[:,args.train_step_to_ignore:output.size(1)-args.train_step_to_ignore]
        target = target[:,args.train_step_to_ignore:target.size(1)-args.train_step_to_ignore]

        if args.regression:
            #Converting the output of the sigmoid between 0 and 1 to a scale between -0.5 and class_nb+0.5
            output = (torch.sigmoid(output)*(args.class_nb+1)-0.5)
            loss = F.mse_loss(output.view(-1),target.view(-1).float())
        elif args.uncertainty:
            loss = uncertaintyLoss(F.softplus(output)+1,target,model,data,args.uncer_loss_type,args.uncer_exact_inf_div,args.uncert_inf_div_weight,args.uncer_ll_ratio_weight,args.uncer_max_adv_entr_weight,args.class_nb)
        else:
            loss = F.cross_entropy(output.view(output.size(0)*output.size(1),-1), target.view(-1))

        loss.backward()
        if args.distributed:
            average_gradients(model)

        optim.step()
        if validBatch <= 10 and args.debug:
            if args.cuda:
                updateOccupiedGPURamCSV(epoch,"train",args.exp_id,args.model_id)
            updateOccupiedRamCSV(epoch,"train",args.exp_id,args.model_id)
            updateOccupiedCPUCSV(epoch,"train",args.exp_id,args.model_id)
        optim.zero_grad()

        #Metrics
        metDictSample = metrics.binaryToMetrics(output,target,model.transMat,args.regression,args.uncertainty)
        #for key in metDictSample.keys():
        #    metrDict[key] += metDictSample[key]
        metDictSample["Loss"] = loss.detach().data.item()

        metrDict = metrics.updateMetrDict(metrDict,metDictSample)

        validBatch += 1

        if validBatch > 3 and args.debug:
            break

    #If the training set is empty (which we might want to just evaluate the model), then allOut and allGT will still be None
    if validBatch > 0:
        torch.save(model.state_dict(), "../models/{}/model{}_epoch{}".format(args.exp_id,args.model_id, epoch))
        writeSummaries(metrDict,validBatch,writer,epoch,"train",args.model_id,args.exp_id)

    if args.debug:
        totalTime = time.time() - start_time
        updateTimeCSV(epoch,"train",args.exp_id,args.model_id,totalTime)

def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        if not param.grad is None:
            dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
            param.grad.data /= size

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [x for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

def updateOccupiedGPURamCSV(epoch,mode,exp_id,model_id):

    occRamDict = get_gpu_memory_map()

    csvPath = "../results/{}/{}_occRam_{}.csv".format(exp_id,model_id,mode)

    if not os.path.exists(csvPath):
        with open(csvPath,"w") as text_file:
            print("epoch,"+",".join([str(device) for device in occRamDict.keys()]),file=text_file)
            print(str(epoch)+","+",".join([occRamDict[device] for device in occRamDict.keys()]),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+",".join([occRamDict[device] for device in occRamDict.keys()]),file=text_file)

def updateOccupiedCPUCSV(epoch,mode,exp_id,model_id):

    cpuOccList = [x / psutil.cpu_count() * 100 for x in psutil.getloadavg()]

    csvPath = "../results/{}/{}_cpuLoad_{}.csv".format(exp_id,model_id,mode)

    if not os.path.exists(csvPath):
        with open(csvPath,"w") as text_file:
            print("epoch,"+",".join([str(i) for i in range(len(cpuOccList))]),file=text_file)
            print(str(epoch)+","+",".join([cpuOcc for cpuOcc in cpuOccList]),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+",".join([cpuOcc for cpuOcc in cpuOccList]),file=text_file)

def updateOccupiedRamCSV(epoch,mode,exp_id,model_id):

    ramOcc = psutil.virtual_memory()._asdict()["percent"]

    csvPath = "../results/{}/{}_occCPURam_{}.csv".format(exp_id,model_id,mode)

    if not os.path.exists(csvPath):
        with open(csvPath,"w") as text_file:
            print("epoch,"+","+"percent",file=text_file)
            print(str(epoch)+","+str(ramOcc),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+str(ramOcc),file=text_file)

def updateTimeCSV(epoch,mode,exp_id,model_id,totalTime):

    csvPath = "../results/{}/{}_time_{}.csv".format(exp_id,model_id,mode)

    if not os.path.exists(csvPath):
        with open(csvPath,"w") as text_file:
            print("epoch,"+","+"time",file=text_file)
            print(str(epoch)+","+str(totalTime),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+str(totalTime),file=text_file)

def logBeta(alpha):
    alpha_0 = alpha.sum(-1)
    return torch.lgamma(alpha).sum(-1) - torch.lgamma(alpha_0)

def uncertaintyLoss(alpha,target,model,data,lossType,exactInfDiv,infDivWeight,llRatioWeight,maxAdvEntrWeight,classNb):

    meanDistr = alpha/alpha.sum(dim=-1,keepdim=True)

    target = target.view(-1)
    # One hot encoding buffer that you create out of the loop and just keep reusing
    target_one_hot = torch.FloatTensor(target.size(0), classNb).to(target.device)

    # In your for loop
    #target.to("cpu")
    target_one_hot.zero_()
    target_one_hot.scatter_(1, target.view(-1,1), 1)

    if lossType == "MSE":
        loss = F.mse_loss(meanDistr.view(meanDistr.size(0)*meanDistr.size(1),-1),target_one_hot.float())
    elif lossType == "CE":
        loss = F.cross_entropy(meanDistr.view(meanDistr.size(0)*meanDistr.size(1),-1), target.view(-1))
    else:
        raise ValueError("Unknown loss type : ",lossType)

    assert 0<llRatioWeight and llRatioWeight<=1

    ############# Information divergence term #############

    alpha = alpha.view(alpha.size(0)*alpha.size(1),-1)

    alpha_prime = (1-target_one_hot)+target_one_hot*alpha

    if exactInfDiv:

        if llRatioWeight == 1:
            distr_alpha = torch.distributions.dirichlet.Dirichlet(alpha)
            distr_alpha_prime = torch.distributions.dirichlet.Dirichlet(alpha_prime)

            infDivTerm = torch.distributions.kl.kl_divergence(distr_alpha, distr_alpha_prime)
        else:
            infDivTerm = logBeta(alpha_prime)-logBeta(alpha)+1/(llRatioWeight-1)*(logBeta(llRatioWeight*alpha+(1-llRatioWeight)*alpha_prime)-logBeta(alpha))
    else:
        raise NotImplementedError
        #infDivTerm = (llRatioWeight/2)*((target_one_hot*torch.pow(alpha-1,2)*polyg(alpha)).sum(dim=-1)-(target_one_hot*torch.pow(alpha-1,2)).sum(dim=-1)*polyg(alpha.sum(dim=-1)))

    ############ Maximum adversarial entropy term #############

    pgd_attack = FGSM(model)
    data_adv = pgd_attack(data, target.view(data.size(0),data.size(1))).to(data.device)
    alpha_adv = model(data_adv)

    alpha_adv_0 = alpha_adv.sum(dim=-1)
    maxAdvEntrTerm = logBeta(alpha_adv)+(alpha_adv_0-classNb)*torch.digamma(alpha_adv_0)-((alpha_adv-1)*torch.digamma(alpha_adv)).sum(dim=-1)
    maxAdvEntrTerm = torch.clamp(maxAdvEntrTerm,-300,300)

    loss += infDivWeight*infDivTerm.mean()-maxAdvEntrWeight*maxAdvEntrTerm.mean()

    return loss

def computeTransMat(dataset,transMat,priors,propStart,propEnd):

    videoPaths = load_data.findVideos(dataset,propStart,propEnd)

    for videoPath in videoPaths:
        videoName = os.path.splitext(os.path.basename(videoPath))[0]
        target = load_data.getGT(videoName,dataset)
        #Updating the transition matrix
        for i in range(len(target)-1):
            transMat[target[i],target[i+1]] += 1
            priors[target[i]] += 1

        #Taking the last target of the sequence into account only for prior
        priors[target[-1]] += 1

    #Just in case where propStart==propEnd, which is true for example, when the training set is empty
    if len(videoPaths) > 0:
        return transMat/transMat.sum(dim=1,keepdim=True),priors/priors.sum()
    else:
        return transMat,priors

def epochSeqVal(model,log_interval,loader, epoch, args,writer,metricEarlyStop,mode="val"):
    '''
    Validate a model. This function computes several metrics and return the best value found until this point.

    Args:
     - model (torch.nn.Module): the model to validate
     - log_interval (int): the number of epochs to wait before printing a log
     - loader (load_data.TrainLoader): the train data loader
     - epoch (int): the current epoch
     - args (Namespace): the namespace containing all the arguments required for training and building the network
     - writer (tensorboardX.SummaryWriter): the writer to use to log metrics evolution to tensorboardX
     - width (int): the width of the triangular window (i.e. the number of steps over which the window is spreading)
     - metricEarlyStop (str): the name of the metric to use for early stopping. Can be any of the metrics computed in the metricDict variable of the writeSummaries function
     - metricLastVal (float): the best value of the metric to use early stopping until now
     - maximiseMetric (bool): If true, the model maximising this metric will be kept.
    '''

    if args.debug:
        start_time = time.time()

    model.eval()

    print("Epoch",epoch," : ",mode)

    metrDict = metrics.emptyMetrDict(args.uncertainty)

    nbVideos = 0

    outDict,targDict = {},{}

    frameIndDict = {}

    #The writer dict for the attention maps. The will be one writer per class
    fullAttMapSeq,fullAffTransSeq = None,None
    revLabelDict = formatData.getReversedLabels()
    precVidName = "None"
    videoBegining = True
    validBatch = 0
    nbVideos = 0

    for batch_idx, (data,target,vidName,frameInds,timeElapsedTensor) in enumerate(loader):

        newVideo = (vidName != precVidName) or videoBegining

        if (batch_idx % log_interval == 0):
            print("\t",loader.sumL+1,"/",loader.nbImages)

        if args.cuda:
            data, target,frameInds,timeElapsedTensor = data.cuda(), target.cuda(),frameInds.cuda(),timeElapsedTensor.cuda()

        visualDict = model.computeVisual(data)
        feat = visualDict["x"].data

        fullAttMapSeq = catAttMap(visualDict,fullAttMapSeq)
        fullAffTransSeq = catAffineTransf(visualDict,fullAffTransSeq)

        update.updateFrameDict(frameIndDict,frameInds,vidName)

        if newVideo and not videoBegining:
            allOutput,nbVideos = update.updateMetrics(args,model,allFeat,allTimeElapsedTensor,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)
            fullAttMapSeq = saveAttMap(fullAttMapSeq,args.exp_id,args.model_id,epoch,precVidName)
            fullAffTransSeq = saveAffineTransf(fullAffTransSeq,args.exp_id,args.model_id,epoch,precVidName)

        if nbVideos<=5 and args.debug:
            if args.cuda:
                updateOccupiedGPURamCSV(epoch,mode,args.exp_id,args.model_id)
            updateOccupiedRamCSV(epoch,mode,args.exp_id,args.model_id)
            updateOccupiedCPUCSV(epoch,mode,args.exp_id,args.model_id)
        if newVideo:
            allTarget = target
            allFeat = feat.unsqueeze(0)
            allTimeElapsedTensor = timeElapsedTensor

            videoBegining = False
        else:
            allTarget = torch.cat((allTarget,target),dim=1)
            allFeat = torch.cat((allFeat,feat.unsqueeze(0)),dim=1)

            if torch.is_tensor(allTimeElapsedTensor):
                allTimeElapsedTensor = torch.cat((allTimeElapsedTensor,timeElapsedTensor),dim=1)

        precVidName = vidName

        if nbVideos > 1 and args.debug:
            break

    if not args.debug:
        allOutput,nbVideos = update.updateMetrics(args,model,allFeat,allTimeElapsedTensor,allTarget,precVidName,nbVideos,metrDict,outDict,targDict)
        fullAttMapSeq = saveAttMap(fullAttMapSeq,args.exp_id,args.model_id,epoch,precVidName)

    for key in outDict.keys():
        fullArr = torch.cat((frameIndDict[key].float(),outDict[key].squeeze(0).squeeze(1)),dim=1)
        np.savetxt("../results/{}/{}_epoch{}_{}.csv".format(args.exp_id,args.model_id,epoch,key),fullArr.cpu().detach().numpy())

    writeSummaries(metrDict,validBatch,writer,epoch,mode,args.model_id,args.exp_id,nbVideos=nbVideos)

    if args.debug:
        totalTime = time.time() - start_time
        updateTimeCSV(epoch,mode,args.exp_id,args.model_id,totalTime)

    return outDict,targDict,metrDict[metricEarlyStop]

def catAffineTransf(visualDict,fullAffTransSeq):

    if "theta" in visualDict.keys():
        if fullAffTransSeq is None:
            fullAffTransSeq = visualDict["theta"].cpu()
        else:
            fullAffTransSeq = torch.cat((fullAffTransSeq,visualDict["theta"].cpu()),dim=0)

    return fullAffTransSeq

def saveAffineTransf(fullAffTransSeq,exp_id,model_id,epoch,precVidName):
    if not fullAffTransSeq is None:
        np.save("../results/{}/affTransf_{}_epoch{}_{}.npy".format(exp_id,model_id,epoch,precVidName),fullAffTransSeq.numpy())
        fullAffTransSeq = None
    return fullAffTransSeq

def catAttMap(visualDict,fullAttMapSeq):
    if "attention" in visualDict.keys():
        if fullAttMapSeq is None:
            fullAttMapSeq = (visualDict["attention"].cpu()*255).byte()
        else:
            fullAttMapSeq = torch.cat((fullAttMapSeq,(visualDict["attention"].cpu()*255).byte()),dim=0)

    return fullAttMapSeq

def saveAttMap(fullAttMapSeq,exp_id,model_id,epoch,precVidName):
    if not fullAttMapSeq is None:
        np.save("../results/{}/attMaps_{}_epoch{}_{}.npy".format(exp_id,model_id,epoch,precVidName),fullAttMapSeq.numpy())
        fullAttMapSeq = None
    return fullAttMapSeq

def writeSummaries(metrDict,batchNb,writer,epoch,mode,model_id,exp_id,nbVideos=None):
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

    sampleNb = batchNb if mode == "train" else nbVideos

    for metric in metrDict.keys():

        if metric.find("Entropy") == -1:
            metrDict[metric] /= sampleNb
        else:
            metrDict[metric] = torch.median(metrDict[metric])

    for metric in metrDict:
        writer.add_scalars(metric,{model_id+"_"+mode:metrDict[metric]},epoch)

    if not os.path.exists("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode)):
        header = [metric.lower().replace(" ","_") for metric in metrDict.keys()]
    else:
        header = ""

    with open("../results/{}/model{}_epoch{}_metrics_{}.csv".format(exp_id,model_id,epoch,mode),"a") as text_file:
        print(header,file=text_file)
        print(",".join([str(metrDict[metric]) for metric in metrDict.keys()]),file=text_file)

    return metrDict

def get_OptimConstructor_And_Kwargs(optimStr,momentum):
    '''Return the apropriate constructor and keyword dictionnary for the choosen optimiser
    Args:
        optimStr (str): the name of the optimiser. Can be \'AMSGrad\', \'SGD\' or \'Adam\'.
        momentum (float): the momentum coefficient. Will be ignored if the choosen optimiser does require momentum
    Returns:
        the constructor of the choosen optimiser and the apropriate keyword dictionnary
    '''

    if optimStr != "AMSGrad":
        optimConst = getattr(torch.optim,optimStr)
        if optimStr == "SGD":
            kwargs= {'momentum': momentum}
        elif optimStr == "Adam":
            kwargs = {}
        else:
            raise ValueError("Unknown optimisation algorithm : {}".format(args.optim))
    else:
        optimConst = torch.optim.Adam
        kwargs = {'amsgrad':True}

    return optimConst,kwargs

def initialize_Net_And_EpochNumber(net,exp_id,model_id,cuda,start_mode,init_path,strict):
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

    if start_mode == "scratch":
        #Saving initial parameters
        torch.save(net.state_dict(), "../models/{}/{}_epoch0".format(exp_id,model_id))
        startEpoch = 1
    elif start_mode == "fine_tune":

        params = torch.load(init_path)

        #Checking if the key of the model start with "module."
        startsWithModule = (list(net.state_dict().keys())[0].find("module.") != -1)

        if startsWithModule:
            paramsFormated = {}
            for key in params.keys():
                keyFormat =  "module."+key if key.find("module") == -1 else key
                paramsFormated[keyFormat] = params[key]
            params = paramsFormated

        else:
            paramsFormated = {}
            for key in params.keys():
                keyFormat =  key.replace("module.","")
                paramsFormated[keyFormat] = params[key]
            params = paramsFormated

        #Removing keys corresponding to parameter which shape are different in the checkpoint and in the current model
        #For example, this is necessary to load a model trained on n classes to bootstrap a model with m != n classes.
        keysToRemove = []
        for key in params.keys():
            if key in net.state_dict().keys():
                if net.state_dict()[key].size() != params[key].size():
                    keysToRemove.append(key)
        for key in keysToRemove:
            params.pop(key)

        if hasattr(net,"tempModel"):
            if not hasattr(net.tempModel,"linLay"):
                def checkAndReplace(key):
                    if key.find("tempModel.linLay") != -1:
                        key = key.replace("tempModel.linLay","tempModel.linTempMod.linLay")
                    return key
                params = {checkAndReplace(k):params[k] for k in params.keys()}

        res = net.load_state_dict(params,strict)

        #Depending on the pytorch version the load_state_dict() method can return the list of missing and unexpected parameters keys or nothing
        if not res is None:
            missingKeys,unexpectedKeys = res
            print("missing keys",missingKeys)
            print("unexpected keys",unexpectedKeys)

        #Start epoch is 1 if strict if false because strict=False means that it is another model which is being trained
        if strict:
            startEpoch = utils.findLastNumbers(init_path)
        else:
            startEpoch = 1

    return startEpoch

def evalAllImages(exp_id,model_id,model,testLoader,cuda,log_interval):
    '''
    Pass all the images and/or the sound extracts of a loader in a feature model and save the feature vector in one csv for each image.
    Args:
    - exp_id (str): The experience id
    - model (nn.Module): the model to process the images
    - testLoader (load_data.TestLoader): the image and/or sound loader
    - cuda (bool): True is the computation has to be done on cuda
    - log_interval (int): the number of batches to wait before logging progression
    '''

    for batch_idx, (data, _,vidName,frameInds) in enumerate(testLoader):

        if (batch_idx % log_interval == 0):
            print("\t",testLoader.sumL+1,"/",testLoader.nbShots)

        if not data is None:
            if cuda:
                data = data.cuda()
            data = data[:,:len(frameInds)]
            data = data.view(data.size(0)*data.size(1),data.size(2),data.size(3),data.size(4))

        if not os.path.exists("../results/{}/{}".format(exp_id,vidName)):
            os.makedirs("../results/{}/{}".format(exp_id,vidName))

        feats = model(data)
        for i,feat in enumerate(feats):
            imageName = frameInds[i]
            if not os.path.exists("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id)):

                np.savetxt("../results/{}/{}/{}_{}.csv".format(exp_id,vidName,imageName,model_id),feat.detach().cpu().numpy())

def addInitArgs(argreader):
    argreader.parser.add_argument('--start_mode', type=str,metavar='SM',
                help='The mode to use to initialise the model. Can be \'scratch\' or \'fine_tune\'.')
    argreader.parser.add_argument('--init_path', type=str,metavar='SM',
                help='The path to the weight file to use to initialise the network')
    argreader.parser.add_argument('--strict_init', type=str2bool,metavar='SM',
                help='Set to True to make torch.load_state_dict throw an error when not all keys match (to use with --init_path)')

    return argreader
def addOptimArgs(argreader):
    argreader.parser.add_argument('--lr', type=args.str2FloatList,metavar='LR',
                        help='learning rate (it can be a schedule : --lr 0.01,0.001,0.0001)')
    argreader.parser.add_argument('--momentum', type=float, metavar='M',
                        help='SGD momentum')
    argreader.parser.add_argument('--optim', type=str, metavar='OPTIM',
                        help='the optimizer to use (default: \'SGD\')')
    return argreader
def addValArgs(argreader):
    argreader.parser.add_argument('--train_step_to_ignore', type=int,metavar='LMAX',
                    help='Number of steps that will be ignored at the begining and at the end of the training sequence for binary cross entropy computation')

    argreader.parser.add_argument('--val_l_temp', type=int,metavar='LMAX',help='Length of sequences for computation of scores when using a CNN temp model.')

    argreader.parser.add_argument('--metric_early_stop', type=str,metavar='METR',
                    help='The metric to use to choose the best model')
    argreader.parser.add_argument('--maximise_val_metric', type=args.str2bool,metavar='BOOL',
                    help='If true, The chosen metric for chosing the best model will be maximised')
    argreader.parser.add_argument('--max_worse_epoch_nb', type=int,metavar='NB',
                    help='The number of epochs to wait if the validation performance does not improve.')
    argreader.parser.add_argument('--run_test', type=args.str2bool,metavar='NB',
                    help='Evaluate the model on the test set')

    argreader.parser.add_argument('--compute_metrics_during_eval', type=args.str2bool,metavar='BOOL',
                    help='If false, the metrics will not be computed during validation, but the scores produced by the models will still be saved')

    return argreader
def addLossTermArgs(argreader):

    argreader.parser.add_argument('--uncer_max_adv_entr_weight', type=float,metavar='FLOAT',
                    help='The weight of the maximum divergence entropy term (only used when uncertainty is True)')

    argreader.parser.add_argument('--uncert_inf_div_weight', type=float,metavar='FLOAT',
                    help='The weight of the information divergence term (only used when uncertainty is True)')

    argreader.parser.add_argument('--uncer_loss_type', type=str,metavar='FLOAT',
                    help='The loss to use for the computation of uncertainty loss. Can be "MSE" or "CE".')

    argreader.parser.add_argument('--uncer_exact_inf_div', type=args.str2bool,metavar='FLOAT',
                    help='Set to True for exact computation of the information divergence term of the uncertainty loss.')

    argreader.parser.add_argument('--uncer_ll_ratio_weight', type=float,metavar='FLOAT',
                    help='The ratio between the likelihood in the information divergence term of the uncertainty loss. It should be between 0 (excluded) and 1 (included).')

    return argreader

def init_process(args,rank,size,fn,backend='gloo'):
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

    paramToOpti = []

    trainLoader,_ = load_data.buildSeqTrainLoader(args)

    valLoader = load_data.TestLoader(args.dataset_val,args.val_l,args.val_part_beg,args.val_part_end,args.prop_set_int_fmt,\
                                        args.img_size,args.orig_img_size,args.resize_image,\
                                        args.exp_id,args.mask_time_on_image,args.min_phase_nb)

    #Building the net
    net = modelBuilder.netBuilder(args)

    if args.cuda:
        net = net.cuda()

    trainFunc = epochSeqTr
    valFunc = epochSeqVal

    kwargsTr = {'log_interval':args.log_interval,'loader':trainLoader,'args':args,'writer':writer}
    kwargsVal = kwargsTr.copy()

    kwargsVal['loader'] = valLoader
    kwargsVal["metricEarlyStop"] = args.metric_early_stop

    for p in net.parameters():
        paramToOpti.append(p)

    paramToOpti = (p for p in paramToOpti)

    #Getting the contructor and the kwargs for the choosen optimizer
    optimConst,kwargsOpti = get_OptimConstructor_And_Kwargs(args.optim,args.momentum)

    startEpoch = initialize_Net_And_EpochNumber(net,args.exp_id,args.model_id,args.cuda,args.start_mode,args.init_path,args.strict_init)

    #If no learning rate is schedule is indicated (i.e. there's only one learning rate),
    #the args.lr argument will be a float and not a float list.
    #Converting it to a list with one element makes the rest of processing easier
    if type(args.lr) is float:
        args.lr = [args.lr]

    lrCounter = 0

    transMat,priors = computeTransMat(args.dataset_train,net.transMat,net.priors,args.train_part_beg,args.train_part_end)
    net.setTransMat(transMat)
    net.setPriors(priors)

    epoch = startEpoch

    if args.start_mode == "scratch":
        worseEpochNb = 0
        bestEpoch = epoch
    else:
        bestModelPaths = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id,args.model_id))
        if len(bestModelPaths) == 0:
            worseEpochNb = 0
            bestEpoch = epoch
        elif len(bestModelPaths) == 1:
            bestModelPath = bestModelPaths[0]
            bestEpoch = int(os.path.basename(bestModelPath).split("epoch")[1])
            worseEpochNb = startEpoch - bestEpoch
        else:
            raise ValueError("Wrong number of best model weight file : ",len(bestModelPaths))

    if args.maximise_val_metric:
        bestMetricVal = -np.inf
        isBetter = lambda x,y:x>y
    else:
        bestMetricVal = np.inf
        isBetter = lambda x,y:x<y


    while epoch < args.epochs + 1 and worseEpochNb < args.max_worse_epoch_nb:

        kwargsOpti,kwargsTr,lrCounter = update.updateLR(epoch,args.epochs,args.lr,startEpoch,kwargsOpti,kwargsTr,lrCounter,net,optimConst)

        kwargsTr["epoch"],kwargsVal["epoch"] = epoch,epoch
        kwargsTr["model"],kwargsVal["model"] = net,net

        if not args.no_train:
            trainFunc(**kwargsTr)
        else:
            net.load_state_dict(torch.load("../models/{}/model{}_epoch{}".format(args.no_train[0],args.no_train[1],epoch)))

        with torch.no_grad():
            _,_,metricVal = valFunc(**kwargsVal)

        if isBetter(metricVal,bestMetricVal):
            if os.path.exists("../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id,bestEpoch)):
                os.remove("../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id,bestEpoch))

            torch.save(net.state_dict(), "../models/{}/model{}_best_epoch{}".format(args.exp_id,args.model_id, epoch))
            bestEpoch = epoch
            bestMetricVal = metricVal
            worseEpochNb = 0
        else:
            worseEpochNb += 1

        epoch += 1

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--comp_feat',type=str2bool,help='To compute and write in a file the features of all images in the test set. All the arguments used to \
                                    build the model and the test data loader should be set.')
    argreader.parser.add_argument('--no_train',type=str2bool,help='To use to re-evaluate a model at each epoch after training. At each epoch, the model is not trained but \
                                                                            the weights of the corresponding epoch are loaded and then the model is evaluated.\
                                                                            The arguments --exp_id_no_train and the --model_id_no_train must be set')

    argreader.parser.add_argument('--exp_id_no_train',type=str,help="To use when --no_train is set to True. This is the exp_id of the model to get the weights from.")
    argreader.parser.add_argument('--model_id_no_train',type=str,help="To use when --no_train is set to True. This is the model_id of the model to get the weights from.")

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.redirect_out:
        sys.stdout = open("python.out", 'w')

    #The folders where the experience file will be written
    if not (os.path.exists("../vis/{}".format(args.exp_id))):
        os.makedirs("../vis/{}".format(args.exp_id))
    if not (os.path.exists("../results/{}".format(args.exp_id))):
        os.makedirs("../results/{}".format(args.exp_id))
    if not (os.path.exists("../models/{}".format(args.exp_id))):
        os.makedirs("../models/{}".format(args.exp_id))

    #Write the arguments in a config file so the experiment can be re-run
    argreader.writeConfigFile("../models/{}/{}.ini".format(args.exp_id,args.model_id))

    print("Model :",args.model_id,"Experience :",args.exp_id)

    if args.comp_feat:

        testLoader = load_data.TestLoader(args.val_l,args.dataset_test,args.test_part_beg,args.test_part_end,args.prop_set_int_fmt,args.img_size,args.orig_img_size,\
                                          args.resize_image,args.exp_id,args.mask_time_on_image,args.min_phase_nb)

        if args.feat != "None":
            featModel = modelBuilder.buildFeatModel(args.feat,args.pretrain_dataset,args.lay_feat_cut)
            if args.cuda:
                featModel = featModel.cuda()
            if args.init_path_visual != "None":
                featModel.load_state_dict(torch.load(args.init_path_visual))
            elif args.init_path != "None":
                model = modelBuilder.netBuilder(args)
                params = torch.load(args.init_path)
                state_dict = {k.replace("module.cnn.","cnn.module."): v for k,v in params.items()}
                model.load_state_dict(state_dict)
                featModel = model.featModel

            featModel.eval()
        else:
            featModel = None

        with torch.no_grad():
            evalAllImages(args.exp_id,args.model_id,featModel,testLoader,args.cuda,args.log_interval)

    else:

        if args.distributed:
            size = args.distrib_size
            processes = []
            for rank in range(size):
                p = Process(target=init_process, args=(args,rank,size,run))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
        else:
            run(args)

if __name__ == "__main__":
    main()
