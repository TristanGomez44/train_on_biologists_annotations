
from args import ArgReader

import os
import glob

import torch
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import matplotlib.cm as cm
import matplotlib.patches as patches
import pims
import cv2
from PIL import Image

import load_data
import modelBuilder

import metrics
import utils
import formatData
import trainVal
import formatData

import sys

import configparser

import matplotlib.patheffects as path_effects

def evalModel(dataset,partBeg,partEnd,exp_id,model_id,epoch,regression,uncertainty,nbClass):
    '''
    Evaluate a model. It requires the scores for each video to have been computed already with the trainVal.py script. Check readme to
    see how to compute the scores for each video.

    It computes the performance of a model using the default decision threshold (0.5) and the best decision threshold.

    The best threshold is computed for each video by looking for the best threshold on all the other videos. The best threshold is also
    computed for each metric.

    To find the best threshold, a range of threshold are evaluated and the best is selected.

    Args:
    - exp_id (str): the name of the experience
    - model_id (str): the id of the model to evaluate. Eg : "res50_res50_youtLarg"
    - epoch (int): the epoch at which to evaluate

    '''

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch)),key=utils.findNumbers))
    videoNameDict = buildVideoNameDict(dataset,partBeg,partEnd,resFilePaths)
    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))

    #Store the value of the f-score of for video and for each threshold
    metTun = {}

    metricNameList = metrics.emptyMetrDict().keys()
    metEval={}
    for metricName in metricNameList:
        if metricName.find("Accuracy") != -1:
            metEval[metricName] = np.zeros(len(resFilePaths))

    transMat,priors = torch.zeros((nbClass,nbClass)).float(),torch.zeros((nbClass)).float()
    transMat,_ = trainVal.computeTransMat(dataset,transMat,priors,partBeg,partEnd)

    totalFrameNb = 0

    for j,path in enumerate(resFilePaths):

        fileName = os.path.basename(os.path.splitext(path)[0])
        videoName = videoNameDict[path]

        #Compute the metrics with the default threshold (0.5) and with a threshold tuned on each video with a leave-one out method
        metEval["Accuracy"][j],frameNb = computeMetrics(path,dataset,videoName,resFilePaths,videoNameDict,metTun,"Accuracy",transMat,regression,uncertainty)
        metEval["Accuracy (Viterbi)"][j],_ = computeMetrics(path,dataset,videoName,resFilePaths,videoNameDict,metTun,"Accuracy (Viterbi)",transMat,regression,uncertainty)

        metEval["Accuracy"][j] *= frameNb
        metEval["Accuracy (Viterbi)"][j] *= frameNb

        totalFrameNb += frameNb

    #Writing the latex table
    printHeader = not os.path.exists("../results/{}/metrics.csv".format(exp_id))
    with open("../results/{}/metrics.csv".format(exp_id),"a") as text_file:
        if printHeader:
            print("Model,Accuracy,Accuracy (Viterbi)",file=text_file)

        print(model_id+","+str(metEval["Accuracy"].sum()/totalFrameNb)+","+str(metEval["Accuracy (Viterbi)"].sum()/totalFrameNb),file=text_file)

def computeMetrics(path,dataset,videoName,resFilePaths,videoNameDict,metTun,metric,transMat,regression,uncertainty):
    '''
    Evaluate a model on a video by using the default threshold and a threshold tuned on all the other video

    Args:
    - path (str): the path to the video to evaluate
    - videoName (str): the name of the video
    - resFilePaths (list): the paths to the scores given by the model for each video of the dataset
    - videoNameDict (dict): a dictionnary mapping the score file paths to the video names
    - metTun (dict): a dictionnary containing the performance of the model for each threshold and each video. It allows to not repeat computation. \
                    this dictionnary is updated during the execution of this function.
    - metric (str): the metric to evaluate.
    Returns:
    - metr_dict[metric] (float): the value of the metric using the tuned threshold
    - def_metr_dict[metric]: the metric using default threshold

    '''

    gt = load_data.getGT(videoName,dataset).astype(int)
    scores = np.genfromtxt(path,delimiter=" ")[:,1:]

    metr_dict = metrics.binaryToMetrics(torch.tensor(scores[np.newaxis,:]).float(),torch.tensor(gt[np.newaxis,:]),transMat,regression,uncertainty)

    return metr_dict[metric],len(scores)

def formatMetr(mean,std):
    return "$"+str(round(mean,2))+" \pm "+str(round(std,2))+"$"

def plotScore(dataset,exp_id,model_id,epoch,trainPartBeg,trainPartEnd,scoreAxNb=4):
    ''' This function plots the scores given by a model to seral videos.

    It also plots the distance between shot features and it also produces features showing the correlation between
    scene change and the score value, or its second derivative.

    Args:
    - exp_id (str): the experiment id
    - model_id (str): the model id
    - plotDist (bool): set to True to plot the distance between features.
    - epoch (int): the epoch at which the model is evaluated.

    '''

    #This dictionnary returns a label using its index
    revLabelDict = formatData.getReversedLabels()
    labDict = formatData.getLabels()
    reverLabDict = formatData.getReversedLabels()
    cmap = cm.hsv(np.linspace(0, 1, len(revLabelDict.keys())))

    resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}*.csv".format(exp_id,model_id,epoch)))
    videoPaths = load_data.findVideos(dataset,propStart=0,propEnd=1)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    for path in resFilePaths:

        videoName = None
        for candidateVideoName in videoNames:
            if "_"+candidateVideoName.replace("__","_")+".csv" in path:
                videoName = candidateVideoName

        if not videoName is None:
            fileName = os.path.basename(os.path.splitext(path)[0])

            scores = np.genfromtxt(path,delimiter=" ")
            nbFrames = scores[-1,0]
            scores = scores[:,1:]

            f, axList = plt.subplots(scoreAxNb+3, 1,figsize=(30,5))
            ax1 = axList[0]

            #ax1.set_xlim(0,nbFrames)
            ax1.set_xlabel("GT")

            legHandles = []

            #Plot the ground truth phases
            gt = np.genfromtxt("../data/"+dataset+"/annotations/"+videoName+"_phases.csv",dtype=str,delimiter=",")
            legHandles = plotPhases(gt,legHandles,labDict,cmap,axList[0],nbFrames,"GT")

            #Plot the scores
            expVal = np.exp(scores)
            scores = expVal/expVal.sum(axis=-1,keepdims=True)

            axList[1].set_ylabel("Scores",rotation="horizontal",fontsize=20,horizontalalignment="right",position=(0,-2.5))
            for i in range(scores.shape[1]):
                ax = axList[i%((scores.shape[1]+1)//scoreAxNb)+1]
                ax.set_xlim(0,nbFrames)
                fill = ax.fill_between(np.arange(len(scores[:,i])), 0, scores[:,i],label=revLabelDict[i],color=cmap[i])

            #Plot the prediction only considering the scores and not the state transition matrix
            predSeq = scores.argmax(axis=-1)
            predSeq = labelIndList2FrameInd(predSeq,reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[-2],nbFrames,"Prediction")

            #Plot the prediction with viterbi decoding
            transMat = torch.zeros((scores.shape[1],scores.shape[1]))
            priors = torch.zeros((scores.shape[1],))
            transMat,_ = trainVal.computeTransMat(dataset,transMat,priors,trainPartBeg,trainPartEnd)
            predSeqs,_ = metrics.viterbi_decode(torch.log(torch.tensor(scores).float()),torch.log(transMat),top_k=1)
            predSeq = labelIndList2FrameInd(predSeqs[0],reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[-1],nbFrames,"Prediction (Viterbi)")

            plt.xlabel("Time (frame index)")
            ax1.legend(bbox_to_anchor=(1.1, 1.05),prop={'size': 15})
            plt.subplots_adjust(hspace=0.6)
            plt.savefig("../vis/{}/{}_epoch{}_video{}_scores.png".format(exp_id,model_id,epoch,fileName))
            plt.close()
            sys.exit(0)
        else:
            raise ValueError("Unkown video : ",path)

def plotPhases(phases,legHandles,labDict,cmap,ax,nbFrames,ylab):
    for i,phase in enumerate(phases):
        rect = patches.Rectangle((int(phase[1]),0),int(phase[2])-int(phase[1]),1,linewidth=1,color=cmap[labDict[phase[0]]],alpha=1,label=phase[0])
        legHandles += [ax.add_patch(rect)]
    ax.set_xlim(0,nbFrames)
    ax.set_ylabel(ylab,rotation="horizontal",fontsize=20,horizontalalignment="right",position=(0,0.1))

    return legHandles

def labelIndList2FrameInd(labelList,reverLabDict):

    currLabel = labelList[0]
    phases = []
    currStartFrame = 0
    for i in range(len(labelList)):

        if labelList[i] != currLabel:
            phases.append((reverLabDict[currLabel],currStartFrame,i-1))
            currStartFrame = i
            currLabel = labelList[i]

    phases.append((reverLabDict[currLabel],currStartFrame,i))
    return phases

def buildVideoNameDict(dataset,test_part_beg,test_part_end,resFilePaths):

    ''' Build a dictionnary associating a path to a video name (it can be the path to any file than contain the name of a video in its file name) '''

    videoPaths = load_data.findVideos(dataset,test_part_beg,test_part_end)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))
    videoNameDict = {}

    for path in resFilePaths:
        for videoName in videoNames:
            if "_"+videoName.replace("__","_")+".csv" in path.replace("__","_"):
                videoNameDict[path] = videoName

    #if len(videoNameDict.keys()) < len(videoNames):
    #    raise ValueError("Some result file could not get their video identified. Files identified :",videoNameDict.keys())

    return videoNameDict

def plotData(nbClass,dataset):

    transMat = torch.zeros((nbClass,nbClass))
    priors = torch.zeros((nbClass,))
    transMat,priors = trainVal.computeTransMat(dataset,transMat,priors,0,1)

    labels = list(formatData.getLabels().keys())[:nbClass]

    plt.figure()
    image = plt.imshow(torch.sqrt(transMat), cmap='hot', interpolation='nearest')
    plt.xticks(np.arange(len(labels)),labels,rotation=45)
    plt.yticks(np.arange(len(labels)),labels)
    plt.xlabel("Following phase")
    plt.ylabel("Current phase")
    ticks = np.arange(10)/10
    cb = plt.colorbar(image,ticks=ticks)
    cb.ax.set_yticklabels([round(i*i,2) for i in ticks])
    plt.tight_layout()
    plt.savefig("../vis/transMat.png")

    videoPaths = load_data.findVideos(dataset,0,1)
    nbImages=0
    for videoPath in videoPaths:
        nbImages += len(load_data.getGT(os.path.splitext(os.path.basename(videoPath))[0],dataset))

    plt.figure()
    plt.bar(np.arange(nbClass),priors*nbImages)
    plt.xticks(np.arange(nbClass),labels,rotation=45)
    plt.xlabel("Developpement phases")
    plt.ylabel("Number of image")
    plt.tight_layout()
    plt.savefig("../vis/prior.png")

def agregatePerfs(exp_id,paramAgr,keysRef,namesRef):

    csv = np.genfromtxt("../results/{}/metrics.csv".format(exp_id),delimiter=",",dtype="str")

    keyToNameDict = {key:name for key,name in zip(keysRef,namesRef)}
    nameToKeyDict = {name:key for key,name in zip(keysRef,namesRef)}

    groupedLines = {}
    metricNames = csv[0,1:]
    for line in csv[1:]:

        key = readConfFile("../models/{}/{}.ini".format(exp_id,line[0]),paramAgr)

        if key in groupedLines.keys():
            groupedLines[key].append(line)
        else:
            groupedLines[key] = [line]

    csvStr = "Model&"+"&".join(metricNames)+"\\\\ \n \hline \n"
    mean = np.zeros((len(groupedLines.keys()),csv.shape[1]-1))
    std =  np.zeros((len(groupedLines.keys()),csv.shape[1]-1))

    keys = groupedLines.keys()

    #Reordering the keys
    orderedKeys = []
    for name in nameToKeyDict.keys():
        orderedKeys.append(nameToKeyDict[name])
    keys = orderedKeys

    for i,key in enumerate(keys):
        groupedLines[key] = np.array(groupedLines[key])[:,1:].astype(float)

        mean[i] =  groupedLines[key].mean(axis=0)
        std[i] = groupedLines[key].std(axis=0)

        csvStr += keyToNameDict[key]
        for j in range(len(mean[0])):
            csvStr += "&"+formatMetr(mean[i,j],std[i,j])

        csvStr += "\\\\ \n"

    with open("../results/{}/metrics_agr.csv".format(exp_id),"w") as text_file:
        print(csvStr,file=text_file)


def readConfFile(path,keyList):
    ''' Read a config file and get the value of desired argument

    Args:
        path (str): the path to the config file
        keyList (list): the list of argument to read name)
    Returns:
        the argument value, in the same order as in keyList
    '''

    conf = configparser.ConfigParser()
    conf.read(path)
    conf = conf["default"]
    resList = []
    for key in keyList:
        resList.append(conf[key])

    return ",".join(resList)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--epoch_to_process',type=int,metavar="N",help='The epoch to process. This argument should be set using the --plot_score or the --eval_model arguments.')

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--plot_score',action="store_true",help='To plot the probabilities produced by a model for all the videos processed by this model during validation for some epoch.\
                                                                            The --model_id argument must be set, along with the --exp_id, --dataset_test, --epoch_to_process, --train_part_beg, --train_part_end (for \
                                                                            computing the state transition matrix.)')

    ########## COMPUTE METRICS AND PUT THEM IN AN LATEX TABLE #############
    argreader.parser.add_argument('--eval_model',action="store_true",help='Evaluate a model using the csv files containing the scores. The value of this arg is the epoch at which to evaluate the model \
                                                                            The --exp_id argument must be set, along with the --test_part_beg, --test_part_end and \
                                                                            --dataset_test, --regression, --class_nb and --epoch_to_process arguments. The arguments --param_agr, --keys, and --names \
                                                                            arguments can also be set.')

    argreader.parser.add_argument('--param_agr',type=str,nargs="*",metavar="PARAM",help='A list of meta-parameter to use to agregate the performance of several models.')
    argreader.parser.add_argument('--keys',type=str,nargs="*",metavar="KEY",help='The list of key that will appear during aggregation. In the final csv file, each key value will be replaced by a string of the list --names.\
                                                                                  to make it easier to read.')
    argreader.parser.add_argument('--names',type=str,nargs="*",metavar="NAME",help='The list of string to replace each key by during agregation.')

    ######################## Database plot #################################

    argreader.parser.add_argument('--plot_data',type=int,metavar="N",help='To plot the state transition matrix and the prior vector. The value is the number of classes. The --dataset_test must be set.')

    argreader = load_data.addArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_score:
        plotScore(args.dataset_test,args.exp_id,args.model_id,args.epoch_to_process,args.train_part_beg,args.train_part_end)
    if not args.eval_model is None:

        if os.path.exists("../results/{}/metrics.csv".format(args.exp_id)):
            os.remove("../results/{}/metrics.csv".format(args.exp_id))

        model_ids = list(map(lambda x:os.path.splitext(os.path.basename(x))[0],sorted(glob.glob("../models/{}/*.ini".format(args.exp_id)))))
        for model_id in model_ids:
            evalModel(args.dataset_test,args.test_part_beg,args.test_part_end,args.exp_id,model_id,epoch=args.epoch_to_process,\
                        regression=args.regression,uncertainty=args.uncertainty,nbClass=args.class_nb)
        if len(args.param_agr) > 0:
            agregatePerfs(args.exp_id,args.param_agr,args.keys,args.names)
    if not args.plot_data is None:
        plotData(args.plot_data,args.dataset_test)

if __name__ == "__main__":
    main()
