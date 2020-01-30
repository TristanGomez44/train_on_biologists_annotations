
from args import ArgReader
from args import str2bool
import os
import glob

import torch
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import sklearn
import matplotlib.cm as cm
import matplotlib.patches as patches
import pims
import cv2
from PIL import Image

import load_data


import metrics
import utils
import formatData
import trainVal
import formatData
import scipy
import sys

import configparser

import matplotlib.patheffects as path_effects
import imageio
from skimage import img_as_ubyte

from scipy import stats
import math
from PIL import Image
from PIL import Image, ImageEnhance
def evalModel(dataset,partBeg,partEnd,propSetIntFormat,exp_id,model_id,epoch,regression,uncertainty,nbClass):
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

    try:
        resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch)),key=utils.findNumbers))
        videoNameDict = buildVideoNameDict(dataset,partBeg,partEnd,propSetIntFormat,resFilePaths)
    except ValueError:
        testEpoch = len(glob.glob("../models/{}/model{}_epoch*".format(exp_id,model_id)))
        resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,testEpoch)),key=utils.findNumbers))
        videoNameDict = buildVideoNameDict(dataset,partBeg,partEnd,propSetIntFormat,resFilePaths)

    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))

    #Store the value of the f-score of for video and for each threshold
    metTun = {}

    metricNameList = metrics.emptyMetrDict().keys()
    metEval={}
    for metricName in metricNameList:
        if metricName.find("Accuracy") != -1:
            metEval[metricName] = np.zeros(len(resFilePaths))
        if metricName == "Correlation":
            metEval[metricName] = []


    transMat,priors = torch.zeros((nbClass,nbClass)).float(),torch.zeros((nbClass)).float()
    transMat,_ = trainVal.computeTransMat(dataset,transMat,priors,partBeg,partEnd,propSetIntFormat)

    totalFrameNb = 0

    for j,path in enumerate(resFilePaths):

        fileName = os.path.basename(os.path.splitext(path)[0])
        videoName = videoNameDict[path]

        #Compute the metrics with the default threshold (0.5) and with a threshold tuned on each video with a leave-one out method
        metrDict,frameNb = computeMetrics(path,dataset,videoName,resFilePaths,videoNameDict,metTun,transMat,regression,uncertainty)

        for metricName in metEval.keys():

            if metricName.find("Accuracy") != -1 and metricName.find("Temp") == -1:
                metEval[metricName][j] = metrDict[metricName]
                metEval[metricName][j] *= frameNb

            if metricName == "Correlation":
                metEval[metricName] += metrDict[metricName]

            if metricName == "Temp Accuracy":
                metEval[metricName][j] = metrDict[metricName]

        totalFrameNb += frameNb

    metEval["Correlation"] = np.array(metEval["Correlation"])
    metEval["Correlation"] = np.corrcoef(metEval["Correlation"][:,0],metEval["Correlation"][:,1])[0,1]

    #Writing the latex table
    printHeader = not os.path.exists("../results/{}/metrics.csv".format(exp_id))
    with open("../results/{}/metrics.csv".format(exp_id),"a") as text_file:
        if printHeader:
            print("Model,Accuracy,Accuracy (Viterbi),Correlation,Temp Accuracy",file=text_file)

        print(model_id+","+str(metEval["Accuracy"].sum()/totalFrameNb)+","+str(metEval["Accuracy (Viterbi)"].sum()/totalFrameNb)+","\
                           +str(metEval["Correlation"])+","+str(metEval["Temp Accuracy"].mean()),file=text_file)

def computeMetrics(path,dataset,videoName,resFilePaths,videoNameDict,metTun,transMat,regression,uncertainty):
    '''
    Evaluate a model on a video by using the default threshold and a threshold tuned on all the other video

    Args:
    - path (str): the path to the video to evaluate
    - videoName (str): the name of the video
    - resFilePaths (list): the paths to the scores given by the model for each video of the dataset
    - videoNameDict (dict): a dictionnary mapping the score file paths to the video names
    - metTun (dict): a dictionnary containing the performance of the model for each threshold and each video. It allows to not repeat computation. \
                    this dictionnary is updated during the execution of this function.
    Returns:
    - metr_dict (dict): the dict containing all metrics

    '''

    gt = load_data.getGT(videoName,dataset).astype(int)
    frameStart = (gt == -1).sum()
    gt = gt[frameStart:]

    scores = np.genfromtxt(path,delimiter=" ")[:,1:]

    #print(dataset,videoName,scores.shape,gt.shape)

    gt = gt[:len(scores)]

    metr_dict = metrics.binaryToMetrics(torch.tensor(scores[np.newaxis,:]).float(),torch.tensor(gt[np.newaxis,:]),transMat,regression,uncertainty,videoNames=[videoName])

    return metr_dict,len(scores)

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

            #Plot the scores
            expVal = np.exp(scores)
            scores = expVal/expVal.sum(axis=-1,keepdims=True)

            axList[1].set_ylabel("Scores",rotation="horizontal",fontsize=20,horizontalalignment="right",position=(0,-2.5))
            for i in range(scores.shape[1]):
                ax = axList[i%((scores.shape[1]+1)//scoreAxNb)]
                ax.set_xlim(0,nbFrames)
                fill = ax.fill_between(np.arange(len(scores[:,i])), 0, scores[:,i],label=revLabelDict[i],color=cmap[i])

            #Plot the prediction only considering the scores and not the state transition matrix
            predSeq = scores.argmax(axis=-1)
            predSeq = labelIndList2FrameInd(predSeq,reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[-3],nbFrames,"Prediction")

            #Plot the prediction with viterbi decoding
            transMat = torch.zeros((scores.shape[1],scores.shape[1]))
            priors = torch.zeros((scores.shape[1],))
            transMat,_ = trainVal.computeTransMat(dataset,transMat,priors,trainPartBeg,trainPartEnd)
            predSeqs,_ = metrics.viterbi_decode(torch.log(torch.tensor(scores).float()),torch.log(transMat),top_k=1)
            predSeq = labelIndList2FrameInd(predSeqs[0],reverLabDict)
            legHandles = plotPhases(predSeq,legHandles,labDict,cmap,axList[-2],nbFrames,"Prediction (Viterbi)")

            #Plot the ground truth phases
            gt = np.genfromtxt("../data/"+dataset+"/annotations/"+videoName+"_phases.csv",dtype=str,delimiter=",")
            legHandles = plotPhases(gt,legHandles,labDict,cmap,axList[-1],nbFrames,"GT")

            plt.xlabel("Time (frame index)")
            ax1.legend(bbox_to_anchor=(1.1, 1.05),prop={'size': 15})
            plt.subplots_adjust(hspace=0.6)
            plt.savefig("../vis/{}/{}_epoch{}_video{}_scores.png".format(exp_id,model_id,epoch,fileName))
            plt.close()

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

def buildVideoNameDict(dataset,test_part_beg,test_part_end,propSetIntFormat,resFilePaths,raiseError=True):

    ''' Build a dictionnary associating a path to a video name (it can be the path to any file than contain the name of a video in its file name) '''

    videoPaths = load_data.findVideos(dataset,test_part_beg,test_part_end,propSetIntFormat)
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))

    videoNameDict = {}
    for path in resFilePaths:
        for videoName in videoNames:
            if videoName in path:
                videoNameDict[path] = videoName

    if len(videoNameDict.keys()) < len(videoNames):
        if raiseError:
            raise ValueError("Could not find result files corresponding to some videos. Files identified :",sorted(list(videoNameDict.keys())))

    return videoNameDict

def plotData(nbClass,dataset):

    transMat = torch.zeros((nbClass,nbClass))
    priors = torch.zeros((nbClass,))
    transMat,priors = trainVal.computeTransMat(dataset,transMat,priors,0,1,False)

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
    plt.savefig("../vis/transMat_{}.png".format(dataset))

    videoPaths = load_data.findVideos(dataset,0,1)
    nbImages=0
    for videoPath in videoPaths:
        nbImages += len(load_data.getGT(os.path.splitext(os.path.basename(videoPath))[0],dataset))

    plt.figure()
    plt.bar(np.arange(nbClass),priors*nbImages)
    plt.xticks(np.arange(nbClass),labels,rotation=45)
    plt.xlabel("Developpement phases")
    plt.ylabel("Number of images")
    plt.title("Dataset : '{}'".format(dataset))
    plt.tight_layout()
    plt.savefig("../vis/prior_{}.png".format(dataset))

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

    for i,key in enumerate(orderedKeys):
        groupedLines[key] = np.array(groupedLines[key])[:,1:].astype(float)

        mean[i] =  groupedLines[key].mean(axis=0)
        std[i] = groupedLines[key].std(axis=0)

        csvStr += keyToNameDict[key]
        for j in range(len(mean[0])):
            csvStr += "&"+formatMetr(mean[i,j],std[i,j])

        csvStr += "\\\\ \n"

    with open("../results/{}/metrics_agr.csv".format(exp_id),"w") as text_file:
        print(csvStr,file=text_file)

    metricNames = csv[0,1:]

    #Ploting the performance
    plotRes(mean,std,csv,orderedKeys,exp_id,metricNames,keyToNameDict)

    #Computing the t-test
    ttest_matrix(groupedLines,orderedKeys,exp_id,metricNames,keyToNameDict)

def plotRes(mean,std,csv,keys,exp_id,metricNames,keyToNameDict):

    csv = csv[1:]

    #Plot the agregated results
    fig = plt.figure()
    plt.subplots_adjust(bottom=0.2)
    #plt.tight_layout()
    ax = fig.add_subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    for i in range(len(mean)):
        ax.bar(np.arange(csv.shape[1]-1)+0.1*i,mean[i],width=0.1,label=keyToNameDict[keys[i]],yerr=std[i])

    fig.legend(loc='right')
    plt.ylim(0,1)
    plt.ylabel("Performance")
    plt.gca().set_ylim(bottom=0)
    plt.xticks(np.arange(csv.shape[1]-1),metricNames,rotation=35,horizontalalignment="right")
    #plt.tight_layout()
    plt.savefig("../vis/{}/performance.png".format(exp_id))

def ttest_matrix(groupedLines,keys,exp_id,metricNames,keyToNameDict):
    ''' Computes the two sample t-test over groud of models

    Each combination of meta-parameters is put against every other by computing the p value of the two sample t-test.

    Args:
        groupedLines (dict): a dictionnary containing the error (or inclusion percentage) of several models having the same combination of varying parameters
        exp_id (str): the experience name
        metricNames (list): the names of the metrics

    '''

    mat = np.zeros((len(keys),len(keys),len(metricNames)))

    for i in range(len(keys)):
        for j in range(len(keys)):
            for k in range(len(metricNames)):

                _,mat[i,j,k] = scipy.stats.ttest_ind(groupedLines[keys[i]][:,k],groupedLines[keys[j]][:,k],equal_var=True)

    mat = mat.astype(str)
    for k in range(len(metricNames)):
        csv = "\t"+"\t".join([keyToNameDict[key] for key in keys])+"\n"
        for i in range(len(keys)):
            csv += keyToNameDict[keys[i]]+"\t"+"\t".join(mat[i,:,k])+"\n"

        with open("../results/{}/ttest_{}.csv".format(exp_id,metricNames[k]),"w") as text_file:
            print(csv,file=text_file)

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

def plotAttentionMaps(dataset,exp_id,model_id,plotFeatMaps):

    if plotFeatMaps:
        featMapPaths = sorted(glob.glob("../results/{}/featMaps_{}_epoch*_*.npy".format(exp_id,model_id)))
    else:
        featMapPaths = sorted(glob.glob("../results/{}/attMaps_{}_epoch*_*.npy".format(exp_id,model_id)))

    videoNameDict = buildVideoNameDict(dataset,0,100,True,featMapPaths,raiseError=False)

    conf = configparser.ConfigParser()
    conf.read("../models/{}/{}.ini".format(exp_id,model_id))
    conf = conf["default"]
    nbClass = int(conf["class_nb"])

    cmap = cm.hsv(np.linspace(0, 1, nbClass))

    endImages = None

    for vidInd,featMapPath in enumerate(featMapPaths):
        featMaps = np.load(featMapPath)
        videoName = videoNameDict[featMapPath]

        dataset_of_the_video = load_data.getDataset(videoName)
        video = pims.Video("../data/{}/{}.avi".format(dataset_of_the_video,videoName))
        epoch = int(os.path.splitext(featMapPath.split("epoch")[1].split("_"+videoName)[0])[0])

        predictions = np.genfromtxt("../results/{}/{}_epoch{}_{}.csv".format(exp_id,model_id,epoch,videoName))[:,1:].argmax(axis=1)
        gt = load_data.getGT(videoName,dataset_of_the_video).astype(int)
        frameStart = (gt == -1).sum()

        if plotFeatMaps:
            videoPath = '../vis/{}/featMaps_{}_{}.mp4'.format(exp_id,model_id,videoName)
        else:
            videoPath = '../vis/{}/attMaps_{}_{}.mp4'.format(exp_id,model_id,videoName)

        with imageio.get_writer(videoPath, mode='I') as writer:

            i=frameStart

            print(vidInd+1,"/",len(featMapPaths),videoName)

            if not endImages is None:
                featMaps = np.concatenate((endImages,featMaps),axis=0)

            if featMaps.shape[0] != predictions.shape[0]:
                endImages = featMaps[predictions.shape[0]:]
                featMaps = featMaps[:predictions.shape[0]]

            if featMaps.shape[0] != predictions.shape[0]:
                raise ValueError("featMaps and predictions are not of the same length : ",featMaps.shape,predictions.shape)

            frameNb = utils.getVideoFrameNb("../data/{}/{}.avi".format(dataset_of_the_video,videoName))

            while i < frameNb:

                frame = video[i]

                frame = frame[frame.shape[0]-frame.shape[1]:,:]

                nearestLowerDiv = frame.shape[0]//16
                nearestHigherDiv = (nearestLowerDiv+1)*16
                frame = resize(frame, (nearestHigherDiv,nearestHigherDiv),anti_aliasing=True,mode="constant",order=0)*255

                #Getting the attention map corresponding to the class predicted at this frame
                resizedAttFeatMap = resize(featMaps[i-frameStart,predictions[i-frameStart]], (frame.shape[0],frame.shape[1]),anti_aliasing=True,mode="constant",order=0)*255

                resizedAttFeatMap = resizedAttFeatMap[:,:,np.newaxis]*2/3+1/3

                color = cmap[predictions[i-frameStart]]
                color = color+(1-color)*0.75

                #frame = frame.astype("float")*resizedAttFeatMap*color[np.newaxis,np.newaxis,:-1]
                frame = resizedAttFeatMap*color[np.newaxis,np.newaxis,:-1]

                writer.append_data(img_as_ubyte(frame.astype("uint8")))
                i+=1

def plotMultiAttentionMaps(dataset,exp_id,model_id,epochToProcess):

    featMapPaths = sorted(glob.glob("../results/{}/attMaps_{}_epoch{}_*.npy".format(exp_id,model_id,epochToProcess)))

    videoNameDict = buildVideoNameDict(dataset,0,100,True,featMapPaths,raiseError=False)

    revVideoNameDict = {}

    cm = plt.get_cmap('plasma')

    for featMapPath in videoNameDict.keys():
        if not videoNameDict[featMapPath] in revVideoNameDict.keys():
            revVideoNameDict[videoNameDict[featMapPath]] = [featMapPath]
        else:
            revVideoNameDict[videoNameDict[featMapPath]].append(featMapPath)

    for vidInd,videoName in enumerate(revVideoNameDict.keys()):

        featMapsList = [np.load(featMapPath) for featMapPath in revVideoNameDict[videoName]]

        dataset_of_the_video = load_data.getDataset(videoName)
        video = pims.Video("../data/{}/{}.avi".format(dataset_of_the_video,videoName))

        epoch = epochToProcess

        gt = load_data.getGT(videoName,dataset_of_the_video).astype(int)
        frameStart = (gt == -1).sum()

        videoPath = '../vis/{}/multiAttMaps_{}_{}_{}.mp4'.format(exp_id,model_id,epoch,videoName)

        with imageio.get_writer(videoPath, mode='I',fps=20) as writer:

            print(vidInd+1,"/",len(revVideoNameDict.keys()),videoName)
            frameNb = utils.getVideoFrameNb("../data/{}/{}.avi".format(dataset_of_the_video,videoName))
            i=frameStart

            while i < len(featMapsList[0]):

                frame = video[i]
                frame = frame[frame.shape[0]-frame.shape[1]:,:]

                nbRows = math.ceil(np.sqrt(len(featMapsList)+1))
                nbCols = math.ceil(np.sqrt(len(featMapsList)+1))

                dest = Image.new('RGB', (frame.shape[1]*nbCols,frame.shape[0]*nbRows))

                framePIL = Image.fromarray(frame.astype("uint8"))
                dest.paste(framePIL, (0,0))

                for j,featMaps in enumerate(featMapsList):
                    resizedAttFeatMap = resize(featMaps[i-frameStart][0], (frame.shape[1],frame.shape[0]),anti_aliasing=True,mode="constant",order=0)
                    resizedAttFeatMap = resizedAttFeatMap[:,:,np.newaxis]

                    resizedAttFeatMap = cm(resizedAttFeatMap[:,:,0])[:,:,:3]
                    frame = (frame*resizedAttFeatMap).astype("float")

                    #print(frame.max())
                    frame = 255*(frame/frame.max())

                    framePIL = Image.fromarray(frame.astype("uint8"))
                    dest.paste(framePIL, (framePIL.size[0]*((j+1)%nbCols),framePIL.size[1]*((j+1)//nbRows)))

                dest = np.array(dest)

                nearestLowerDiv = dest.shape[0]//16
                nearestHigherDiv = (nearestLowerDiv+1)*16
                dest = resize(dest, (nearestHigherDiv,nearestHigherDiv),anti_aliasing=True,mode="constant",order=0)*255

                #dest = (255*dest.astype("float"))/dest.mean(axis=-1).max()

                writer.append_data(dest.astype("uint8"))
                i+=1

def phaseNbHist(datasets,density):

    def countRows(x):
        x = np.genfromtxt(x,delimiter=",")
        return x.shape[0]

    def computeLength(x):

        times = np.genfromtxt(x,delimiter=",")[1:]

        return times[-1,1]-times[0,1]

    for dataset in datasets:

        phases_nb_list = list(map(countRows,sorted(glob.glob("../data/{}/annotations/*phases.csv".format(dataset)))))

        plt.figure(1)
        plt.hist(phases_nb_list,label=dataset,alpha=0.5,density=density,bins=16,range=(0,16),edgecolor='black')

        videoLengths = list(map(computeLength,sorted(glob.glob("../data/{}/annotations/*timeElapsed.csv".format(dataset)))))

        plt.figure(2)
        plt.hist(videoLengths,label=dataset,alpha=0.5,bins=20,range=(0,180),density=density,edgecolor='black')

    plt.figure(1)
    plt.xticks(np.arange(16)+0.5,np.arange(16))
    plt.xlabel("Number of phases")
    plt.ylabel("Density")
    plt.title("Number of annotated phases in the dataset(s) : "+",".join(datasets))
    plt.legend()
    plt.savefig("../vis/nbScenes_{}_density{}.png".format("_".join(datasets),density))
    plt.close()

    plt.figure(2)
    plt.legend()
    plt.savefig("../vis/scenesLengths_{}_density{}.png".format("_".join(datasets),density))
    plt.close()

def plotConfusionMatrix(exp_id,model_id):

    bestWeightPath = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_id))[0]
    bestEpoch = bestWeightPath.split("epoch")[1]

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,bestEpoch)),key=utils.findNumbers))

    conf = configparser.ConfigParser()
    conf.read("../models/{}/{}.ini".format(exp_id,model_id))
    conf = conf["default"]

    videoDict = buildVideoNameDict(conf["dataset_val"],float(conf["val_part_beg"]),float(conf["val_part_end"]),str2bool(conf["prop_set_int_fmt"]),resFilePaths)

    revDict = formatData.getReversedLabels()
    labels = formatData.getLabels()
    labelInds = list(revDict.keys())

    for i,resFilePath in enumerate(resFilePaths):

        if i%5==0:
            print(i,"/",len(resFilePaths),resFilePath)

        pred = np.genfromtxt(resFilePath)
        pred = pred[:,1:].argmax(axis=-1)

        gt = load_data.getGT(videoDict[resFilePath],conf["dataset_val"])

        frameStart = (gt == -1).sum()

        pred,gt = pred[frameStart:],gt[frameStart:]
        gt = gt[:len(pred)]

        confMat = sklearn.metrics.confusion_matrix(gt, pred, labels=labelInds).astype("float")
        #confMat = confMat/confMat.sum(axis=1)

        for i in range(len(confMat)):
            if confMat[i].sum() > 0:
                confMat[i] = confMat[i]/confMat[i].sum()

        labelIndsPred,labelIndsGT = list(set(pred)),list(set(gt))
        print(labelIndsPred,labelIndsGT)
        labelsPred,labelsGT = [revDict[labelInd] for labelInd in labelIndsPred],[revDict[labelInd] for labelInd in labelIndsGT]

        plt.figure()
        img = plt.imshow(confMat)
        plt.xlabel("Predictions")
        plt.xticks(np.arange(len(labelInds)),labels,rotation=45)
        plt.ylabel("Ground-truth")
        plt.yticks(np.arange(len(labelInds)),labels)
        plt.colorbar(img)
        plt.tight_layout()
        plt.savefig("../vis/{}/confMat_{}_epoch{}_{}.png".format(exp_id,model_id,bestEpoch,videoDict[resFilePath]))
        plt.close()

def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( h, w,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf

def change_contrast(img, level):
    #from https://stackoverflow.com/questions/42045362/change-contrast-of-image-in-pil
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

def plotPoints(exp_id,model_id,epoch):

    pointsPaths = sorted(glob.glob("../results/{}/points_{}_epoch{}_*.npy".format(exp_id,model_id,epoch)))
    videoNameDict = buildVideoNameDict("small+big",0,1,False,pointsPaths,raiseError=False)

    cm = plt.get_cmap('plasma')

    for pointPath in pointsPaths:
        print(pointPath)

        pointsSeq = np.load(pointPath).astype(int)
        print(pointsSeq.shape)
        conf = configparser.ConfigParser()
        conf.read("../models/{}/{}.ini".format(exp_id,model_id))
        imgSize = int(conf["default"]["img_size"])//4

        videoName = videoNameDict[pointPath]

        xMin,xMax = pointsSeq[:,:,0].min(),pointsSeq[:,:,0].max()
        yMin,yMax = pointsSeq[:,:,1].min(),pointsSeq[:,:,1].max()

        dataset = load_data.getDataset(videoName)
        video = pims.Video("../data/{}/{}.avi".format(dataset,videoName))
        gt = load_data.getGT(videoName,dataset).astype(int)
        frameStart = (gt == -1).sum()

        with imageio.get_writer("../vis/{}/points_{}_{}_{}.mp4".format(exp_id,model_id,epoch,videoName), mode='I',fps=20) as writer:
            for i,points in enumerate(pointsSeq):

                #fig = plt.figure()
                #plt.xlim(xMin,xMax)
                #plt.ylim(yMin,yMax)

                #plt.plot(points[:,0],points[:,1],"*")
                #img = fig2data(fig)
                #plt.close()

                frame = video[i+frameStart]
                frame = frame[frame.shape[0]-frame.shape[1]:,:]
                #frame = np.array(change_contrast(Image.fromarray(frame.astype("uint8")), 0))
                frame = 255*(frame/frame.max())

                #points = (points*frame.shape[0]/imgSize).astype(int)

                mask = np.zeros((imgSize,imgSize,3)).astype("float")
                mask += 0.25

                ptsValues = np.abs(points[:,3:]).sum(axis=-1)
                ptsValues = cm(ptsValues/ptsValues.max())[:,:3]
                mask[points[:,1],points[:,0],:3] = ptsValues
                mask = resize(mask, (frame.shape[0],frame.shape[1]),anti_aliasing=True,mode="constant",order=0)

                frame = frame.astype(float)*mask

                writer.append_data(img_as_ubyte(frame.astype("uint8")))


def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--epoch_to_process',type=int,metavar="N",help='The epoch to process. This argument should be set when using the --plot_score argument.')

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--plot_score',action="store_true",help='To plot the probabilities produced by a model for all the videos processed by this model during validation for some epoch.\
                                                                            The --model_id argument must be set, along with the --exp_id, --dataset_test, --epoch_to_process, --train_part_beg, --train_part_end (for \
                                                                            computing the state transition matrix.)')

    ########## COMPUTE METRICS AND PUT THEM IN AN LATEX TABLE #############
    argreader.parser.add_argument('--eval_model',action="store_true",help='Evaluate a model using the csv files containing the scores. The --exp_id argument must be set, \
                                   along with the --epoch_to_process and the --model_ids arguments. The arguments --param_agr, --keys, and --names arguments can also be set.')

    argreader.parser.add_argument('--param_agr',type=str,nargs="*",metavar="PARAM",help='A list of hyper-parameter to use to agregate the performance of several models.')
    argreader.parser.add_argument('--keys',type=str,nargs="*",metavar="KEY",help='The list of key that will appear during aggregation. In the final csv file, each key value will be replaced by a string of the list --names.\
                                                                                  to make it easier to read.')
    argreader.parser.add_argument('--names',type=str,nargs="*",metavar="NAME",help='The list of string to replace each key by during agregation.')
    argreader.parser.add_argument('--epochs_to_process',nargs="*",type=int,metavar="N",help='The list of epoch at which to evaluate each model. This argument should be set when using the --eval_model argument.')
    argreader.parser.add_argument('--model_ids',type=str,nargs="*",metavar="NAME",help='The id of the models to process.')

    ######################## Database plot #################################

    argreader.parser.add_argument('--plot_data',type=int,metavar="N",help='To plot the state transition matrix and the prior vector. The value is the number of classes. The --dataset_test must be set.')

    ####################### Plot attention maps ###############################

    argreader.parser.add_argument('--plot_attention_maps',action="store_true",help="To plot the attention map of a model. Requires the arguments 'dataset_test', 'exp_id', 'model_id' to be set.")
    argreader.parser.add_argument('--feat_maps',action="store_true",help="To plot the feature maps instead of the attention maps.")

    argreader.parser.add_argument('--plot_multi_attention_maps',action="store_true",help="To plot the attention maps of a model producing several attention maps per image. \
                                    Requires the arguments 'dataset_test', 'exp_id', 'model_id' and 'epoch_to_process' to be set.")

    ####################### Plot  phase number histogram #####################

    argreader.parser.add_argument('--phase_nb_hist',type=str,nargs="*",metavar="DATASET",help='To plot the histogram of phase number of all video in several datasets, \
                                    along with histograms showing the video length distribution. The value of this argument is the names of the datasets.')

    argreader.parser.add_argument('--density',type=str2bool,metavar="BOOl",help='To plot the histogram on a density scale.')

    ####################### Plot confusion matrix #############################

    argreader.parser.add_argument('--plot_confusion_matrix',action="store_true",help='To plot the confusion matrix of a model at its best validation epoch \
                                        on the validation dataset. The --model_id and the --exp_id arguments must be set.')

    ####################### Plot points computed for a point net model #############################

    argreader.parser.add_argument('--plot_points',action="store_true",help='To plot the points computed by the visual module of a point net model.\
                                    The exp_id, model_id and epoch_to_process arg must be set.')

    argreader = load_data.addArgs(argreader)


    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_score:
        plotScore(args.dataset_test,args.exp_id,args.model_id,args.epoch_to_process,args.train_part_beg,args.train_part_end)
    if args.eval_model:

        if os.path.exists("../results/{}/metrics.csv".format(args.exp_id)):
            os.remove("../results/{}/metrics.csv".format(args.exp_id))

        for i,model_id in enumerate(args.model_ids):

            conf = configparser.ConfigParser()
            conf.read("../models/{}/{}.ini".format(args.exp_id,model_id))
            conf = conf["default"]


            evalModel(conf["dataset_test"],float(conf["test_part_beg"]),float(conf["test_part_end"]),str2bool(conf["prop_set_int_fmt"]),args.exp_id,model_id,epoch=args.epochs_to_process[i],\
                        regression=str2bool(conf["regression"]),uncertainty=str2bool(conf["uncertainty"]),nbClass=int(conf["class_nb"]))

        if len(args.param_agr) > 0:
            agregatePerfs(args.exp_id,args.param_agr,args.keys,args.names)
    if not args.plot_data is None:
        plotData(args.plot_data,args.dataset_test)
    if args.plot_attention_maps:
        plotAttentionMaps(args.dataset_test,args.exp_id,args.model_id,args.feat_maps)
    if args.plot_multi_attention_maps:
        plotMultiAttentionMaps(args.dataset_test,args.exp_id,args.model_id,args.epoch_to_process)
    if args.phase_nb_hist:
        phaseNbHist(args.phase_nb_hist,args.density)
    if args.plot_confusion_matrix:
        plotConfusionMatrix(args.exp_id,args.model_id)
    if args.plot_points:
        plotPoints(args.exp_id,args.model_id,args.epoch_to_process)

if __name__ == "__main__":
    main()
