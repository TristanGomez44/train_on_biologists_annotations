
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

import matplotlib.patheffects as path_effects

def evalModel(exp_id,model_id,model_name,epoch):
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
    - model_name (str): the label of the model. It will be used to identify the model in the result table. Eg. : 'Res50-Res50 (Youtube-large)'
    - epoch (int): the epoch at which to evaluate

    '''

    resFilePaths = np.array(sorted(glob.glob("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch)),key=utils.findNumbers))
    videoNameDict = buildVideoNameDict(0,1,resFilePaths)
    resFilePaths = np.array(list(filter(lambda x:x in videoNameDict.keys(),resFilePaths)))
    print("../results/{}/{}_epoch{}_*.csv".format(exp_id,model_id,epoch))
    #Store the value of the f-score of for video and for each threshold
    metTun = {}
    metEval = {"Accuracy":    np.zeros(len(resFilePaths))}

    for j,path in enumerate(resFilePaths):

        fileName = os.path.basename(os.path.splitext(path)[0])
        videoName = videoNameDict[path]

        #Compute the metrics with the default threshold (0.5) and with a threshold tuned on each video with a leave-one out method
        metEval["Accuracy"][j] = computeMetrics(path,videoName,resFilePaths,videoNameDict,metTun,"Accuracy")

    #Writing the latex table
    printHeader = not os.path.exists("../results/metrics.csv")
    with open("../results/metrics.csv","a") as text_file:
        if printHeader:
            print("Model,Accuracy",file=text_file)

        print("\multirow{2}{*}{"+model_name+"}"+"&"+formatMetr(metEval["Accuracy"])+"\\\\",file=text_file)
        print("\hline",file=text_file)

    print("Accuracy : ",str(round(metEval["Accuracy"].mean(),2)))

def computeMetrics(path,videoName,resFilePaths,videoNameDict,metTun,metric):
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

    gt = load_data.getGT(videoName).astype(int)
    scores = np.genfromtxt(path,delimiter=",")[:,1:]

    pred = scores.argmax(axis=-1).astype(int)

    print(pred[np.newaxis,:].shape)
    print(gt[np.newaxis,:].shape)
    metr_dict = metrics.binaryToMetrics(torch.tensor(pred[np.newaxis,:]),torch.tensor(gt[np.newaxis,:]))

    return metr_dict[metric]

def formatMetr(metricValuesArr):
    return "$"+str(round(metricValuesArr.mean(),2))+" \pm "+str(round(metricValuesArr.std(),2))+"$"

def plotScore(exp_id,model_id,epoch,trainPartBeg,trainPartEnd):
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
    cmap = cm.hsv(np.linspace(0, 1, len(revLabelDict.keys())))

    resFilePaths = sorted(glob.glob("../results/{}/{}_epoch{}*.csv".format(exp_id,model_id,epoch)))
    print("../results/{}/{}_epoch{}*.csv".format(exp_id,model_id,epoch))
    videoPaths = load_data.findVideos(propStart=0,propEnd=1)
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

            scoreAxNb = 4

            f, axList = plt.subplots(scoreAxNb+2, 1,figsize=(30,5))
            #ax1 = fig.add_subplot(111)
            ax1 = axList[0]
            #box = ax1.get_position()
            #ax1.set_position([box.x0, box.y0, box.width * 0.1, box.height])

            ax1.set_xlim(0,nbFrames)
            ax1.set_xlabel("GT")
            #for item in ([ax1.title, ax1.xaxis.label, ax1.yaxis.label] + ax1.get_xticklabels() + ax1.get_yticklabels()):
            #    item.set_fontsize(20)

            legHandles = []

            #Plot the ground truth phases
            gt = np.genfromtxt("../data/annotations/"+videoName+"_phases.csv",dtype=str,delimiter=",")
            for i,phase in enumerate(gt):
                rect = patches.Rectangle((int(phase[1]),0),int(phase[2])-int(phase[1]),1,linewidth=1,color=cmap[labDict[phase[0]]],alpha=1,label=phase[0])
                legHandles += [ax1.add_patch(rect)]

            #for i in range(scores.shape[0]):
                #scores = torch.nn.functional.softmax(torch.tensor(scores[i,:]),dim=-1).numpy()
            expVal = np.exp(scores)
            scores = expVal/expVal.sum(axis=-1,keepdims=True)

            #Plot the scores
            for i in range(scores.shape[1]):
                #scoresSoftM = torch.nn.functional.softmax(torch.tensor(scores[:,i]),dim=-1)

                ax = axList[i//((scores.shape[1]+1)//scoreAxNb)+1]
                ax.set_xlim(0,nbFrames)
                fill = ax.fill_between(np.arange(len(scores[:,i])), 0, scores[:,i],label=revLabelDict[i],color=cmap[i])
                #ax.plot(np.arange(len(scores[:,i])),scores[:,i],label=revLabelDict[i],color=cmap[i])

            #Plot the prediction with viterbi decoding
            transMat = torch.zeros((scores.shape[1],scores.shape[1]))
            priors = torch.zeros((scores.shape[1],))
            transMat,_ = trainVal.computeTransMat(transMat,priors,trainPartBeg,trainPartEnd)

            predSeqs,_ = metrics.viterbi_decode(torch.log(torch.tensor(scores).float()),torch.log(transMat),top_k=1)

            reverLabDict = formatData.getReversedLabels()

            predSeq = labelIndList2FrameInd(predSeqs[0],reverLabDict)

            for i,phase in enumerate(predSeq):
                rect = patches.Rectangle((int(phase[1]),0),int(phase[2])-int(phase[1]),1,linewidth=1,color=cmap[labDict[phase[0]]],alpha=1,label=phase[0])
                axList[-1].set_xlim(0,nbFrames)
                axList[-1].add_patch(rect)

            plt.xlabel("Time (frame index)")
            #plt.ylabel("Probability")
            #plt.tight_layout()
            #plt.text(10,0,)
            plt.text(0, 15, "Prediction", fontsize=14, transform=plt.gcf().transFigure)
            ax1.legend(bbox_to_anchor=(1.1, 1.05),prop={'size': 15})
            plt.subplots_adjust(hspace=0.6)
            plt.savefig("../vis/{}/{}_epoch{}_video{}_scores.png".format(exp_id,model_id,epoch,fileName))
            plt.close()

        else:
            raise ValueError("Unkown video : ",path)

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

def buildVideoNameDict(test_part_beg,test_part_end,resFilePaths):

    ''' Build a dictionnary associating a path to a video name (it can be the path to any file than contain the name of a video in its file name) '''

    videoPaths = list(filter(lambda x:x.find(".wav") == -1,sorted(glob.glob("../data/*.*"))))
    videoPaths = list(filter(lambda x:x.find(".xml") == -1,videoPaths))
    videoPaths = list(filter(lambda x:os.path.isfile(x),videoPaths))
    videoPaths = np.array(videoPaths)[int(test_part_beg*len(videoPaths)):int(test_part_end*len(videoPaths))]
    videoNames = list(map(lambda x:os.path.basename(os.path.splitext(x)[0]),videoPaths))
    videoNameDict = {}

    for path in resFilePaths:
        for videoName in videoNames:
            if "_"+videoName.replace("__","_")+".csv" in path.replace("__","_"):
                videoNameDict[path] = videoName
        if path not in videoNameDict.keys():
            raise ValueError("The path "+" "+path+" "+"doesnt have a video name")

    return videoNameDict

def plotData(nbClass):

    transMat = torch.zeros((nbClass,nbClass))
    priors = torch.zeros((nbClass,))
    transMat,priors = trainVal.computeTransMat(transMat,priors,0,1)

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

    videoPaths = load_data.findVideos(0,1)
    nbImages=0
    for videoPath in videoPaths:
        nbImages += len(load_data.getGT(os.path.splitext(os.path.basename(videoPath))[0]))

    plt.figure()
    plt.bar(np.arange(nbClass),priors*nbImages)
    plt.xticks(np.arange(nbClass),labels,rotation=45)
    plt.xlabel("Developpement phases")
    plt.ylabel("Number of image")
    plt.tight_layout()
    plt.savefig("../vis/prior.png")

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--plot_score',action="store_true",help='To plot the probabilities produced by a model for all the videos processed by this model during validation for some epoch.\
                                                                            The --model_id argument must be set, along with the --exp_id, --epoch_to_plot, --train_part_beg, --train_part_end (for \
                                                                            computing the state transition matrix.)')

    argreader.parser.add_argument('--epoch_to_plot',type=int,metavar="N",help='The epoch at which to plot the predictions when using the --plot_score argument')

    ########## COMPUTE METRICS AND PUT THEM IN AN LATEX TABLE #############
    argreader.parser.add_argument('--eval_model',type=int,help='Evaluate a model using the csv files containing the scores. The value of this arg is the epoch at which to evaluate the model \
                                                                            The --model_id argument must be set, along with the --model_name, --exp_id arguments.')
    argreader.parser.add_argument('--model_name',type=str,metavar="NAME",help='The name of the model as will appear in the latex table produced by the --eval_model argument.')

    ######################## Database plot #################################

    argreader.parser.add_argument('--plot_data',type=int,metavar="N",help='To plot the state transition matrix and the prior vector/ The value is the number of classes.')

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_score:
        plotScore(args.exp_id,args.model_id,args.epoch_to_plot,args.train_part_beg,args.train_part_end)
    if not args.eval_model is None:
        evalModel(args.exp_id,args.model_id,args.model_name,epoch=args.eval_model)
    if not args.plot_data is None:
        plotData(args.plot_data)

if __name__ == "__main__":
    main()
