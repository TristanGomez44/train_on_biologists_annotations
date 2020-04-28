
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
import cv2
from PIL import Image

import load_data

import metrics
import utils
import scipy
import sys

import configparser

import matplotlib.patheffects as path_effects
import imageio
from skimage import img_as_ubyte

from scipy import stats
import math
from PIL import ImageEnhance
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

from scipy.signal import argrelextrema
import scipy.ndimage.filters as filters
import scipy.ndimage.morphology as morphology

import torchvision
import torch_cluster

def plotPointsImageDataset(imgNb,redFact,plotDepth,args):

    cm = plt.get_cmap('plasma')

    exp_id = args.exp_id
    model_id = args.model_id

    pointPaths = sorted(glob.glob("../results/{}/points_{}_epoch*_.npy".format(exp_id,model_id)),key=utils.findNumbers)

    points = np.concatenate(list(map(lambda x:np.load(x)[:imgNb][:,:,:][np.newaxis],pointPaths)),axis=0)
    points = np.transpose(points, axes=[1,0,2,3])

    args.normalize_data = False
    imgLoader = load_data.buildTestLoader(args,"val")

    batchNb = imgNb//args.val_batch_size
    totalImgNb = 0

    for batchInd,(imgBatch,_) in enumerate(imgLoader):
        print(batchInd,imgBatch.size())
        for imgInd in range(len(imgBatch)):

            if totalImgNb<imgNb:
                print("\t",imgInd,"/",totalImgNb)
                print("\t","Writing video",imgInd)
                with imageio.get_writer("../vis/{}/points_{}_img{}_depth={}.mp4".format(exp_id,model_id,totalImgNb,plotDepth), mode='I',fps=20,quality=9) as writer:

                    img = imgBatch[imgInd].detach().permute(1,2,0).numpy().astype(float)

                    for epoch in range(len(points[imgInd])):

                        pts = points[imgInd,epoch]
                        mask = np.ones((img.shape[0]//redFact,img.shape[1]//redFact,3)).astype("float")

                        if plotDepth:
                            ptsValues = pts[:,2]
                        else:
                            ptsValues = np.abs(pts[:,3:]).sum(axis=-1)

                        ptsValues = cm(ptsValues/ptsValues.max())[:,:3]
                        mask[pts[:,1].astype(int),pts[:,0].astype(int)] = ptsValues

                        mask = resize(mask, (img.shape[0],img.shape[1]),anti_aliasing=True,mode="constant",order=0)

                        imgMasked = img*255*mask

                        imgMasked = Image.fromarray(imgMasked.astype("uint8"))
                        draw = ImageDraw.Draw(imgMasked)
                        draw.text((0, 0),str(epoch),(0,0,0))
                        imgMasked = np.asarray(imgMasked)
                        cv2.imwrite("testProcessResults.png",imgMasked)
                        writer.append_data(img_as_ubyte(imgMasked.astype("uint8")))

                    totalImgNb += 1
        if batchInd>=batchNb:
            break

def plotPointsImageDatasetGrid(exp_id,imgNb,epochs,model_ids,reduction_fact_list,inverse_xy,args):

    imgSize = 224

    ptsImage = torch.zeros((3,imgSize,imgSize))
    gridImage = None

    args.normalize_data = False
    args.val_batch_size = imgNb
    imgLoader = load_data.buildTestLoader(args,"val")
    imgBatch,_ = next(iter(imgLoader))
    cm = plt.get_cmap('plasma')

    for i in range(imgNb):
        print("Img",i)
        if gridImage is None:
            gridImage = imgBatch[i:i+1]
        else:
            gridImage = torch.cat((gridImage,imgBatch[i:i+1]),dim=0)

        for j in range(len(model_ids)):

            pts = torch.tensor(np.load("../results/{}/points_{}_epoch{}_val.npy".format(exp_id,model_ids[j],epochs[j])))[i,:,:2]

            pts = (pts*reduction_fact_list[j]).long()

            if inverse_xy[j]:
                x,y = pts[:,0],pts[:,1]
            else:
                y,x = pts[:,0],pts[:,1]

            ptsImageCopy = ptsImage.clone()

            if os.path.exists("../results/{}/pointWeights_{}_epoch{}_val.npy".format(exp_id,model_ids[j],epochs[j])):
                ptsWeights = np.load("../results/{}/pointWeights_{}_epoch{}_val.npy".format(exp_id,model_ids[j],epochs[j]))[i]

                ptsWeights = cm(ptsWeights/ptsWeights.max())[:,:3]

                ptsImageCopy[:,y,x] =  torch.tensor(ptsWeights).permute(1,0).float()
            else:
                ptsImageCopy[:,y,x] = 1

            ptsImageCopy = ptsImageCopy.unsqueeze(0)

            gridImage = torch.cat((gridImage,ptsImageCopy),dim=0)

    torchvision.utils.save_image(gridImage, "../vis/{}/points_grid.png".format(exp_id), nrow=len(model_ids)+1,padding=5,pad_value=0.5)

def plotProbMaps(imgNb,args,norm=False):

    exp_id = args.exp_id
    model_id = args.model_id

    probMapPaths = sorted(glob.glob("../results/{}/prob_map_{}_epoch*.npy".format(exp_id,model_id)),key=utils.findNumbers)
    cm = plt.get_cmap('plasma')

    probmaps = np.concatenate(list(map(lambda x:np.load(x)[:imgNb][:,:,:][np.newaxis],probMapPaths)),axis=0)

    imgLoader = load_data.buildTestLoader(args,"val",normalize=False)

    batchNb = imgNb//args.val_batch_size
    totalImgNb = 0

    for batchInd,(imgBatch,_) in enumerate(imgLoader):
        for imgInd in range(len(imgBatch)):

            if totalImgNb<imgNb:
                print("\t",imgInd,"/",totalImgNb)
                print("\t","Writing video",imgInd)
                with imageio.get_writer("../vis/{}/probmap_{}_img{}.mp4".format(exp_id,model_id,totalImgNb), mode='I',fps=20,quality=9) as writer:

                    img = imgBatch[imgInd].detach().permute(1,2,0).numpy().astype(float)

                    for epoch in range(len(probmaps)):
                        dest = Image.new('RGB', (img.shape[1]*2,img.shape[0]))
                        imgPIL = Image.fromarray((255*img).astype("uint8"))
                        dest.paste(imgPIL, (0,0))

                        probmap = probmaps[epoch,imgInd]

                        if norm:
                            probmap = (probmap-probmap.min())/(probmap.max()-probmap.min())
                            probmap *= 255

                        if args.pn_topk:
                            horizontPadd = np.zeros((probmap.shape[0],3,probmap.shape[2]))
                            probmap = np.concatenate((probmap,horizontPadd),axis=1)
                            probmap = np.concatenate((horizontPadd,probmap),axis=1)

                            verticaPadd = np.zeros((probmap.shape[0],probmap.shape[1],3))
                            probmap = np.concatenate((probmap,verticaPadd),axis=2)
                            probmap = np.concatenate((verticaPadd,probmap),axis=2)

                        probmap = resize(probmap[0], (img.shape[0],img.shape[1]),anti_aliasing=True,mode="constant",order=0)

                        probmapPIL = Image.fromarray(probmap.astype("uint8"))

                        dest.paste(probmapPIL, (img.shape[1],0))

                        draw = ImageDraw.Draw(dest)
                        draw.text((0, 0),str(epoch),(0,0,0))
                        dest = np.asarray(dest)
                        cv2.imwrite("testProcessResults.png",dest)
                        writer.append_data(img_as_ubyte(dest.astype("uint8")))

                    totalImgNb += 1
        if batchInd>=batchNb:
            break

def listBestPred(exp_id):

    bestPaths = sorted(glob.glob("../models/{}/*best*".format(exp_id)))
    bestPredPaths = []
    for path in bestPaths:

        bestEpoch = utils.findNumbers(os.path.basename(path).split("best")[-1])
        #Removing the last character because it is a "_"
        model_id = os.path.basename(path).split("best")[0][:-1].replace("model","")

        bestPredPath = "../results/{}/{}_epoch{}.csv".format(exp_id,model_id,bestEpoch)

        if os.path.exists(bestPredPath):
            bestPredPaths.append(bestPredPath)
        else:
            print("file {} does not exist".format(bestPredPath))

    with open("../results/{}/bestPred.txt".format(exp_id),"w") as text_file:
        for path in bestPredPaths:
            print(path,file=text_file)

def findHardImage(exp_id,dataset_size,threshold,datasetName,trainProp,nbClass):

    allBestPredLists = sorted(glob.glob("../results/{}/bestPred_*".format(exp_id)))

    allAccuracy = []
    allClassErr = []

    for bestPredList in allBestPredLists:
        bestPredList = np.genfromtxt(bestPredList,dtype=str)

        for i,bestPred in enumerate(bestPredList):
            label = np.genfromtxt(bestPred,delimiter=",")[1:,0]

            if len(label) == dataset_size:
                bestPred = np.genfromtxt(bestPred,delimiter=",")[1:,1:].argmax(axis=1)
                accuracy = (label==bestPred)

                if accuracy.mean() >= threshold:
                    allAccuracy.append(accuracy[np.newaxis])

                    classErr = np.zeros(nbClass)

                    for i in range(nbClass):
                        classErr[i] = ((label==i)*(bestPred!=i)).sum()
                    allClassErr.append(classErr[np.newaxis])

    print("Nb models :",len(allAccuracy))
    allAccuracy = np.concatenate(allAccuracy,axis=0)
    allAccuracy = allAccuracy.mean(axis=0)
    sortedInds = np.argsort(allAccuracy)

    plt.figure()
    plt.ylabel("Proportion of models to answer correctly")
    plt.xlabel("Image index")
    plt.plot(np.arange(len(allAccuracy))/len(allAccuracy),allAccuracy[sortedInds])
    plt.savefig("../vis/{}/failCases.png".format(exp_id))
    plt.close()

    test_dataset = torchvision.datasets.ImageFolder("../data/{}".format(datasetName))

    np.random.seed(1)
    torch.manual_seed(1)

    totalLength = len(test_dataset)
    _, test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength * trainProp),
                                                                   totalLength - int(totalLength * trainProp)])

    printImage("../vis/{}/failCases/".format(exp_id),sortedInds[:20],test_dataset)
    printImage("../vis/{}/sucessCases/".format(exp_id),sortedInds[-20:],test_dataset)

    allClassErr = np.concatenate(allClassErr,axis=0).mean(axis=0)
    plt.figure()
    plt.plot(allClassErr)
    plt.xlabel("Class index")
    plt.ylabel("Average error number")
    plt.savefig("../vis/{}/classErr.png".format(exp_id))
    plt.close()

    ratioList = []
    for i in range(len(sortedInds)):
        shape = test_dataset.__getitem__(i)[0].size
        ratio = shape[0]/shape[1]
        ratioList.append(ratio)
    plt.figure()
    plt.plot(ratioList,allAccuracy,"*")
    plt.savefig("../vis/{}/ratioAcc.png".format(exp_id))


def printImage(path,indexs,test_dataset):
    if not os.path.exists(path):
        os.makedirs(path)
    for index in indexs:
        image = test_dataset.__getitem__(index)[0]
        image.save(path+"/{}.png".format(index))

def efficiencyPlot(exp_id,model_ids,epoch_list):

    if not os.path.exists("../vis/{}/".format(exp_id)):
        os.makedirs("../vis/{}/".format(exp_id))

    plt.figure()
    latList = []
    accList = []

    for i in range(len(model_ids)):
        latency =np.genfromtxt("../results/{}/latency_{}_epoch{}.csv".format(exp_id,model_ids[i],epoch_list[i]),delimiter=",")[1:-1,0].mean()
        accuracy = np.genfromtxt("../results/{}/model{}_epoch{}_metrics_test.csv".format(exp_id,model_ids[i],epoch_list[i]),dtype=str)[1,0]

        if accuracy.find("tensor") != -1:
            accuracy = float(accuracy.replace("tensor","").replace(",","").replace("(",""))

        latList.append(latency)
        accList.append(accuracy)

        plt.plot(latency,accuracy,"*",label=model_ids[i])

    plt.legend()
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.savefig("../vis/{}/acc_vs_lat.png".format(exp_id))

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ####################### Plot points computed for a point net model #############################

    argreader.parser.add_argument('--plot_points_image_dataset',action="store_true",help='To plot the points computed by the visual module of a point net model \
                                    on an image dataset. The -c (config file), --image_nb and --reduction_fact arg must be set.')

    argreader.parser.add_argument('--plot_prob_maps',action="store_true",help='To plot the points computed by the visual module of a point net model \
                                    on an image dataset. The -c (config file), --image_nb arg must be set.')

    argreader.parser.add_argument('--image_nb',type=int,metavar="INT",help='For the --plot_points_image_dataset and the --plot_prob_maps args. \
                                    The number of images to plot the points of.')

    argreader.parser.add_argument('--reduction_fact',type=int,metavar="INT",help='For the --plot_points_image_dataset arg.\
                                    The reduction factor of the point cloud size compared to the full image. For example if the image has size \
                                    224x224 and the points cloud lies in a 56x56 frame, this arg should be 224/56=4')

    argreader.parser.add_argument('--plot_depth',type=str2bool,metavar="BOOL",help='For the --plot_points_image_dataset arg. Plots the depth instead of point feature norm.')
    argreader.parser.add_argument('--norm',type=str2bool,metavar="BOOL",help='For the --plot_prob_maps arg. Normalise each prob map.')

    ######################################## GRID #################################################

    argreader.parser.add_argument('--plot_points_image_dataset_grid',action="store_true",help='Same as --plot_points_image_dataset but plot only on image and for several model.')
    argreader.parser.add_argument('--epoch_list',type=int,metavar="INT",nargs="*",help='The list of epochs at which to get the points.')
    argreader.parser.add_argument('--model_ids',type=str,metavar="IDS",nargs="*",help='The list of model ids.')
    argreader.parser.add_argument('--reduction_fact_list',type=float,metavar="INT",nargs="*",help='The list of reduction factor.')
    argreader.parser.add_argument('--inverse_xy',type=str2bool,nargs="*",metavar="BOOL",help='To inverse x and y')

    ######################################## Find failure cases #########################################""

    argreader.parser.add_argument('--list_best_pred',action="store_true",help='To create a file listing the prediction for all models at their best epoch')
    argreader.parser.add_argument('--find_hard_image',action="store_true",help='To find the hard image indexs')
    argreader.parser.add_argument('--dataset_size',type=int,metavar="INT",help='Size of the dataset (not the whole dataset, but the concerned part)')
    argreader.parser.add_argument('--threshold',type=float,metavar="INT",help='Accuracy threshold above which a model is taken into account')
    argreader.parser.add_argument('--dataset_name',type=str,metavar="NAME",help='Name of the dataset')
    argreader.parser.add_argument('--nb_class',type=int,metavar="NAME",help='Nb of big classes')

    ####################################### Efficiency plot #########################################"""

    argreader.parser.add_argument('--efficiency_plot',action="store_true",help='to plot accuracy vs latency/model size. --exp_id, --model_ids and --epoch_list must be set.')

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_points_image_dataset:
        plotPointsImageDataset(args.image_nb,args.reduction_fact,args.plot_depth,args)
    if args.plot_points_image_dataset_grid:
        plotPointsImageDatasetGrid(args.exp_id,args.image_nb,args.epoch_list,args.model_ids,args.reduction_fact_list,args.inverse_xy,args)
    if args.plot_prob_maps:
        plotProbMaps(args.image_nb,args,args.norm)
    if args.list_best_pred:
        listBestPred(args.exp_id)
    if args.find_hard_image:
        findHardImage(args.exp_id,args.dataset_size,args.threshold,args.dataset_name,args.train_prop,args.nb_class)
    if args.efficiency_plot:
        efficiencyPlot(args.exp_id,args.model_ids,args.epoch_list)
if __name__ == "__main__":
    main()
