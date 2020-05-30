
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

import torchvision
import torch_cluster

from torch.distributions.normal import Normal
from torch import tensor

import torch.nn.functional as F

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

def plotPointsImageDatasetGrid(exp_id,imgNb,epochs,model_ids,reduction_fact_list,inverse_xy,mode,nbClass,useDropped_list,forceFeat,fullAttMap,threshold,plotId,args):

    imgSize = 224

    ptsImage = torch.zeros((3,imgSize,imgSize))
    gridImage = None

    args.normalize_data = False
    args.val_batch_size = imgNb
    imgLoader = load_data.buildTestLoader(args,mode,shuffle=False)
    imgBatch,labels = next(iter(imgLoader))
    cmPlasma = plt.get_cmap('plasma')

    if len(inverse_xy):
        inverse_xy = [True for _ in range(len(model_ids))]

    if len(epochs) == 0:
        for j in range(len(model_ids)):

            paths = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_ids[j]))
            if len(paths) > 1:
                raise ValueError("There should only be one best weight file.",model_ids[j],"has several.")

            fileName = os.path.basename(paths[0])
            epochs.append(utils.findLastNumbers(fileName))

    pointPaths,pointWeightPaths = [],[]
    for j in range(len(model_ids)):

        if useDropped_list[j]:
            pointPaths.append("../results/{}/points_dropped_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("../results/{}/points_dropped_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
        elif fullAttMap[j]:
            pointPaths.append("../results/{}/attMaps_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("")
        else:
            pointPaths.append("../results/{}/points_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("../results/{}/pointWeights_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))

    meanVecList = []
    for i in range(imgNb):
        print("Img",i)
        if gridImage is None:
            gridImage = imgBatch[i:i+1]
        else:
            gridImage = torch.cat((gridImage,imgBatch[i:i+1]),dim=0)

        for j in range(len(pointPaths)):

            if fullAttMap[j]:
                ptsImageCopy = ptsImage.clone()
                attMap = np.load(pointPaths[j])[i]
                attMap = cmPlasma(attMap[0])[:,:,:3]
                ptsImageCopy = torch.tensor(resize(attMap, (ptsImageCopy.shape[1],ptsImageCopy.shape[2]),anti_aliasing=True,mode="constant",order=0)).permute(2,0,1).float().unsqueeze(0)
            else:

                ptsOrig = torch.tensor(np.load(pointPaths[j]))[i]

                if (ptsOrig[:,:3] < 0).sum() > 0:
                    pts = (((ptsOrig[:,:3] + 1)/2)).long()
                else:
                    pts = (ptsOrig).long()

                ptsImageCopy = F.interpolate(ptsImage.unsqueeze(0), scale_factor=1/reduction_fact_list[j]).squeeze(0)

                if os.path.exists(pointWeightPaths[j]) and not forceFeat[j]:
                    if useDropped_list[j]:
                        ptsWeights = np.load(pointWeightPaths[j])[i][:,-1]
                    else:
                        ptsWeights = np.load(pointWeightPaths[j])[i]
                    plt.figure()
                    plt.hist(ptsWeights,range=(0,1),bins=10)
                    plt.savefig("../vis/{}/grid_weight_hist_{}_img{}.png".format(exp_id,model_ids[j],i))
                    plt.close()
                else:
                    if useDropped_list[j]:
                        ptsWeights = torch.sqrt(torch.pow(ptsOrig[:,3:-1],2).sum(dim=-1)).numpy()
                    else:
                        ptsWeights = torch.sqrt(torch.pow(ptsOrig[:,3:],2).sum(dim=-1)).numpy()
                        print("Feat",ptsWeights.min(),ptsWeights.mean(),ptsWeights.max())

                if threshold[j]:
                    #pts = pts[ptsWeights > 75]
                    #ptsWeights = ptsWeights[ptsWeights > 75]

                    #pi = em(torch.tensor(ptsWeights).unsqueeze(1),2)

                    plt.figure(i*len(pointPaths)+j)
                    bins = plt.hist(ptsWeights,range=(0,200),bins=20,alpha=0.5)
                    #median = (bins[1][np.argmax(bins[0])]+bins[1][np.argmax(bins[0])+1])/2
                    medianInd = np.nonzero(np.r_[True, bins[0][1:] > bins[0][:-1]] & np.r_[bins[0][:-1] > bins[0][1:], True])[0][0]

                    #print(medianInd,bins[0])
                    median = (bins[1][medianInd]+bins[1][medianInd+1])/2
                    #if (2*median < ptsWeights).sum() > 0:
                    #    pts = pts[2*median < ptsWeights]
                    #    ptsWeights = ptsWeights[2*median < ptsWeights]
                    #else:
                    print(median)
                    pts = pts[2*median < ptsWeights]
                    ptsWeights = ptsWeights[2*median < ptsWeights]
                    plt.hist(ptsWeights,range=(0,200),bins=20,alpha=0.5)
                    plt.savefig("../vis/{}/grid_norm_hist_{}_img{}.png".format(exp_id,model_ids[j],i))
                    plt.close()

                    #inds = ptsWeights.argsort()[-256:]
                    #inds = inds[ptsWeights[inds] > 75]
                    #bounding_pts = pts[inds][:,:2]
                    #min,max = bounding_pts.min(dim=0)[0],bounding_pts.max(dim=0)[0]
                    #ptsWeights = ptsWeights[(min[0] < pts[:,0])*(pts[:,0] < max[0])*(min[1] < pts[:,1])*(pts[:,1] < max[1])]
                    #pts = pts[(min[0] < pts[:,0])*(pts[:,0] < max[0])*(min[1] < pts[:,1])*(pts[:,1] < max[1])]

                if inverse_xy[j]:
                    x,y = pts[:,0],pts[:,1]
                else:
                    y,x = pts[:,0],pts[:,1]

                ptsWeights = (ptsWeights-ptsWeights.min())/(ptsWeights.max()-ptsWeights.min())
                ptsWeights = cmPlasma(ptsWeights)[:,:3]
                ptsImageCopy[:,y,x] =torch.tensor(ptsWeights).permute(1,0).float()

                ptsImageCopy = ptsImageCopy.unsqueeze(0)
                ptsImageCopy = F.interpolate(ptsImageCopy, scale_factor=reduction_fact_list[j])

                ptsImageCopy = 0.5*ptsImageCopy+0.5*imgBatch[i:i+1]

            gridImage = torch.cat((gridImage,ptsImageCopy),dim=0)

    torchvision.utils.save_image(gridImage, "../vis/{}/points_grid_{}_{}.png".format(exp_id,mode,plotId), nrow=len(model_ids)+1,padding=5,pad_value=0.5)

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

    for i in range(len(model_ids)):
        accuracy = np.genfromtxt("../results/{}/model{}_epoch{}_metrics_test.csv".format(exp_id,model_ids[i],epoch_list[i]),dtype=str)[1,0]
        if accuracy.find("tensor") != -1:
            accuracy = float(accuracy.replace("tensor","").replace(",","").replace("(",""))

        if os.path.exists("../results/{}/latency_{}_epoch{}.csv".format(exp_id,model_ids[i],epoch_list[i])):
            latency_and_batchsize =np.genfromtxt("../results/{}/latency_{}_epoch{}.csv".format(exp_id,model_ids[i],epoch_list[i]),delimiter=",")
            latency = latency_and_batchsize[1:-1,0].mean()
            latency /= latency_and_batchsize[1,1]

            plt.figure(0)
            plt.plot(latency,accuracy,"*",label=model_ids[i])

        if os.path.exists("../models/{}/model{}_best_epoch{}".format(exp_id,model_ids[i],epoch_list[i])):
            weights = torch.load("../models/{}/model{}_best_epoch{}".format(exp_id,model_ids[i],epoch_list[i]),map_location=torch.device('cpu'))
            totalElem = 0
            for key in weights:
                totalElem += weights[key].numel()

            plt.figure(1)
            plt.plot(totalElem,accuracy,"*",label=model_ids[i])

    plt.figure(0)
    plt.legend()
    plt.xlabel("Latency (seconds)")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.savefig("../vis/{}/acc_vs_lat.png".format(exp_id))

    plt.figure(1)
    plt.legend()
    plt.xlabel("Parameter number")
    plt.ylabel("Accuracy")
    plt.ylim(0,1)
    plt.savefig("../vis/{}/acc_vs_paramNb.png".format(exp_id))


def em(x,k):

    n = len(x)  # must be even number
    dims = 1
    eps = torch.finfo(torch.float32).eps

    mu = torch.tensor([50,100]).float()
    covar = torch.tensor([1,50]).float()
    converged = False
    i = 0
    h = None

    while not converged:

        prev_mu = mu.clone()
        prev_covar = covar.clone()

        h = Normal(mu, covar)

        llhood = h.log_prob(x)

        log_sum_lhood = torch.logsumexp(llhood, dim=1, keepdim=True)
        log_posterior = llhood - log_sum_lhood

        pi = torch.exp(log_posterior.reshape(n, k))
        pi = pi * (1- k * eps) + eps

        mu = torch.sum(x * pi, dim=0) / torch.sum(pi, dim=0)

        delta = pi * (x - mu)

        covar = (delta*delta).sum(dim=0)/pi.sum(dim=0)

        converged = (torch.abs(mu-prev_mu).mean() < 1) and (torch.abs(covar-prev_covar).mean() < 1)

        i += 1

    return pi

def getTestPerf(path):

    perf = np.genfromtxt(path,delimiter="?",dtype=str)[1].split(",")[0]
    perf = float(perf.replace("tensor(",""))
    return perf

def compileTest(exp_id,id_to_label_dict):

    header = 'Pixel weighting,Pixel selection,Classification,Accuracy'
    testFilePaths = glob.glob("../results/{}/*metrics_test*".format(exp_id))

    model_id_list = []
    perf_list = []

    for testFilePath in testFilePaths:

        model_id = os.path.basename(testFilePath).replace("model","").replace("_metrics_test.csv","")
        model_id = model_id.split("_epoch")[0]

        test_perf = getTestPerf(testFilePath)

        model_id_list.append(model_id)
        perf_list.append(test_perf)

    model_id_list,perf_list = np.array(model_id_list),np.array(perf_list)

    bestPerf = perf_list.max()

    dic = {}

    for i in range(len(model_id_list)):

        keys = model_id_list[i].split("_")

        if not keys[0] in dic:
            dic[keys[0]] = {}

        if not keys[1] in dic[keys[0]]:
            dic[keys[0]][keys[1]] = {}

        if not keys[2] in dic[keys[0]][keys[1]]:
            dic[keys[0]][keys[1]][keys[2]] = perf_list[i]

        model_id_list[i] = ','.join(keys)

    new_model_id_list = []
    new_perf_list = []

    for key1 in sorted(dic):
        for key2 in sorted(dic[key1]):
            for key3 in sorted(dic[key1][key2]):
                new_model_id_list.append(",".join([key1,key2,key3]))
                new_perf_list.append(dic[key1][key2][key3])

    model_id_list = new_model_id_list
    perf_list = new_perf_list

    model_id_list,perf_list = np.array(model_id_list),np.array(perf_list)

    latexTable = '\\begin{table}[t]  \n' + \
                  '\\centering  \n' + \
                  '\\begin{tabular}{*4c}\\toprule  \n' + \
                  'Pixel weighting & Pixel selection & Classification & Accuracy \\\\ \n' + \
                  '\\hline \n'

    for key1 in sorted(dic):
        n = sum([len(dic[key1][tmp_key2]) for tmp_key2 in dic[key1]])
        latexTable += '\\multirow{'+str(n)+'}{*}{'+id_to_label_dict[key1]+'} &'

        for j,key2 in enumerate(sorted(dic[key1])):
            m = len(dic[key1][key2])

            if j == 0:
                latexTable += '\\multirow{'+str(m)+'}{*}{'+id_to_label_dict[key2]+'} &'
            else:
                latexTable += '& \\multirow{'+str(m)+'}{*}{'+id_to_label_dict[key2]+'} &'

            for i,key3 in enumerate(sorted(dic[key1][key2])):

                if i > 0:
                    latexTable += "&&"

                latexTable += id_to_label_dict[key3] + " & "

                if dic[key1][key2][key3] == bestPerf:
                    latexTable += "$\\mathbf{"+str(round(dic[key1][key2][key3],2)) + "}$ \\\\ \n"
                else:
                    latexTable += "$"+str(round(dic[key1][key2][key3],2)) + "$ \\\\ \n"

            if j < len(dic[key1]) -1:
                latexTable += '\\cline{2-4} \n'
        latexTable += '\\hline \n'

    latexTable += "\\end{tabular} \n\\caption{} \n\\end{table}"

    with open("../results/{}/test.csv".format(exp_id),"w") as text_file:
        print(latexTable,file=text_file)

    #sorted_args = np.argsort(perf_list)
    #model_id_list = model_id_list[sorted_args]
    #perf_list = perf_list[sorted_args]

    #fullArray = np.concatenate((model_id_list[:,np.newaxis],perf_list[:,np.newaxis]),axis=1)
    #np.savetxt("../results/{}/test.csv".format(exp_id),fullArray,header=header,fmt="%s",delimiter=",")



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
                                    The number of images to plot the points of.',default=10)

    argreader.parser.add_argument('--reduction_fact',type=int,metavar="INT",help='For the --plot_points_image_dataset arg.\
                                    The reduction factor of the point cloud size compared to the full image. For example if the image has size \
                                    224x224 and the points cloud lies in a 56x56 frame, this arg should be 224/56=4')

    argreader.parser.add_argument('--plot_depth',type=str2bool,metavar="BOOL",help='For the --plot_points_image_dataset arg. Plots the depth instead of point feature norm.')
    argreader.parser.add_argument('--norm',type=str2bool,metavar="BOOL",help='For the --plot_prob_maps arg. Normalise each prob map.')

    ######################################## GRID #################################################

    argreader.parser.add_argument('--plot_points_image_dataset_grid',action="store_true",help='Same as --plot_points_image_dataset but plot only on image and for several model.')
    argreader.parser.add_argument('--epoch_list',type=int,metavar="INT",nargs="*",help='The list of epochs at which to get the points.',default=[])
    argreader.parser.add_argument('--model_ids',type=str,metavar="IDS",nargs="*",help='The list of model ids.')
    argreader.parser.add_argument('--reduction_fact_list',type=float,metavar="INT",nargs="*",help='The list of reduction factor.')
    argreader.parser.add_argument('--inverse_xy',type=str2bool,nargs="*",metavar="BOOL",help='To inverse x and y',default=[])
    argreader.parser.add_argument('--use_dropped_list',type=str2bool,nargs="*",metavar="BOOL",help='To plot the dropped point instead of all the points',default=[])
    argreader.parser.add_argument('--full_att_map',type=str2bool,nargs="*",metavar="BOOL",help='A list of boolean indicating if the model produces full attention maps or selects points.',default=[])
    argreader.parser.add_argument('--use_threshold',type=str2bool,nargs="*",metavar="BOOL",help='To apply the threshold to filter out points',default=[])

    argreader.parser.add_argument('--mode',type=str,metavar="MODE",help='Can be "val" or "test".',default="val")
    argreader.parser.add_argument('--force_feat',type=str2bool,nargs="*",metavar="BOOL",help='To force feature plotting even when there is attention weights available.',default=[])
    argreader.parser.add_argument('--plot_id',type=str,metavar="ID",help='The plot id',default="")

    ######################################## Find failure cases #########################################""

    argreader.parser.add_argument('--list_best_pred',action="store_true",help='To create a file listing the prediction for all models at their best epoch')
    argreader.parser.add_argument('--find_hard_image',action="store_true",help='To find the hard image indexs')
    argreader.parser.add_argument('--dataset_size',type=int,metavar="INT",help='Size of the dataset (not the whole dataset, but the concerned part)')
    argreader.parser.add_argument('--threshold',type=float,metavar="INT",help='Accuracy threshold above which a model is taken into account')
    argreader.parser.add_argument('--dataset_name',type=str,metavar="NAME",help='Name of the dataset')
    argreader.parser.add_argument('--nb_class',type=int,metavar="NAME",help='Nb of big classes')

    ####################################### Efficiency plot #########################################"""

    argreader.parser.add_argument('--efficiency_plot',action="store_true",help='to plot accuracy vs latency/model size. --exp_id, --model_ids and --epoch_list must be set.')

    ######################################## Compile test performance ##################################

    argreader.parser.add_argument('--compile_test',action="store_true",help='To compile the test performance of all model of an experiment. The --exp_id arg must be set.')



    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_points_image_dataset:
        plotPointsImageDataset(args.image_nb,args.reduction_fact,args.plot_depth,args)
    if args.plot_points_image_dataset_grid:
        if args.exp_id == "default":
            args.exp_id = "CUB3"
        plotPointsImageDatasetGrid(args.exp_id,args.image_nb,args.epoch_list,args.model_ids,args.reduction_fact_list,args.inverse_xy,args.mode,\
                                    args.class_nb,args.use_dropped_list,args.force_feat,args.full_att_map,args.use_threshold,args.plot_id,args)
    if args.plot_prob_maps:
        plotProbMaps(args.image_nb,args,args.norm)
    if args.list_best_pred:
        listBestPred(args.exp_id)
    if args.find_hard_image:
        findHardImage(args.exp_id,args.dataset_size,args.threshold,args.dataset_name,args.train_prop,args.nb_class)
    if args.efficiency_plot:
        efficiencyPlot(args.exp_id,args.model_ids,args.epoch_list)
    if args.compile_test:

        id_to_label_dict = {"1x1":"Score prediction","none":"None","sobel":"Sobel","patchsim":"Patch Similarity","norm":"Norm",
                            "topk":"Top-K","topksag":"Topk-K (SAG)","all":"All",
                            "pn":"PointNet","pnnorm":"PointNet (norm)","avglin":"Linear"}

        compileTest(args.exp_id,id_to_label_dict)
if __name__ == "__main__":
    main()
