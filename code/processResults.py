
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

import umap
from math import log10, floor

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

def compRFKernel(recField):
    ker = torch.abs(torch.arange(recField)-recField//2)
    ker = torch.max(ker.unsqueeze(0),ker.unsqueeze(1))
    ker = recField//2 - ker + 1
    return ker.unsqueeze(0).unsqueeze(0).float()/ker.max()

def compRecField(architecture):

    #Initial 7x7 conv with stride=2 and 3x3 max pool with stride=2
    rec_field = 1 + (6+1) + (2+1)

    if architecture == "resnet18":
        #There 8 3x3 conv
        rec_field += 8*2
    else:
        raise ValueError("Unkown architecture",architecture)

    return rec_field

def plotPointsImageDatasetGrid(exp_id,imgNb,epochs,model_ids,reduction_fact_list,inverse_xy,mode,nbClass,\
                                useDropped_list,forceFeat,fullAttMap,threshold,maps_inds,plotId,luminosity,\
                                receptive_field,cluster,cluster_attention,pond_by_norm,gradcam,nrows,correctness,args):

    if (correctness == "True" or correctness == "False") and len(model_ids)>1:
        raise ValueError("correctness can only be used with a single model.")

    torch.manual_seed(1)
    imgSize = 224

    ptsImage = torch.zeros((3,imgSize,imgSize))
    gridImage = None

    args.normalize_data = False
    args.val_batch_size = imgNb

    if len(epochs) == 0:
        for j in range(len(model_ids)):

            paths = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_ids[j]))
            if len(paths) > 1:
                raise ValueError("There should only be one best weight file.",model_ids[j],"has several.")

            fileName = os.path.basename(paths[0])
            epochs.append(utils.findLastNumbers(fileName))

    if mode == "val":
        imgLoader,_ = load_data.buildTestLoader(args,mode,shuffle=False)
        inds = torch.arange(imgNb)
        imgBatch,_ = next(iter(imgLoader))
    else:
        imgLoader,testDataset = load_data.buildTestLoader(args,mode,shuffle=False)

        if (correctness == "True" or correctness == "False"):
            targPreds = np.genfromtxt("../results/{}/{}_epoch{}.csv".format(exp_id,model_ids[0],epochs[0]),delimiter=",")[-len(testDataset):]
            targ = targPreds[:,0]
            preds = np.argmax(targPreds[:,1:],axis=1)
            correct = (targ==preds)
            if correctness == "True":
                correctInd = torch.arange(len(testDataset))[correct]
                inds = correctInd[torch.randperm(len(correctInd))][:imgNb]
            else:
                incorrectInd = torch.arange(len(testDataset))[~correct]
                inds = incorrectInd[torch.randperm(len(incorrectInd))][:imgNb]
        else:
            inds = torch.randint(len(testDataset),size=(imgNb,))
        imgBatch = torch.cat([testDataset[ind][0].unsqueeze(0) for ind in inds],dim=0)

    cmPlasma = plt.get_cmap('plasma')

    if len(inverse_xy):
        inverse_xy = [True for _ in range(len(model_ids))]

    pointPaths,pointWeightPaths = [],[]
    for j in range(len(model_ids)):

        if useDropped_list[j]:
            pointPaths.append("../results/{}/points_dropped_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("../results/{}/points_dropped_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
        elif fullAttMap[j]:
            pointPaths.append("../results/{}/attMaps_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("")
        elif gradcam[j]:
            pointPaths.append("../results/{}/gradcam_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("")
        else:
            pointPaths.append("../results/{}/points_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("../results/{}/pointWeights_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))

    meanVecList = []

    normDict = {}
    for j in range(len(pointPaths)):
        if pond_by_norm[j]:

            if j>0 and model_ids[j] == model_ids[j-1] and pond_by_norm[j-1]:
                normDict[j] = normDict[j-1]
            else:
                if not os.path.exists(pointPaths[j].replace("attMaps","norm")):
                    features = np.load(pointPaths[j].replace("attMaps","features"))
                    normDict[j] = np.sqrt(np.power(features,2).sum(axis=1,keepdims=True))
                    np.save(pointPaths[j].replace("attMaps","norm"),normDict[j])
                else:
                    normDict[j] = np.load(pointPaths[j].replace("attMaps","norm"))
        else:
            normDict[j] = None

    for i in range(imgNb):
        print("Img",i)
        if gridImage is None:
            gridImage = imgBatch[i:i+1]
        else:
            gridImage = torch.cat((gridImage,imgBatch[i:i+1]),dim=0)

        for j in range(len(pointPaths)):

            if fullAttMap[j] or gradcam[j]:
                ptsImageCopy = ptsImage.clone()
                attMap = np.load(pointPaths[j])[inds[i]]

                if gradcam[j]:
                    attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                if attMap.shape[0] != 1:
                    attMap = attMap[maps_inds[j]:maps_inds[j]+1]
                    attMap = attMap.astype(float)
                    attMap /= 255
                    attMap *= 5
                    attMap = torch.softmax(torch.tensor(attMap).float().view(-1),dim=-1).view(attMap.shape[0],attMap.shape[1],attMap.shape[2]).numpy()
                    attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                if pond_by_norm[j]:
                    norm = normDict[j][inds[i]]
                    norm = norm/norm.max()
                    attMap = norm*attMap
                    #attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                if not luminosity:
                    if cluster[j]:
                        features = np.load(pointPaths[j].replace("attMaps","features"))[inds[i]]
                        attMap = umap.UMAP(n_components=3).fit_transform(features.transpose(1,2,0).reshape(features.shape[1]*features.shape[2],features.shape[0]))
                        attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
                        attMap = attMap.reshape(features.shape[1],features.shape[2],3)
                    elif cluster_attention[j]:
                        features = np.load(pointPaths[j].replace("attMaps","features"))[inds[i]]
                        attMap = np.power(features,2).sum(axis=0,keepdims=True)
                        print(attMap.shape,attMap.min(),attMap.mean(),attMap.max())
                        if model_ids[j].lower().find("norm") != -1 or model_ids[j].lower().find("none") != -1:
                            segMask = attMap>25000
                        elif model_ids[j].lower().find("relu") != -1:
                            segMask = attMap>0
                        elif model_ids[j].lower().find("softm") != -1 or model_ids[j].lower().find("sigm") != -1:
                            segMask = attMap>0.5
                        else:
                            raise ValueError("Unkown attention :",model_ids[j])
                        flatSegMask = segMask.reshape(-1)

                        features = features.transpose(1,2,0).reshape(features.shape[1]*features.shape[2],features.shape[0])

                        embeddings = umap.UMAP(n_components=3).fit_transform(features[flatSegMask])
                        embeddings = (embeddings-embeddings.min())/(embeddings.max()-embeddings.min())
                        attMap = np.zeros((attMap.shape[0],attMap.shape[1],attMap.shape[2],3))
                        origSize = attMap.shape

                        attMap = attMap.reshape(-1,3)
                        segMask = segMask.reshape(-1)
                        attMap[segMask] = embeddings
                        attMap = attMap.reshape(origSize)[0]

                    else:
                        attMap = cmPlasma(attMap[0])[:,:,:3]
                else:
                    attMap = attMap[0][:,:,np.newaxis]
                ptsImageCopy = torch.tensor(resize(attMap, (ptsImageCopy.shape[1],ptsImageCopy.shape[2]),anti_aliasing=True,mode="constant",order=0)).permute(2,0,1).float().unsqueeze(0)
            else:

                ptsOrig = torch.tensor(np.load(pointPaths[j]))[inds[i]]

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
                else:
                    if useDropped_list[j]:
                        ptsWeights = torch.sqrt(torch.pow(ptsOrig[:,3:-1],2).sum(dim=-1)).numpy()
                    else:
                        ptsWeights = torch.sqrt(torch.pow(ptsOrig[:,3:],2).sum(dim=-1)).numpy()
                        print("Feat",ptsWeights.min(),ptsWeights.mean(),ptsWeights.max())

                if inverse_xy[j]:
                    x,y = pts[:,0],pts[:,1]
                else:
                    y,x = pts[:,0],pts[:,1]

                ptsWeights = (ptsWeights-ptsWeights.min())/(ptsWeights.max()-ptsWeights.min())
                if not luminosity:

                    if cluster[j]:
                        ptsWeights = umap.UMAP(n_components=3).fit_transform(ptsOrig[:,3:].cpu().detach().numpy())
                        ptsWeights = (ptsWeights-ptsWeights.min())/(ptsWeights.max()-ptsWeights.min())
                        #ptsWeights = np.concatenate((np.zeros((ptsWeights.shape[0],1)),ptsWeights),axis=-1)
                    else:
                        ptsWeights = cmPlasma(ptsWeights)[:,:3]

                ptsImageCopy[:,y,x] =torch.tensor(ptsWeights).permute(1,0).float()

                ptsImageCopy = ptsImageCopy.unsqueeze(0)
                ptsImageCopy = F.interpolate(ptsImageCopy, scale_factor=reduction_fact_list[j])

            if receptive_field[j]:
                rf_size = compRecField("resnet18")
                rf_kernel = compRFKernel(rf_size)
                ptsImageCopy = F.conv_transpose2d(ptsImageCopy,rf_kernel,padding=rf_size//2)
                ptsImageCopy = ptsImageCopy/ptsImageCopy.max()

            #if (not (fullAttMap[j] or gradcam[j])) or (fullAttMap[j] and luminosity):

            if luminosity:
                ptsImageCopy = ptsImageCopy*imgBatch[i:i+1]
            else:
                ptsImageCopy = 0.8*ptsImageCopy+0.2*imgBatch[i:i+1].mean(dim=1,keepdim=True)

            gridImage = torch.cat((gridImage,ptsImageCopy),dim=0)

    torchvision.utils.save_image(gridImage, "../vis/{}/points_grid_{}_{}_corr={}.png".format(exp_id,mode,plotId,correctness), nrow=(len(model_ids)+1)*nrows,padding=5,pad_value=0.5)

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

    bestPredList = sorted(glob.glob("../results/{}/*_test.csv".format(exp_id)))
    bestPredList = list(filter(lambda x:x.find("metrics") == -1,bestPredList))

    print(bestPredList)

    allAccuracy = []
    allClassErr = []

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

    printImage("../vis/{}/failCases/".format(exp_id),sortedInds[:200],test_dataset)
    printImage("../vis/{}/sucessCases/".format(exp_id),sortedInds[-200:],test_dataset)

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

def round_sig(x, sig=2):
    if x == 0:
        return 0
    else:
        return round(x, sig-int(floor(log10(abs(x))))-1)

def compileTest(exp_id,id_to_label_dict,table_id,model_ids):

    metricsToMax = {"accuracy":True,"latency":False,"sparsity":False,"sparsity_normalised":True,"ios":True}
    testFilePaths = glob.glob("../results/{}/*metrics_test*".format(exp_id))

    if not model_ids is None:
        testFilePaths = list(filter(lambda x:os.path.basename(x).replace("model","").split("_epoch")[0] in model_ids,testFilePaths))

    model_id_list = []
    perf_list = []

    for testFilePath in testFilePaths:

        model_id = os.path.basename(testFilePath).replace("model","").replace("_metrics_test.csv","")
        model_id = model_id.split("_epoch")[0]

        accuracy_rawcrop = model_id.find("Drop") != -1 and model_id.find("Crop") != -1

        test_perf = getTestPerf(testFilePath,accuracy_rawcrop)

        model_id_list.append(model_id)
        perf_list.append(test_perf)

    model_id_list = np.array(model_id_list)

    for i in range(len(perf_list)):
        print(perf_list[i],model_id_list[i])

    perf_list = {metric:np.array([perf_list[i][metric] for i in range(len(perf_list))]) for metric in metricsToMax.keys()}

    bestPerf = {}
    for metric in perf_list.keys():
        if metricsToMax[metric]:
            bestPerf[metric] = np.ma.array(perf_list[metric], mask=np.isnan(perf_list[metric])).max()
        else:
            bestPerf[metric] = np.ma.array(perf_list[metric], mask=np.isnan(perf_list[metric])).min()

    dic = {}

    for i in range(len(model_id_list)):

        keys = model_id_list[i].split("_")

        if not keys[0] in dic:
            dic[keys[0]] = {}

        if not keys[1] in dic[keys[0]]:
            dic[keys[0]][keys[1]] = {}

        if not keys[2] in dic[keys[0]][keys[1]]:
            dic[keys[0]][keys[1]][keys[2]] = {metric:perf_list[metric][i] for metric in perf_list.keys()}

        model_id_list[i] = ','.join(keys)

    latexTable = '\\begin{table}[t]  \n' + \
                  '\\begin{tabular}{*8c}\\toprule  \n' + \
                  'Pixel weighting & Pixel selection & Classification & Accuracy & Latency & Sparsity & Sparsity (Norm.) & IoS \\\\ \n' + \
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

                for l,metric in enumerate(bestPerf.keys()):

                    if not np.isnan(dic[key1][key2][key3][metric]):
                        print(dic[key1][key2][key3][metric],bestPerf[metric],2)
                        print(round_sig(dic[key1][key2][key3][metric],2),round_sig(bestPerf[metric],2))
                        if round_sig(dic[key1][key2][key3][metric],2) == round_sig(bestPerf[metric],2):
                            latexTable += "$\\mathbf{"+str(round_sig(dic[key1][key2][key3][metric],2)) + "}$ "
                        else:
                            latexTable += "$"+str(round_sig(dic[key1][key2][key3][metric],2)) + "$ "
                    else:
                        latexTable += "-"

                    if l < len(bestPerf.keys()) - 1:
                        latexTable += " & "

                latexTable += " \\\\ \n"

            if j < len(dic[key1]) -1:
                latexTable += '\\cline{2-4} \n'
        latexTable += '\\hline \n'

    latexTable += "\\end{tabular} \n\\caption{} \n\\end{table}"

    with open("../results/{}/performanceTable_{}.txt".format(exp_id,table_id),"w") as text_file:
        print(latexTable,file=text_file)

def getTestPerf(path,accuracy_rawcrop):

    try:
        perf = np.genfromtxt(path,delimiter=",",dtype=str)
        newFormat=True
    except ValueError:
        newFormat=False

    if not newFormat:
        perf = np.genfromtxt(path,delimiter="?",dtype=str)[1].split(",")
        if accuracy_rawcrop:
            perf = perf[8]
        else:
            perf = perf[0]
        perf = float(perf.replace("tensor(",""))
        return {"accuracy":perf,"sparsity":np.nan,"sparsity_normalised":np.nan,"latency":np.nan,"ios":np.nan}
    else:

        perf = np.genfromtxt(path,delimiter=",",dtype=str)

        metrics = ["accuracy","sparsity","sparsity_normalised","ios"]
        if accuracy_rawcrop:
            metrics[0] = "accuracy_rawcrop"

        metrics_dict = {}
        for metric in metrics:
            if (perf[0] == metric).sum() > 0:
                if accuracy_rawcrop and metric == "accuracy":
                    metrics_dict[metric] = float(perf[1][np.argwhere(perf[0] == "accuracy_rawcrop")])
                else:
                    metrics_dict[metric] = float(perf[1][np.argwhere(perf[0] == metric)])
            else:
                metrics_dict[metric] = np.nan
        latency_path = path.replace("model","latency_").replace("_metrics_test","")
        latency = np.genfromtxt(latency_path,delimiter=",")[3:-1,0].mean()

        metrics_dict["latency"] = latency

        return metrics_dict


def umapPlot(exp_id,model_id):
    cm = plt.get_cmap('plasma')
    bestPaths = sorted(glob.glob("../models/{}/*{}*best*".format(exp_id,model_id)))
    if len(bestPaths) > 1:
        raise ValueError("Multiple best weight files for model {} : {}".format(model_id,len(bestPaths)))
    bestPath = bestPaths[0]

    bestEpoch = utils.findNumbers(os.path.basename(bestPath).split("best")[-1])
    features = np.load("../results/{}/points_{}_epoch{}_val.npy".format(exp_id,model_id,bestEpoch))[:,:,3:]
    print(features.shape)
    for i in range(10):
        #feat = features[i].transpose(1,2,0).reshape(features[i].shape[1]*features[i].shape[2],features[i].shape[0])
        feat = features[i]
        features_emb = umap.UMAP(n_components=2).fit_transform(feat)
        features_norm = np.sqrt(np.power(feat,2).sum(axis=-1))
        features_norm = (features_norm-features_norm.min())/(features_norm.max()-features_norm.min())
        plt.figure()
        plt.scatter(features_emb[:,0],features_emb[:,1],color=cm(features_norm)[:,:3])
        plt.savefig("../vis/{}/umap_{}_img{}.png".format(exp_id,model_id,i))

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
    argreader.parser.add_argument('--maps_inds',type=int,nargs="*",metavar="INT",help='The index of the attention map to use when there is several. If there only one or if there is none, set this to -1',default=[])
    argreader.parser.add_argument('--receptive_field',type=str2bool,nargs="*",metavar="BOOL",help='To plot with the effective receptive field',default=[])
    argreader.parser.add_argument('--cluster',type=str2bool,nargs="*",metavar="BOOL",help='To cluster points with UMAP',default=[])
    argreader.parser.add_argument('--cluster_attention',type=str2bool,nargs="*",metavar="BOOL",help='To cluster attended points with UMAP',default=[])
    argreader.parser.add_argument('--pond_by_norm',type=str2bool,nargs="*",metavar="BOOL",help='To also show the norm of pixels along with the attention weights.',default=[])


    argreader.parser.add_argument('--gradcam',type=str2bool,nargs="*",metavar="BOOL",help='To plot gradcam instead of attention maps',default=[])
    argreader.parser.add_argument('--correctness',type=str,metavar="CORRECT",help='Set this to True to only show image where the model has given a correct answer.',default=None)



    argreader.parser.add_argument('--luminosity',type=str2bool,metavar="BOOL",help='To plot the attention maps not with a cmap but with luminosity',default=False)
    argreader.parser.add_argument('--nrows',type=int,metavar="INT",help='The number of rows',default=1)

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

    argreader.parser.add_argument('--compile_test',action="store_true",help='To compile the test performance of all model of an experiment. The --exp_id arg must be set. \
                                    The --model_ids can be set to put only some models in the table')

    argreader.parser.add_argument('--table_id',type=str,metavar="NAME",help='Name of the table file')

    ####################################### UMAP ############################################

    argreader.parser.add_argument('--umap',action="store_true",help='To plot features using UMAP')

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

        if len(args.receptive_field) == 0:
            args.receptive_field = [False for _ in range(len(args.model_ids))]
        if len(args.gradcam) == 0:
            args.gradcam = [False for _ in range(len(args.model_ids))]
        if len(args.cluster) == 0:
            args.cluster = [False for _ in range(len(args.model_ids))]
        if len(args.cluster_attention) == 0:
            args.cluster_attention = [False for _ in range(len(args.model_ids))]
        if len(args.pond_by_norm) == 0:
            args.pond_by_norm = [False for _ in range(len(args.model_ids))]
        plotPointsImageDatasetGrid(args.exp_id,args.image_nb,args.epoch_list,args.model_ids,args.reduction_fact_list,args.inverse_xy,args.mode,\
                                    args.class_nb,args.use_dropped_list,args.force_feat,args.full_att_map,args.use_threshold,args.maps_inds,args.plot_id,\
                                    args.luminosity,args.receptive_field,args.cluster,args.cluster_attention,args.pond_by_norm,args.gradcam,args.nrows,args.correctness,args)
    if args.plot_prob_maps:
        plotProbMaps(args.image_nb,args,args.norm)
    if args.list_best_pred:
        listBestPred(args.exp_id)
    if args.find_hard_image:
        findHardImage(args.exp_id,args.dataset_size,args.threshold,args.dataset_name,args.train_prop,args.nb_class)
    if args.efficiency_plot:
        efficiencyPlot(args.exp_id,args.model_ids,args.epoch_list)
    if args.umap:
        umapPlot(args.exp_id,args.model_id)
    if args.compile_test:

        id_to_label_dict = {"1x1":"Score prediction","none":"None","noneNoRed":"None - Stride=1","sobel":"Sobel","patchsim":"Patch Similarity","norm":"Norm","normDropCrop":"Norm + WS-DAN",
                            "1x1DropCrop":"Score prediction + WS - DAN","1x1reluDropAndCrop":"Score prediction - ReLU + WS - DAN","1x1softmscalemDropAndCrop":"Score prediction - SoftMax + WS - DAN",
                            "topk":"Top-256","topksag":"Topk-K (SAG)","all":"All","multitopk":"Multiple Top-K","top1024":"Top-1024",
                            "pn":"PointNet","pnnorm":"PointNet (norm)","avglin":"Linear","avglinzoom":"Linear + Zoom","avglinzoomindep":"Linear + Zoom Indep",
                            "1x1softmscale":"Score prediction - SoftMax","1x1softmscalenored":"Score prediction - SoftMax -- Stride=1",
                            "1x1softmscalenoredbigimg":"Score prediction - SoftMax -- Stride=1 -- Big Input Image",
                            "1x1relu":"Score prediction - ReLU",
                            "1x1NA":"Score prediction - No Aux",
                            "normNoRed":"Norm - Stride = 2",
                            "noneR50":"None - ResNet50",
                            "noneHyp":"None - BS=12, Image size=448, StepLR",
                            "noneNoRedR50":"None - Stride=1 - ResNet50",
                            "normNoAux":"Norm - No Aux",
                            "normNoAuxR50":"Norm - No Aux - ResNet50",
                            "normR50":"Norm - Resnet50",
                            "1x1reluNA":"Score prediction - ReLU - NA","1x1softmscaleNA":"Score prediction - SoftMax - NA",
                            "noneNoRedNA":"None - Stride=1 - NA","noneNoRedSupSegNA":"None - Stride=1 - SupSeg - NA",
                            "noneNoRedSupSegNosClassNA":"None - Stride=1 - SupSeg - NoClass - NA",
                            "noneR101":"None - ResNet101","normNoAuxR101":"None - No Aux. - ResNet101",
                            "bil":"Bilinear","bilreg001":"Bilinear ($\\lambda=0.01$)","bilreg01":"Bilinear ($\\lambda=0.1$)","bilreg1":"Bilinear ($\\lambda=1$)",
                            "bilreg10":"Bilinear ($\\lambda=10$)","bilreg20":"Bilinear ($\\lambda=20$)","bilreg60":"Bilinear ($\\lambda=60$)",
                            "bilSigm":"Bilinear - Sigmoid","bilRelu":"Bilinear - ReLU","bilReluMany":"Bilinear - ReLU - 32 Maps",
                            "bilClus":"Bilinear - Clustering","bilClusEns":"Bilinear - Clustering + Ensembling","bilClusEnsHidLay2": "Bil. - Clust + Ens - Hid. Lay.",
                            "bilClusEnsHidLay2Gate": "Bil. - Clust + Ens - Hid. Lay. + Gate",
                            "bilClusEnsGate":"Bil. - Clust + Ens - Gate",
                            "bilClusEnsHidLay2GateDrop": "Bil. - Clust + Ens - Hid. Lay. + Gate + Drop",
                            "bilClusEnsHidLay2GateRandDrop": "Bil. - Clust + Ens - Hid. Lay. + Gate + RandDrop",
                            "bilClusEnsHidLay2GateSoftm":"Bil. - Clust + Ens - Hid. Lay. + Gate + Softm",
                            "noneNoRedHidLay2":"None - Stride=1 - Hid. Lay.",
                            "noneSTR1":"None - Stride=1 at test","noneSTR1DIL2":"None - Stride=1,Dil=2 at test",
                            "bilFeatNorm":"Bilinear - Feature normalisation",
                            "bilReluMany00001CL":"Bilinear - ReLU - 32 Maps - $\\lambda_{CL}=0,0001$","bilReluMany0001CL":"Bilinear - ReLU - 32 Maps - $\\lambda_{CL}=0,001$",
                            "patchnoredtext":"Patch (No Red) (Text. model)"}

        compileTest(args.exp_id,id_to_label_dict,args.table_id,args.model_ids)
if __name__ == "__main__":
    main()
