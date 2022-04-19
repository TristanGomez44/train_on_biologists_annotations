
from asyncio import run
from sndhdr import what
from scipy.ndimage.interpolation import rotate
from args import ArgReader
from args import str2bool
import os
import glob

import torch
import numpy as np

from skimage.transform import resize
from skimage import img_as_ubyte
import matplotlib.pyplot as plt
from matplotlib import cm 
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import sklearn
import matplotlib
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


from scipy import stats
import math
from PIL import ImageEnhance
from PIL import ImageFont
from PIL import ImageDraw

import torchvision

from torch.distributions.normal import Normal
from torch import tensor

import torch.nn.functional as F

import umap
from sklearn.decomposition import PCA
from sklearn.manifold import  TSNE
from sklearn.cluster import AgglomerativeClustering

from math import log10, floor

import io
import skimage

import scipy.stats
import formatData

import sqlite3

from scipy.signal import savgol_filter
from scipy.stats import kendalltau

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
                                receptive_field,cluster,cluster_attention,pond_by_norm,gradcam,gradcam_maps,gradcam_pp,score_map,varGrad,smoothGradSq,rise,nrows,correctness,\
                                agregateMultiAtt,plotVecEmb,onlyNorm,class_index,ind_to_keep,interp,direct_ind,no_ref,viz_id,args):

    if (correctness == "True" or correctness == "False") and len(model_ids)>1:
        raise ValueError("correctness can only be used with a single model.")

    torch.manual_seed(1)
    imgSize = 448

    ptsImage = torch.zeros((3,imgSize,imgSize))
    gridImage = None

    args.normalize_data = False
    args.val_batch_size = imgNb

    if len(epochs) == 0:
        for j in range(len(model_ids)):
            paths = glob.glob("../models/{}/model{}_best_epoch*".format(exp_id,model_ids[j]))
            if len(paths) > 1:
                raise ValueError("There should only be one best weight file.",model_ids[j],"has several.")
            elif len(paths) == 0:

                paths = glob.glob("../results/{}/attMaps_{}_epoch*_test.npy".format(exp_id,model_ids[j]))

                if len(paths) > 1:
                    raise ValueError("There should only be one best att maps file.",model_ids[j],"has several.")
                else:
                    epochs.append(os.path.basename(paths[0]).split("epoch")[1].split("_{}".format(mode))[0])
            else:
                fileName = os.path.basename(paths[0])
                epochs.append(utils.findLastNumbers(fileName))

    pointPaths,pointWeightPaths = [],[]
    suff = "" if viz_id == "" else "{}_".format(viz_id)
    for j in range(len(model_ids)):
        if gradcam_maps[j]:
            pointPaths.append("../results/{}/gradcam_maps_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")
        elif gradcam_pp[j]:
            pointPaths.append("../results/{}/gradcam_pp_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")
        elif score_map[j]:
            pointPaths.append("../results/{}/score_maps_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")   
        elif varGrad[j]:
            pointPaths.append("../results/{}/vargrad_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")   
        elif smoothGradSq[j]:
            pointPaths.append("../results/{}/smoothgrad_sq_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")              
        elif gradcam[j]:
            pointPaths.append("../results/{}/gradcam_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")
        elif rise[j]:
            pointPaths.append("../results/{}/rise_maps_{}_epoch{}_{}{}.npy".format(exp_id,model_ids[j],epochs[j],suff,mode))
            pointWeightPaths.append("")
        elif fullAttMap[j]:
            pointPaths.append("../results/{}/attMaps_{}_epoch{}_{}.npy".format(exp_id,model_ids[j],epochs[j],mode))
            pointWeightPaths.append("")
        else:
            raise ValueError("Unvalid choice")

    if mode == "val":
        imgLoader,testDataset = load_data.buildTestLoader(args,mode,shuffle=False)
        #inds = torch.arange(imgNb)
        #imgBatch,_ = next(iter(imgLoader))

        inds = []
        earlyImgs = []
        for i in range(len(testDataset)):
            if testDataset[i][1]==0 or testDataset[i][1]==13:
                inds.append(i)
                earlyImgs.append(testDataset[i][0])

        imgBatch = torch.cat([img.unsqueeze(0) for img in earlyImgs],dim=0)
        #imgBatch = torch.cat([testDataset[ind][0].unsqueeze(0) for ind in inds],dim=0)
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
            maxInd = None

            for i in range(len(pointPaths)):

                if not direct_ind[i]:
                    if maxInd is None:
                        if gradcam[i]:
                            maxInd = len(np.load(pointPaths[i].replace("attMaps","gradcam"),mmap_mode="r"))
                        else:
                            maxInd = len(np.load(pointPaths[i]))
                    else:
                        indAtt = len(np.load(pointPaths[i],mmap_mode="r"))

                        if os.path.exists(pointPaths[i].replace("attMaps","features")):
                            print(pointPaths[i])
                            indFeat = len(np.load(pointPaths[i].replace("attMaps","features"),mmap_mode="r"))
                        else:
                            indFeat = len(np.load(pointPaths[i].replace("attMaps","norm"),mmap_mode="r"))

                        if maxInd < indAtt:
                            maxInd = indAtt
                        if maxInd < indFeat:
                            maxInd = indFeat

            #Looking for the image at which the class we want begins
            if not class_index is None:
                startInd = 0

                classes = sorted(map(lambda x:x.split("/")[-2],glob.glob("../data/{}/*/".format(args.dataset_test))))
                for ind in range(class_index):
                    className = classes[ind]
                    startInd += len(glob.glob("../data/{}/{}/*".format(args.dataset_test,className)))

                className = classes[class_index]

                endInd = startInd + len(glob.glob("../data/{}/{}/*".format(args.dataset_test,className)))

            else:
                startInd = 0
                endInd = maxInd

            inds = torch.randint(startInd,endInd,size=(imgNb,))

            if not ind_to_keep is None:
                ind_to_keep = np.array(ind_to_keep)-1
                inds = inds[ind_to_keep]

            print("inds",inds)

            #In case there is not enough images
            imgNb = min(len(inds),imgNb)

        if args.shuffle_test_set:
            perm = load_data.RandomSampler(testDataset,args.seed).randPerm

            #inds = [perm[ind] for ind in inds]
            imgBatch = torch.cat([testDataset[perm[ind]][0].unsqueeze(0) for ind in inds],dim=0)
        else:
            imgBatch = torch.cat([testDataset[ind][0].unsqueeze(0) for ind in inds],dim=0)

    cmPlasma = plt.get_cmap('plasma')

    if len(inverse_xy):
        inverse_xy = [True for _ in range(len(model_ids))]

    meanVecList = []

    normDict = {}
    for j in range(len(pointPaths)):
        if (pond_by_norm[j] or onlyNorm[j]) and ((not gradcam[j]) or gradcam_maps[j]):

            if j>0 and model_ids[j] == model_ids[j-1] and pond_by_norm[j-1] and (not normDict[j-1] is None):
                normDict[j] = normDict[j-1]
            else:
                if gradcam_maps[j]:
                    normDict[j] = np.load(pointPaths[j].replace("gradcam_maps","gradcam"))
                else:
                    if not os.path.exists(pointPaths[j].replace("attMaps","norm")):
                        normDict[j] = compNorm(pointPaths[j].replace("attMaps","features"))
                        np.save(pointPaths[j].replace("attMaps","norm"),normDict[j])
                    else:
                        normDict[j] = np.load(pointPaths[j].replace("attMaps","norm"))
                if len(normDict[j].shape) == 3:
                    normDict[j] = normDict[j][:,np.newaxis]

        else:
            normDict[j] = None

    vecEmb_list = []
    for j in range(len(pointPaths)):
        if plotVecEmb[j]:
            vecEmb = np.load("../results/{}/vecEmb_{}_test.npy".format(exp_id,model_ids[j]))
            vecEmb = (vecEmb-vecEmb.min())/(vecEmb.max()-vecEmb.min())
            vecEmb_list.append(vecEmb)
        else:
            vecEmb_list.append(None)

    fnt = ImageFont.truetype("arial.ttf", 40)

    if exp_id.find("EMB") != -1:
        pred = np.genfromtxt("../results/{}/{}_epoch{}_test.csv".format(exp_id,model_ids[0],epochs[0]),delimiter=",")[1:,1:].argmax(axis=1)
        class_aSort = list(formatData.labelDict.keys())
        class_aSort.sort()
        class_realSort = [formatData.getRevLab()[i] for i in range(len(class_aSort))]

    for i in range(imgNb):

        if i % 10 == 0:
            print("i",i)

        img = imgBatch[i:i+1]

        img = (img-img.min())/(img.max()-img.min())

        if args.print_ind:
            imgPIL = Image.fromarray((255*img[0].permute(1,2,0).numpy()).astype("uint8"))
            imgDraw = ImageDraw.Draw(imgPIL)

            rectW = 180

            imgDraw.rectangle([(0,0), (rectW, 40)],fill="white")

            imgDraw.text((0,0), str(i+1)+" ", font=fnt,fill=(0,0,0))

            if exp_id.find("EMB") != -1:
                if pred[inds[i]] == testDataset[inds[i]][1]:
                    imgDraw.text((60,0), class_realSort[pred[inds[i]]], font=fnt,fill=(0,230,0))
                else:
                    imgDraw.text((60,0), class_realSort[pred[inds[i]]], font=fnt,fill=(255,0,0))
                    imgDraw.text((150,0), "("+class_realSort[testDataset[inds[i]][1]]+")", font=fnt,fill=(0,0,0))

            img = torch.tensor(np.array(imgPIL)).permute(2,0,1).unsqueeze(0).float()/255

        if gridImage is None:
            gridImage = img
        else:
            gridImage = torch.cat((gridImage,img),dim=0)

        for j in range(len(pointPaths)):

            if fullAttMap[j] or gradcam[j] or rise[j]:
                ptsImageCopy = ptsImage.clone()

                if not direct_ind[j]:
                    attMap = np.load(pointPaths[j],mmap_mode="r")[inds[i]]
                else:
                    attMap = np.load(pointPaths[j],mmap_mode="r")[i]

                if no_ref[j]:
                    attMap_max = attMap.max(axis=-1,keepdims=True).max(axis=-2,keepdims=True)
                    attMap = attMap*(attMap==attMap_max)

                if gradcam[j] or rise[j]:
                    attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                    if gradcam_maps[j] or varGrad[j] or smoothGradSq[j]:
                        if attMap.shape[0] == 3:
                            if gradcam_maps[j]:
                                attMap = np.abs(attMap-0.5)
                            attMap = attMap.mean(axis=0,keepdims=True)

                            attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                        else:
                            raise ValueError("AttMaps has wrong shape. Model",j,model_ids[j])

                if attMap.shape[0] != 1 and not onlyNorm[j]:
                    if maps_inds[j] == -1:

                        if attMap.shape[0] == 4:
                            attMap = attMap[:3]

                        attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                        if agregateMultiAtt[j]:
                            attMap = attMap.mean(axis=0,keepdims=True)
                            attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
                        elif plotVecEmb[j]:
                            attMap = attMap[0:1]*vecEmb_list[j][inds[i]][0][:,np.newaxis,np.newaxis]
                            attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
                    else:
                        attMap = attMap[maps_inds[j]:maps_inds[j]+1]
                        attMap = attMap.astype(float)
                        attMap /= 255
                        attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
                        attMap = (attMap > 0.5).astype(float)
                else:
                    if not gradcam[j] and not rise[j]:
                        norm = normDict[j][inds[i]]
                        norm = (norm-norm.min())/(norm.max()-norm.min())
                        attMap = norm

                if pond_by_norm[j] and not (gradcam[j] or rise[j]):
                    if direct_ind[j]:
                        norm = normDict[j][i]
                    else:
                        norm = normDict[j][inds[i]]
                    norm = (norm-norm.min())/(norm.max()-norm.min())

                    if norm.shape[1:] != attMap.shape[1:]:
                        norm = resize(np.transpose(norm,(1,2,0)), (attMap.shape[1],attMap.shape[2]),anti_aliasing=True,mode="constant",order=0)
                        norm = np.transpose(norm,(2,0,1))

                    attMap = norm*attMap
                    attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

                if not luminosity:
                    if cluster[j]:
                        features = np.load(pointPaths[j].replace("attMaps","features"))[inds[i]]
                        attMap = umap.UMAP(n_components=3).fit_transform(features.transpose(1,2,0).reshape(features.shape[1]*features.shape[2],features.shape[0]))
                        attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())
                        attMap = attMap.reshape(features.shape[1],features.shape[2],3)
                    elif cluster_attention[j]:
                        features = np.load(pointPaths[j].replace("attMaps","features"))[inds[i]]
                        attMap = np.power(features,2).sum(axis=0,keepdims=True)

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
                        if attMap.shape[0] == 1:
                            attMap = cmPlasma(attMap[0])[:,:,:3]
                        else:
                            attMap = np.transpose(attMap,(1,2,0))
                else:
                    attMap = attMap[0][:,:,np.newaxis]

                interpOrder = 1 if interp[j] else 0
                ptsImageCopy = torch.tensor(resize(attMap, (ptsImageCopy.shape[1],ptsImageCopy.shape[2]),anti_aliasing=True,mode="constant",order=interpOrder)).permute(2,0,1).float().unsqueeze(0)

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

                if inverse_xy[j]:
                    x,y = pts[:,0],pts[:,1]
                else:
                    y,x = pts[:,0],pts[:,1]

                ptsWeights = (ptsWeights-ptsWeights.min())/(ptsWeights.max()-ptsWeights.min())
                if not luminosity:

                    if cluster[j]:
                        ptsWeights = umap.UMAP(n_components=3).fit_transform(ptsOrig[:,3:].cpu().detach().numpy())
                        ptsWeights = (ptsWeights-ptsWeights.min())/(ptsWeights.max()-ptsWeights.min())
                    else:
                        ptsWeights = cmPlasma(ptsWeights)[:,:3]

                ptsImageCopy[:,y,x] =torch.tensor(ptsWeights).permute(1,0).float()

                ptsImageCopy = ptsImageCopy.unsqueeze(0)
                ptsImageCopy = F.interpolate(ptsImageCopy, scale_factor=reduction_fact_list[j])

            if receptive_field[j]:
                rf_size = compRecField("resnet18")
                rf_kernel = compRFKernel(rf_size)
                ptsImageCopy = F.conv_transpose2d(ptsImageCopy,rf_kernel,padding=rf_size//2)

            if luminosity:
                ptsImageCopy = ptsImageCopy*imgBatch[i:i+1]

            else:
                img = imgBatch[i:i+1].mean(dim=1,keepdim=True)
                img = (img-img.min())/(img.max()-img.min())

                ptsImageCopy = 0.8*ptsImageCopy+0.2*img

            gridImage = torch.cat((gridImage,ptsImageCopy),dim=0)

            if len(gridImage)//(1+len(model_ids)) > 100:
                outPath = "../vis/{}/{}_{}.png".format(exp_id,plotId,gridImage[0].mean())
                torchvision.utils.save_image(gridImage, outPath, nrow=(len(model_ids)+1)*nrows)
                os.system("convert  -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))
                gridImage = None

    outPath = "../vis/{}/{}.png".format(exp_id,plotId)
    torchvision.utils.save_image(gridImage, outPath, nrow=(len(model_ids)+1)*nrows)
    os.system("convert  -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))

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

    for i in range(10):
        #feat = features[i].transpose(1,2,0).reshape(features[i].shape[1]*features[i].shape[2],features[i].shape[0])
        feat = features[i]
        features_emb = umap.UMAP(n_components=2).fit_transform(feat)
        features_norm = np.sqrt(np.power(feat,2).sum(axis=-1))
        features_norm = (features_norm-features_norm.min())/(features_norm.max()-features_norm.min())
        plt.figure()
        plt.scatter(features_emb[:,0],features_emb[:,1],color=cm(features_norm)[:,:3])
        plt.savefig("../vis/{}/umap_{}_img{}.png".format(exp_id,model_id,i))

def latency(exp_id):

    model_ids = []
    latencies = []
    latFiles = sorted(glob.glob("../results/{}/latency*".format(exp_id)))
    for latFile in latFiles:
        latency = np.genfromtxt(latFile,delimiter=",")[3:-1,0].mean()
        model_id = os.path.basename(latFile).replace("latency_","").split("epoch")[0][:-1]

        latencies.append(latency)
        model_ids.append(model_id)

    csv = np.concatenate((np.array(model_ids)[:,np.newaxis],np.array(latencies)[:,np.newaxis].astype(str)),axis=1)
    np.savetxt("../results/{}/latencies.csv".format(exp_id),csv,fmt='%s, %s,')

def param_nb(exp_id):

    weightFiles = glob.glob("../models/{}/*best*".format(exp_id))
    model_ids = []
    paramNbList = []
    for i,weightFile in enumerate(weightFiles):

        print(i,"/",len(weightFiles),weightFile)

        state_dict = torch.load(weightFile,map_location=torch.device('cpu'))

        paramCount = 0
        for param in state_dict.keys():
            if torch.is_tensor(state_dict[param]):
                paramCount += state_dict[param].numel()

        model_ids.append(os.path.basename(weightFile).replace("model","").split("best")[0][:-1])
        paramNbList.append(paramCount)

    csv = np.concatenate((np.array(model_ids)[:,np.newaxis],np.array(paramNbList)[:,np.newaxis].astype(str)),axis=1)
    np.savetxt("../results/{}/param_nb.csv".format(exp_id),csv,fmt='%s, %s,')

def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def normalize(img):
    img_min = img.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0]
    img_max = img.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    return (img - img_min)/(img_max - img_min)

def agrVec(exp_id,model_id,args,classMin=0,classMax=19,redDim=2):

    attMaps = torch.tensor(np.load(glob.glob("../results/{}/attMaps_{}_epoch*_test.npy".format(exp_id,model_id))[0]))
    attMaps = attMaps.float()/attMaps.sum(dim=(2,3),keepdim=True)

    feat = torch.tensor(np.load(glob.glob("../results/{}/features_{}_epoch*_test.npy".format(exp_id,model_id))[0]))

    norm = torch.tensor(np.load(glob.glob("../results/{}/norm_{}_epoch*_test.npy".format(exp_id,model_id))[0]))

    targetPaths = glob.glob("../results/{}/{}_epoch*_test.csv".format(exp_id,model_id))

    true_labels = np.genfromtxt(targetPaths[0],delimiter=",")[1:len(feat)+1,0]
    pred = np.genfromtxt(targetPaths[0],delimiter=",")[1:len(feat)+1,1:]
    labels = pred.argmax(axis=1)
    entropy = -(torch.softmax(torch.tensor(pred),dim=1)*F.log_softmax(torch.tensor(pred), dim=1)).sum(dim=1).numpy()

    _,testDataset = load_data.buildTestLoader(args,"test",shuffle=False)
    imgBatch = torch.cat([testDataset[i][0].unsqueeze(0) for i in range(len(labels))],dim=0)

    if not os.path.exists("../results/{}/objSize_{}_test.npy".format(exp_id,model_id)):
        imgTestPaths = sorted(glob.glob("../data/{}/*/*.jpg".format(args.dataset_test)))
        imgSeg = list(map(lambda x:cv2.imread(x.replace("test","seg").replace(".jpg",".png")),imgTestPaths))
        imgSize = imgBatch.size(-1)
        #Resize
        imgSeg = list(map(lambda x:resize(x,(imgSize,imgSize*x.shape[1]//x.shape[0])) if x.shape[0] < x.shape[1] \
                              else resize(x,(imgSize*x.shape[0]//x.shape[1],imgSize)),imgSeg))
        #Center Crop
        for i in range(len(imgSeg)):
            if imgSeg[i].shape[0] < imgSeg[i].shape[1]:
                imgSeg[i] = imgSeg[i][:,(imgSeg[i].shape[1]-imgSeg[i].shape[0])//2:-(imgSeg[i].shape[1]-imgSeg[i].shape[0])//2]
            elif imgSeg[i].shape[1] < imgSeg[i].shape[0]:
                imgSeg[i] = imgSeg[i][(imgSeg[i].shape[0]-imgSeg[i].shape[1])//2:-(imgSeg[i].shape[0]-imgSeg[i].shape[1])//2,:]

        #Sum
        imgSeg = list(map(lambda x:(x.astype("float")/255).sum(),imgSeg))

        imgSeg = np.array(imgSeg)[:len(labels)]
        np.save("../results/{}/objSize_{}_test.npy".format(exp_id,model_id),imgSeg)
        imgSeg = torch.tensor(imgSeg)

    else:
        imgSeg = torch.tensor(np.load("../results/{}/objSize_{}_test.npy".format(exp_id,model_id)))

    attMaps = attMaps[labels==true_labels]
    feat = feat[labels==true_labels]
    norm = norm[labels==true_labels]
    imgBatch = imgBatch[labels==true_labels]
    imgSeg = imgSeg[labels==true_labels]
    entropy = entropy[labels==true_labels]
    labels = labels[labels==true_labels]

    attMaps,feat,norm,imgBatch,imgSeg,entropy,labels = attMaps[(classMin<=labels) * (labels<=classMax)],feat[(classMin<=labels) * (labels<=classMax)],\
                                                        norm[(classMin<=labels) * (labels<=classMax)],imgBatch[(classMin<=labels) * (labels<=classMax)],\
                                                        imgSeg[(classMin<=labels) * (labels<=classMax)],entropy[(classMin<=labels) * (labels<=classMax)],\
                                                        labels[(classMin<=labels) * (labels<=classMax)]

    allAttMaps = attMaps.reshape(attMaps.size(0)*attMaps.size(1),1,attMaps.size(2),attMaps.size(3))
    allNorm = np.repeat(norm,3,1).reshape(norm.shape[0]*3,1,norm.shape[-2],norm.shape[-2])
    imgBatch = imgBatch.unsqueeze(1).expand(-1,3,-1,-1,-1)
    imgSeg = imgSeg.unsqueeze(1).expand(-1,3)
    allImgSeg = imgSeg.reshape(imgSeg.size(0)*3)
    allImgBatch = imgBatch.reshape(imgBatch.size(0)*3,imgBatch.size(2),imgBatch.size(3),imgBatch.size(4))

    labels = np.repeat(labels[:,np.newaxis],attMaps.size(1),axis=1)
    labels = labels.reshape(-1)
    entropy = np.repeat(entropy[:,np.newaxis],attMaps.size(1),axis=1)
    entropy = entropy.reshape(-1)

    labels_cat = []

    classNb = classMax - classMin + 1
    if classNb <= 20:
        cm = plt.get_cmap('tab20')
    else:
        cm = plt.get_cmap('rainbow')

    attMaps_chunks = torch.split(attMaps, 50)
    feat_chunks = torch.split(feat, 50)

    if not os.path.exists("../results/{}/umap_{}_{}to{}.png".format(exp_id,model_id,classMin,classMax)):

        allVec = None
        for i,(attMaps,feat) in enumerate(zip(attMaps_chunks,feat_chunks)):

            if i % 10 == 0:
                print(i,"/",len(attMaps_chunks))

            #vectors = (attMaps.unsqueeze(2)*feat.unsqueeze(1)).sum(dim=(3,4))
            #vectors = vectors.reshape(vectors.size(0)*vectors.size(1),vectors.size(2))

            #vectors = feat.float().mean(dim=(2,3))

            vectors = feat.permute(0,2,3,1).reshape(feat.size(0),feat.size(2)*feat.size(3),feat.size(1))
            vectors = vectors[:,torch.arange(feat.size(-1)*feat.size(-2)) % 300 == 0]
            vectors = vectors.reshape(vectors.size(0)*vectors.size(1),vectors.size(2))

            if allVec is None:
                allVec = vectors
            else:
                allVec = torch.cat((allVec,vectors),dim=0)

        print("Starting UMAP computation")

        allVec_emb = umap.UMAP(n_components=redDim,random_state=0).fit_transform(allVec.numpy())
        np.save("../results/{}/umap_{}_{}to{}.npy".format(exp_id,model_id,classMin,classMax),allVec_emb)

        norm_vec = torch.sqrt(torch.pow(allVec,2).sum(dim=1).float())
        np.save("../results/{}/normVec_{}_test.npy".format(exp_id,model_id),norm_vec)

    else:
        allVec_emb = np.load("../results/{}/umap_{}_{}to{}.npy".format(exp_id,model_id,classMin,classMax))
        norm_vec = np.load("../results/{}/normVec_{}_test.npy".format(exp_id,model_id))

    allVec_emb = allVec_emb - allVec_emb.mean(axis=0,keepdims=True)

    labels = labels - labels.min()

    plt.figure()

    if len(list(set(labels))) == 1:
        plt.scatter(allVec_emb[:,0],allVec_emb[:,1])
    else:
        plt.scatter(allVec_emb[:,0],allVec_emb[:,1],color=cm(labels*1.0/labels.max()))
    plt.title("Class {} to {}".format(classMin,classMax))
    plt.savefig("../vis/{}/umap_{}_{}to{}.png".format(exp_id,model_id,classMin,classMax))
    plt.close()

    corrList = []
    corrNormBySegList = []
    corrList_spar = []
    corrNormBySegList_spar = []
    corrList_norm = []
    for labelInd in range(classNb):
        print("Class",labelInd)

        allVec_emb_norm = np.sqrt(np.power(allVec_emb[labels == labelInd],2).sum(axis=-1))
        sortedInds = np.argsort(allVec_emb_norm)

        norm = allNorm[labels == labelInd]
        norm = norm/norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]

        attMaps = normalize(allAttMaps[labels == labelInd])
        imgBatch = normalize(allImgBatch[labels == labelInd])
        imgSeg = allImgSeg[labels == labelInd]

        attMaps = attMaps*norm
        attMaps = F.interpolate(attMaps,imgBatch.size(-1))

        if len(list(set(labels))) > 1:
            classColor = cm(labels[labels==labelInd][0]*1.0/labels.max())
        else:
            classColor = cm(0)

        if not os.path.exists("../vis/{}/umap_{}_class{}.gif".format(exp_id,model_id,labelInd)):
            with imageio.get_writer("../vis/{}/umap_{}_class{}.gif".format(exp_id,model_id,labelInd), mode='I',duration=1) as writer:

                for i,pts in enumerate(allVec_emb[labels == labelInd][sortedInds]):
                    fig = plt.figure()
                    if len(list(set(labels))) > 1:
                        plt.scatter(allVec_emb[:,0],allVec_emb[:,1],color=cm(labels*1.0/labels.max()))
                        plt.scatter(allVec_emb[labels==labelInd,0],allVec_emb[labels==labelInd,1],color=classColor)
                        plt.scatter(pts[np.newaxis,0],pts[np.newaxis,1],color="black",marker="*")
                    else:
                        plt.scatter(allVec_emb[labels==labelInd,0],allVec_emb[labels==labelInd,1])
                        plt.scatter(pts[np.newaxis,0],pts[np.newaxis,1],color="black",marker="*")

                    #plt.savefig("../vis/{}/umap_{}_class{}_pts{}.png".format(exp_id,model_id,labelInd,i))
                    twoDPlot = get_img_from_fig(fig, dpi=90)
                    plt.close()

                    classColor_np = np.array(classColor[:-1])

                    attMaps_moreLight = attMaps[sortedInds[i]]*classColor_np[:,np.newaxis,np.newaxis]
                    attMaps_moreLight = attMaps_moreLight/attMaps_moreLight.reshape(-1).max()
                    #plt.figure()
                    img = imgBatch[sortedInds[i]]*(attMaps_moreLight*0.95+0.05)
                    img = img.permute(1,2,0).numpy()
                    #plt.imshow(img)
                    #plt.savefig("../vis/{}/img_class{}_pts{}.png".format(exp_id,labelInd,i))
                    #plt.close()

                    img_res = skimage.transform.resize(img, (twoDPlot.shape[0],twoDPlot.shape[0]))*255
                    fullFig = np.concatenate((twoDPlot,img_res),axis=1)
                    #cv2.imwrite("../vis/{}/fullFig_class{}_pts{}.png".format(exp_id,labelInd,i),fullFig[:,:,::-1])
                    writer.append_data(img_as_ubyte(fullFig.astype("uint8")))

        avgAttAct = attMaps.mean(dim=(1,2,3))
        spars = (attMaps/attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]).mean(dim=(1,2,3))

        plt.figure()
        plt.xlabel("Distance to cloud center")
        plt.ylabel("Average activation of the attention map")
        plt.scatter(allVec_emb_norm,avgAttAct,color=classColor)
        plt.savefig("../vis/{}/distToCloud_vs_attAct_{}_class{}.png".format(exp_id,model_id,labelInd))
        plt.close()

        plt.figure()
        plt.xlabel("Distance to cloud center")
        plt.ylabel("Average activation of the attention map (normalised by object size)")
        plt.scatter(allVec_emb_norm,avgAttAct/imgSeg,color=classColor)
        plt.savefig("../vis/{}/distToCloud_vs_attAct_{}_class{}_normByObjSize.png".format(exp_id,model_id,labelInd))
        plt.close()

        plt.figure()
        plt.xlabel("Distance to cloud center")
        plt.ylabel("Sparsity of the attention map")
        plt.scatter(allVec_emb_norm,spars,color=classColor)
        plt.savefig("../vis/{}/distToCloud_vs_spars_{}_class{}.png".format(exp_id,model_id,labelInd))
        plt.close()

        plt.figure()
        plt.xlabel("Distance to cloud center")
        plt.ylabel("Sparsity of the attention map (normalised by object size)")
        plt.scatter(allVec_emb_norm,spars/imgSeg,color=classColor)
        plt.savefig("../vis/{}/distToCloud_vs_spars_{}_class{}_normByObjSize.png".format(exp_id,model_id,labelInd))
        plt.close()

        plt.figure()
        plt.xlabel("Distance to cloud center")
        plt.ylabel("Norm of the vector")
        print(allVec_emb_norm.shape,norm_vec[labels==labelInd].shape)
        plt.scatter(allVec_emb_norm,norm_vec[labels==labelInd],color=classColor)
        plt.savefig("../vis/{}/distToCloud_vs_norm_{}_class{}_normByObjSize.png".format(exp_id,model_id,labelInd))
        plt.close()

        corrList.append(np.corrcoef(allVec_emb_norm,avgAttAct)[0,1])
        corrNormBySegList.append(np.corrcoef(allVec_emb_norm,avgAttAct/imgSeg)[0,1])
        corrList_spar.append(np.corrcoef(allVec_emb_norm,spars)[0,1])
        corrNormBySegList_spar.append(np.corrcoef(allVec_emb_norm,spars/imgSeg)[0,1])
        corrList_norm.append(np.corrcoef(allVec_emb_norm,norm_vec[labels==labelInd])[0,1])

    corrList = np.array(corrList)
    corrNormBySegList = np.array(corrNormBySegList)
    corrList_spar = np.array(corrList_spar)
    corrNormBySegList_spar = np.array(corrNormBySegList_spar)
    corrList_norm = np.array(corrList_norm)

    fullCSV = np.concatenate((np.arange(classNb)[:,np.newaxis],corrList[:,np.newaxis],corrNormBySegList[:,np.newaxis]),axis=1)
    np.savetxt("../results/{}/distToCloud_vs_attAct_corr_{}.csv".format(exp_id,model_id),fullCSV)

    fullCSV = np.concatenate((np.arange(classNb)[:,np.newaxis],corrList_spar[:,np.newaxis],corrNormBySegList_spar[:,np.newaxis]),axis=1)
    np.savetxt("../results/{}/distToCloud_vs_spars_corr_{}.csv".format(exp_id,model_id),fullCSV)

    fullCSV = np.concatenate((np.arange(classNb)[:,np.newaxis],corrList_norm[:,np.newaxis]),axis=1)
    np.savetxt("../results/{}/distToCloud_vs_norm_corr_{}.csv".format(exp_id,model_id),fullCSV)

def compNorm(featPath):

    features = np.load(featPath,mmap_mode="r+")
    nbFeat = features.shape[0]
    splitLen = [100*(i+1) for i in range(nbFeat//100)]
    features_split = np.split(features,splitLen)

    allNorm = None
    for feat in features_split:
        norm = np.sqrt(np.power(feat.astype(float),2).sum(axis=1,keepdims=True))
        if allNorm is None:
            allNorm = norm
        else:
            allNorm = np.concatenate((allNorm,norm),axis=0)
        print(feat.shape)
    return allNorm

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def importancePlot(exp_id,model_id,imgNb=15,debug=False,plotStds=False,plotConfInter=False):

    # attMaps = N 3 H W
    attMaps = torch.tensor(np.load(glob.glob("../results/{}/attMaps_{}_epoch*_test.npy".format(exp_id,model_id))[0]))

    #Masking maps where nothing salient was detected
    mask = torch.zeros(1,1,attMaps.shape[2],attMaps.shape[3])
    mask[0,0,0,0] = 1
    attMaps_max = attMaps.max(dim=-2,keepdim=True)[0].max(dim=-1,keepdim=True)[0]
    attMaps = attMaps * ~((attMaps == attMaps_max) == mask)

    featPath = glob.glob("../results/{}/features_{}_epoch*_test.npy".format(exp_id,model_id))[0]
    epoch = int(os.path.basename(featPath).split("epoch")[1].split("_")[0])

    # norm = N 1 H W
    if not os.path.exists("../results/{}/norm_{}_epoch{}_test.npy".format(exp_id,model_id,epoch)):
        norm = compNorm(featPath)
        np.save("../results/{}/norm_{}_epoch{}_test.npy".format(exp_id,model_id,epoch),norm)
    else:
        norm = np.load("../results/{}/norm_{}_epoch{}_test.npy".format(exp_id,model_id,epoch))

    norm = torch.tensor(norm)
    norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    norm = norm/(norm_max+0.00001)

    if debug:
        torch.manual_seed(1)
        attMaps = attMaps[:640]
        norm = norm[:640]
        inds = torch.randint(len(norm),size=(imgNb,))

    attMaps = attMaps.float()/255
    norm = norm.float()

    # avgNorm = N 1 3
    attMaps = attMaps/(attMaps.sum(dim=(2,3),keepdim=True)+0.0001)
    avgNorm = (attMaps*norm).sum(dim=(2,3))

    if debug:
        for i,ind in enumerate(inds):
            avgNorm_sample = avgNorm[ind]
            plt.figure()
            plt.bar(np.arange(avgNorm.shape[1]),avgNorm_sample.numpy())
            plt.savefig("../vis/{}/importance_{}_ind{}_{}.png".format(exp_id,model_id,i+1,ind))

    if plotStds:
        avgNorm_mean,avgNorm_std = avgNorm.mean(dim=0),avgNorm.std(dim=0)
        plt.figure()
        plt.bar(np.arange(avgNorm.shape[1]),avgNorm_mean.numpy())
        plt.errorbar(np.arange(avgNorm.shape[1]),avgNorm_mean.numpy(),avgNorm_std,fmt="*",color="black")
        plt.savefig("../vis/{}/importance_{}.png".format(exp_id,model_id))
    elif plotConfInter:
        meanList,lowList,highList = [],[],[]
        for i in range(avgNorm.shape[1]):
            mean,low,high = mean_confidence_interval(avgNorm[:,i])
            meanList.append(mean)
            lowList.append(low)
            highList.append(high)
        errors = np.concatenate((np.array(lowList)[np.newaxis],np.array(highList)[np.newaxis]),axis=0)
        plt.figure()
        plt.bar(np.arange(len(meanList)),meanList)
        plt.errorbar(np.arange(len(meanList)),meanList,errors,fmt="*",color="black")
        plt.savefig("../vis/{}/importance_{}.png".format(exp_id,model_id))
    else:
        avgNorm_mean = avgNorm.mean(dim=0)
        plt.figure()
        plt.bar(np.arange(avgNorm.shape[1]),avgNorm_mean.numpy())
        plt.savefig("../vis/{}/importance_{}.png".format(exp_id,model_id))

def repVSGlob(rep_vs_glob):

    weights = np.load(rep_vs_glob)

    vec_weig = np.abs(weights[:,:-2048]).mean(axis=1)
    glob_weig = np.abs(weights[:,-2048:]).mean(axis=1)

    plt.figure()
    plt.bar(np.arange(len(vec_weig)),vec_weig,width=0.45,color="blue")
    plt.bar(np.arange(len(glob_weig))+0.5,glob_weig,width=0.45,color="yellow")
    plt.savefig("../vis/rep_vs_glob.png")

def effPlot(red=True):

    if red:
        idRoot_dic = {"clusRed":"BR-NPA","bilRed":"B-CNN"}
    else:
        idRoot_dic = {"clusMast":"BR-NPA","bilMast":"B-CNN"}

    idRoot_list = list(idRoot_dic.keys())
    resSize = ["18","34","50","101","152"]

    latency_csv = np.genfromtxt("latency.csv",delimiter=",",dtype=str)[1:]
    latency_dic = {row[0]+"-"+row[1].replace("resnet",""):(float(row[3]),float(row[4])) for row in latency_csv}

    memory_csv = np.genfromtxt("memory.csv",delimiter=",",dtype=str)[1:]
    memory_dic = {row[0]+"-"+row[1].replace("resnet",""):(int(row[2])) for row in memory_csv}

    markerList = ["*","o"]

    for j,idRoot in enumerate(idRoot_list):
        perfList,latList,memList = [],[],[]

        model_ids = [idRoot+"-"+resSize[i] for i in range(len(resSize))]

        paths = []

        for id in model_ids:
            perfList.append(np.genfromtxt(glob.glob("../results/CUB10/model{}_epoch*test*".format(id))[0],delimiter=",")[-1,0])
            latList.append(latency_dic[id][0])
            memList.append(memory_dic[id])

        plt.figure(1)
        plt.plot(latList,perfList,"-{}".format(markerList[j]),label=idRoot_dic[idRoot],markersize=14)

        plt.figure(2)
        plt.plot(memList,perfList,"-{}".format(markerList[j]),label=idRoot_dic[idRoot],markersize=14)

    #params = {'axes.labelsize': 20,'axes.titlesize':20,\
    #          'legend.fontsize': 20, 'xtick.labelsize': 20,\
    #          'ytick.labelsize': 20}
    #matplotlib.rcParams.update(params)
    size = 20

    if red:
        lat_xticks = np.arange(0.1,0.5,0.1)
        mem_xticks = np.arange(50,300,50)
    else:
        lat_xticks = np.arange(0.1,1,0.2)
        mem_xticks = np.arange(50,600,100)

    fig = plt.figure(1)
    fig.set_size_inches(5, 5.5)
    plt.legend(fontsize=size)
    plt.yticks(np.arange(0.78,0.87,0.01),np.arange(78,87,1),fontsize=size)
    plt.xticks(lat_xticks,[round(f,2) for f in lat_xticks],fontsize=size)
    plt.xlabel("Latency (s)",fontsize=size)
    plt.ylabel("Accuracy",fontsize=size)
    plt.tight_layout()
    plt.savefig("../vis/CUB10/eff_latency_red={}.png".format(red))

    fig = plt.figure(2)
    fig.set_size_inches(5, 5.5)
    plt.legend(fontsize=size)
    plt.yticks(np.arange(0.78,0.87,0.01),np.arange(78,87,1),fontsize=size)
    plt.xticks(mem_xticks,mem_xticks,fontsize=size)
    plt.xlabel("Maximum batch size",fontsize=size)
    plt.ylabel("Accuracy",fontsize=size)
    plt.tight_layout()
    plt.savefig("../vis/CUB10/eff_memory_red={}.png".format(red))

def attMapsNbPlot():

    plt.figure()

    pattlist = ["../results/CUB10/modelN*test*csv","../results/CUB10/modelbilN*_3_epoch*test*csv"]
    labList = ["BR-CNN","B-CNN"]
    rootList = ["modelN","modelbilN"]

    for i,patt in enumerate(pattlist):
        testPaths = glob.glob(patt)
        perfList,NList = [],[]

        for path in sorted(testPaths,key=lambda x:utils.findNumbers(os.path.basename(x))):
            perfList.append(float(np.genfromtxt(path,delimiter=",")[1,0]))
            NList.append(os.path.basename(path).split(rootList[i])[1].split("_")[0])

        plt.plot(NList,perfList,label=labList[i],marker="o")

    plt.legend()
    plt.ylabel("Test accuracy")
    plt.xlabel("Attention map number (N)")
    plt.savefig("../vis/CUB10/N_acc.png")

def gradExp():

    plt.figure()

    colorDic = {"bilRed":"blue","clusRed":"orange"}
    lineDic = {"worst":":","median":"--","best":"-"}

    for model in ["bilRed","clusRed"]:

        perfDic = getPerfs(model)

        for run in ["worst","median","best"]:
            ratioList = []
            gradPaths = sorted(glob.glob("../results/CUB10/{}*allGrads*{}*".format(model,run)),key=lambda x:utils.findNumbers(os.path.basename(x)))

            snrPath ="../results/CUB10/{}_allSNR_{}.npy".format(model,run)
            if len(gradPaths) > 0 or os.path.exists(snrPath):
                if not os.path.exists(snrPath):
                    for gradPath in gradPaths:
                        print(gradPath)
                        model_id = os.path.basename(gradPath).split("_")[0]
                        run = os.path.basename(gradPath).split("_")[2].replace("HypParams","")

                        grads = torch.load(gradPath,map_location="cpu")
                        grads = grads.view(grads.shape[0],-1)

                        mean = grads.mean(dim=0)
                        std = grads.std(dim=0)
                        snr = mean/(std*std)
                        snr = snr.mean(dim=0)

                        ratioList.append(snr)

                    np.save(snrPath,np.array(ratioList))
                else:
                    ratioList = np.load(snrPath)

                plt.plot(ratioList,lineDic[run],label=model+"_"+run+":"+str(round(perfDic[run],2)),color=colorDic[model])

    plt.legend()
    plt.savefig("../vis/CUB10/gradExp.png")

def getTrialList(curr,optuna_trial_nb):
    curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1')
    query_res = curr.fetchall()

    query_res = list(filter(lambda x:not x[1] is None,query_res))

    trialIds = [id_value[0] for id_value in query_res]
    values = [id_value[1] for id_value in query_res]

    trialIds = trialIds[:optuna_trial_nb]
    values = values[:optuna_trial_nb]
    return trialIds,values

def getPerfs(model):

    config = configparser.ConfigParser()
    config.read('../models/CUB10/{}.ini'.format(model))
    optuna_trial_nb = int(config["default"]["optuna_trial_nb"])

    con = sqlite3.connect("../results/CUB10/{}_hypSearch.db".format(model))
    curr = con.cursor()

    trials,values = getTrialList(curr,optuna_trial_nb)

    best = np.array(values).max()
    worst = np.array(list(filter(lambda x:x>0.1,values))).min()
    median = np.median(np.array(values))

    return {"worst":worst,"median":median,"best":best}

def gradExp_test():

    pts_clusRed = np.genfromtxt("../results/CUB10/snr_clusRed.csv",delimiter=",")[1:,1:]
    pts_bilRed = np.genfromtxt("../results/CUB10/snr_bilRed.csv",delimiter=",")[1:,1:]

    pts_clusRed = pts_clusRed[~np.isnan(pts_clusRed[:,0])]
    pts_bilRed = pts_bilRed[~np.isnan(pts_bilRed[:,0])]

    plt.figure()
    plt.scatter(pts_clusRed[:,1],pts_clusRed[:,0],label="clusRed",color="orange")
    plt.scatter(pts_bilRed[:,1],pts_bilRed[:,0],label="bilRed",color="blue")
    plt.ylim(0,60)
    plt.xlim(0.4,0.9)
    plt.legend()
    plt.savefig("../vis/CUB10/gradExpTest.png")

    plt.figure()

    pts_clusRed = pts_clusRed[pts_clusRed[:,1] > 0.6]
    pts_bilRed = pts_bilRed[pts_bilRed[:,1] > 0.6]

    clusAcc = pts_clusRed[:,1]
    bilAcc = pts_bilRed[:,1]

    best,median,worst = clusAcc.max(),np.median(clusAcc),clusAcc.min()
    snr_best = pts_clusRed[:,0][clusAcc==best][0]
    #snr_median = pts_clusRed[:,0][clusAcc==median][0]
    snr_worst = pts_clusRed[:,0][clusAcc==worst][0]
    #plt.bar(np.arange(3),[snr_worst,snr_median,snr_best],width=0.2,label="clusRed")
    plt.bar(np.arange(2),[snr_worst,snr_best],width=0.2,label="clusRed")

    best,median,worst = bilAcc.max(),np.median(bilAcc),bilAcc.min()
    snr_best = pts_bilRed[:,0][bilAcc==best][0]
    #snr_median = pts_bilRed[:,0][bilAcc==median][0]
    snr_worst = pts_bilRed[:,0][bilAcc==worst][0]
    #plt.bar(np.arange(3)+.1,[snr_worst,snr_median,snr_best],width=0.2,label="bilAcc")
    plt.bar(np.arange(2)+.1,[snr_worst,snr_best],width=0.2,label="bilAcc")


    plt.legend()

    plt.savefig("../vis/CUB10/gradExpTest.png")

def gradExp2():

    convs = ["Conv1","Conv2","Conv3"]

    colors = {"clusRed":{"Conv1":"yellow","Conv2":"orange","Conv3":"red"},
                "bilRed":{"Conv1":"cyan","Conv2":"blue","Conv3":"violet"}}

    epochs = len(glob.glob("../results/CUB10/clusRed_allGradsConv1_bestHypParams_epoch*.th"))


    for conv in convs:
        plt.figure()

        gradNorm,gradNorm_sm = getAllNorm("clusRed",conv,epochs)
        x = np.arange(len(gradNorm))/len(gradNorm)
        plt.plot(x,gradNorm_sm,color="orange",label="BR-CNN")
        plt.plot(x,gradNorm,color="orange",alpha=0.5)

        gradNorm,gradNorm_sm = getAllNorm("bilRed",conv,epochs)
        x = np.arange(len(gradNorm))/len(gradNorm)
        plt.plot(x,gradNorm_sm,color="blue",label="B-CNN")
        plt.plot(x,gradNorm,color="blue",alpha=0.5)

        plt.xlabel("Training epochs")
        ticks = np.arange(epochs//2+1)
        plt.xticks(ticks/(epochs//2),2*ticks,rotation=25)
        plt.ylabel("{} gradient norm".format(conv))

        plt.legend()
        plt.savefig("../vis/CUB10/gradExp2_{}.png".format(conv))

def getAllNorm(model,conv,epochs):

    gradNorm = None
    for epoch in range(1,epochs+1):
        gradNorm_epoch = torch.load("../results/CUB10/{}_allGrads{}_bestHypParams_epoch{}.th".format(model,conv,epoch),map_location="cpu")[:-1]
        gradNorm = cat(gradNorm,gradNorm_epoch)

    gradNorm_sm = smooth(gradNorm.numpy())

    return gradNorm,gradNorm_sm

def smooth(x):
    return savgol_filter(x, 51, 3)

def cat(all,new):
    if all is None:
        all = new
    else:
        all = torch.cat((all,new),dim=0)
    return all

def get_metrics_with_IB():
    return ["Del","DelCorr","Add","AddCorr","AD","ADD","IIC","Lift"]

def getResPaths(exp_id,metric,img_bckgr):
    if img_bckgr and metric in get_metrics_with_IB():
        paths = sorted(glob.glob("../results/{}/attMetr{}-IB_*.npy".format(exp_id,metric)))
    else:
        paths = sorted(glob.glob("../results/{}/attMetr{}_*.npy".format(exp_id,metric)))
        paths = list(filter(lambda x:os.path.basename(x).find("-IB") ==-1,paths))

    paths = removeOldFiles(paths)

    return paths 

def removeOldFiles(paths):
    paths = list(filter(lambda x:os.path.basename(x).find("noise") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("imgBG") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("maskFeat") ==-1,paths))
    return paths

def getModelId(path,metric,img_bckgr):
    suff = "-IB" if img_bckgr else ""
    model_id = os.path.basename(path).split("attMetr{}{}_".format(metric,suff))[1].split(".npy")[0]
    return model_id

def attMetrics(exp_id,metric="Del",ignore_model=False,img_bckgr=False):

    suff = metric

    paths = getResPaths(exp_id,metric,img_bckgr)

    resDic = {}
    resDic_pop = {}

    if ignore_model:
        _,modelToIgn = getIndsToUse(paths,metric)
    else:
        modelToIgn = []

    if metric in ["Del","Add"]:
        for path in paths:

            model_id = getModelId(path,metric,img_bckgr)
            
            if model_id not in modelToIgn:
                pairs = np.load(path,allow_pickle=True)

                allAuC = []

                for i in range(len(pairs)):

                    pairs_i = np.array(pairs[i])

                    if metric == "Add":
                        pairs_i[:,0] = 1-pairs_i[:,0]/pairs_i[:,0].max()
                    else:
                        pairs_i[:,0] = (pairs_i[:,0]-pairs_i[:,0].min())/(pairs_i[:,0].max()-pairs_i[:,0].min())
                        pairs_i[:,0] = 1-pairs_i[:,0]

                    auc = np.trapz(pairs_i[:,1],pairs_i[:,0])
                    allAuC.append(auc)
            
                resDic_pop[model_id] = np.array(allAuC)
                resDic[model_id] = resDic_pop[model_id].mean()
    elif metric == "Lift":

        for path in paths:

            model_id = getModelId(path,metric,img_bckgr)
            
            if model_id not in modelToIgn:
                
                scores = np.load(path,allow_pickle=True)[:,0] 
                scores_mask = np.load(path.replace("Lift","LiftMask"),allow_pickle=True)[:,0] 
                scores_invmask = np.load(path.replace("Lift","LiftInvMask"),allow_pickle=True)[:,0]  

                iic = 100*(scores<scores_mask).mean(keepdims=True)
                diff = (scores-scores_mask)
                ad = 100*diff*(diff>0)/scores
                add = 100*(scores-scores_invmask)/scores

                resDic[model_id] = str(iic.item())+","+str(ad.mean())+","+str(add.mean())
                resDic_pop[model_id] = {"IIC":iic,"AD":ad,"ADD":add}

    else:
        for path in paths:
            
            path = path.replace("-IB","")
            model_id = getModelId(path,metric,img_bckgr=False)
            
            if model_id not in modelToIgn:
                sparsity_list = 1/np.load(path,allow_pickle=True)

            resDic_pop[model_id] = np.array(sparsity_list)
            resDic[model_id] = resDic_pop[model_id].mean() 

    suff += "-IB" if img_bckgr and metric != "Spars" else ""
    csv = "\n".join(["{},{}".format(key,resDic[key]) for key in resDic])
    with open("../results/{}/attMetrics_{}.csv".format(exp_id,suff),"w") as file:
        print(csv,file=file)

    if metric == "Lift":
        for metric in ["IIC","AD","ADD"]:
            csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key][metric].astype("str"))) for key in resDic_pop])
            suff = "-IB" if img_bckgr else ""
            with open("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),"w") as file:
                print(csv,file=file)

    else:
        csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key].astype("str"))) for key in resDic_pop])
        with open("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),"w") as file:
            print(csv,file=file)

def getIndsToUse(paths,metric):
    
    modelToIgn = []

    model_targ_ind = 0

    while model_targ_ind < len(paths) and not os.path.exists(paths[model_targ_ind].replace("Add","Targ").replace("Del","Targ")):
        model_targ_ind += 1

    if model_targ_ind == len(paths):
        use_all_inds = True
    else:
        use_all_inds = False 
        targs = np.load(paths[model_targ_ind],allow_pickle=True)
        
        indsToUseBool = np.array([True for _ in range(len(targs))])
        indsToUseDic = {}

    for path in paths:
        
        model_id = os.path.basename(path).split("attMetr{}_".format(metric))[1].split(".npy")[0]
        
        model_id_nosuff = model_id.replace("-max","").replace("-onlyfirst","").replace("-fewsteps","")

        predPath = path.replace(metric,"Preds").replace(model_id,model_id_nosuff)

        if not os.path.exists(predPath):
            predPath = path.replace(metric,"PredsAdd").replace(model_id,model_id_nosuff)

        if os.path.exists(predPath) and not use_all_inds:
            preds = np.load(predPath,allow_pickle=True)

            if preds.shape != targs.shape:
                inds = []
                for i in range(len(preds)):
                    if i % 2 == 0:
                        inds.append(i)

                preds = preds[inds]
            
            indsToUseDic[model_id] = np.argwhere(preds==targs)
            indsToUseBool = indsToUseBool*(preds==targs)
  
        else:
            modelToIgn.append(model_id)
            print("no predpath",predPath)

    if use_all_inds:
        indsToUse = None
    else:
        indsToUse =  np.argwhere(indsToUseBool)
        
    return indsToUse,modelToIgn 

def get_id_to_label():
    return {"bilRed":"B-CNN",
            "bilRed_1map":"B-CNN (1 map)",
            "clus_masterClusRed":"BR-NPA",
            "clus_mast":"BR-NPA",
            "noneRed":"AM",
            "protopnet":"ProtoPNet",
            "prototree":"ProtoTree",
            "noneRed-gradcam":"Grad-CAM",
            "noneRed-gradcam_pp":"Grad-CAM++",
            "noneRed-score_map":"Score-CAM",
            "noneRed-ablation_cam":"Ablation-CAM",
            "noneRed-rise":"RISE",
            "noneRed_smallimg-varGrad":"VarGrad",
            "noneRed_smallimg-smoothGrad":"SmoothGrad",
            "noneRed_smallimg-guided":"GuidedBP",
            "interbyparts":"InterByParts",
            "abn":"ABN"}

def get_label_to_id():
    id_to_label = get_id_to_label()
    return {id_to_label[id]:id for id in id_to_label}

def get_is_post_hoc():
    return {"bilRed":False,
            "bilRed_1map":False,
            "clus_masterClusRed":False,
            "clus_mast":False,
            "noneRed":True,
            "protopnet":False,
            "prototree":False,
            "noneRed-gradcam":True,
            "noneRed-gradcam_pp":True,
            "noneRed-score_map":True,
            "noneRed-ablation_cam":True,
            "noneRed-rise":True,
            "noneRed_smallimg-varGrad":True,
            "noneRed_smallimg-smoothGrad":True,
            "noneRed_smallimg-guided":True,
            "interbyparts":False,
            "abn":False}

def ttest_attMetr(exp_id,metric="del",img_bckgr=False):

    id_to_label = get_id_to_label()

    suff = metric
    suff += "-IB" if img_bckgr and metric in get_metrics_with_IB() else ""

    arr = np.genfromtxt("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),dtype=str,delimiter=",")

    metric_to_max = []
    what_is_best = get_what_is_best()
    for metric_ in what_is_best:
        if what_is_best[metric_] == "max":
            metric_to_max.append(metric_)

    arr = best_to_worst(arr,ascending=metric in metric_to_max)

    model_ids = arr[:,0]

    labels = [id_to_label[model_id] for model_id in model_ids]

    res_mat = arr[:,1:].astype("float")

    if metric == "add":
        rnd_nb = 2
    elif metric == "del":
        rnd_nb = 3 
    else:
        rnd_nb = 1

    perfs = [(str(round(mean,rnd_nb)),str(round(std,rnd_nb))) for (mean,std) in zip(res_mat.mean(axis=1),res_mat.std(axis=1))]

    p_val_mat = np.zeros((len(res_mat),len(res_mat)))
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            p_val_mat[i,j] = scipy.stats.ttest_ind(res_mat[i],res_mat[j],equal_var=False)[1]

    p_val_mat = (p_val_mat<0.05)

    res_mat_mean = res_mat.mean(axis=1)

    diff_mat = np.abs(res_mat_mean[np.newaxis]-res_mat_mean[:,np.newaxis])
    
    diff_mat_norm = (diff_mat-diff_mat.min())/(diff_mat.max()-diff_mat.min())

    cmap = plt.get_cmap('plasma')

    fig = plt.figure()

    ax = fig.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.imshow(p_val_mat*0,cmap="Greys")
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            if i <= j:
                rad = 0.3 if p_val_mat[i,j] else 0.1
                circle = plt.Circle((i, j), rad, color=cmap(diff_mat_norm[i,j]))
                ax.add_patch(circle)

    plt.yticks(np.arange(len(res_mat)),labels)
    plt.xticks(np.arange(len(res_mat)),["" for _ in range(len(res_mat))])
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(diff_mat.min(),diff_mat.max()),cmap=cmap))
    for i in range(len(res_mat)):
        plt.text(i-0.2,i-0.4,labels[i],rotation=45,ha="left")
    plt.tight_layout()
    plt.savefig("../vis/{}/ttest_{}_attmetr.png".format(exp_id,suff))

def best_to_worst(arr,ascending=True):

    if not ascending:
        key = lambda x:-x[1:].astype("float").mean()
    else:
        key = lambda x:x[1:].astype("float").mean()

    arr = np.array(sorted(arr,key=key))

    return arr

def attMetricsStats(exp_id):

    statsPaths = sorted(glob.glob("../results/{}/attMetrStatsDel_*.npy".format(exp_id)))

    for path in statsPaths:

        model_id = os.path.basename(path).split("attMetrStatsDel_")[1].split(".npy")[0]
        
        statsTriplet = np.load(path,allow_pickle=True)

        for i in range(len(statsTriplet)):
        
            plt.figure()
            plt.plot(np.array(statsTriplet[i])[:,0],label=("min"))
            plt.plot(np.array(statsTriplet[i])[:,1],label=("mean"))   
            plt.plot(np.array(statsTriplet[i])[:,2],label=("max"))   
            plt.legend() 
            plt.savefig("../vis/{}/attMetrStatsDel_{}_{}.png".format(exp_id,model_id,i))
            plt.close()                 

def loadArr(exp_id,metric,best,img_bckgr):
    
    suff = "-IB" if img_bckgr and metric in get_metrics_with_IB() else ""
    arr = np.genfromtxt("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),dtype=str,delimiter=",") 
    arr_f = arr[:,1:].astype("float")
    if best == "max":
        best = arr_f.mean(axis=1).max()
    else:
        best = arr_f.mean(axis=1).min()
    return arr,arr_f,best

def find_perf(i,metric,arr_dic,arr_f_dic,best_dic):
    ind = np.argwhere(arr_dic[metric][:,0] == arr_dic["Del"][i,0])[0,0]
    mean = arr_f_dic[metric][ind].mean()

    if metric in ["Add","DelCorr","AddCorr"]:
        mean_rnd,std_rnd = round(mean,3),round(arr_f_dic[metric][ind].std(),3)
    elif metric in ["IIC"]:
        mean_rnd,std_rnd = round(mean*0.01,3),0
    elif metric in ["AD","ADD"]:
        mean_rnd,std_rnd = round(mean*0.01,3),round(arr_f_dic[metric][ind].std()*0.01,3)
    else:
        raise ValueError("Unkown metric",metric)

    is_best = (mean==best_dic[metric])
    return mean_rnd,std_rnd,is_best

def processRow(csv,row,with_std,metric_list,full=True,postHoc=False,postHocInd=None,nbPostHoc=None):
    
    for metric in metric_list:

        if metric != "Acc" or not postHoc:
            csv += latex_fmt(row[metric],row[metric+"_std"],row["is_best_"+metric] == "True",\
                        with_std=with_std and metric != "IIC",with_start_char=(metric!="Accuracy"))
        else:
            if full:
                if postHocInd > 0:
                    csv += "&"
                else:
                    acc_text = latex_fmt(row["Acc"],row["Acc_std"],row["is_best_Acc"] == "True",with_std=with_std and metric != "IIC",with_start_char=False)
                    csv += "&\multirow{"+str(nbPostHoc)+"}{*}{$"+acc_text+"$}"

    csv += "\\\\ \n"
    return csv 

def formatRow(res_list,i):
    row = res_list[i]
    row = {key:str(row[key]) for key in row}
    return row

def get_what_is_best():
    return {"Del":"min","Add":"max","DelCorr":"max","AddCorr":"max","IIC":"max","AD":"min","ADD":"max","Spars":"max","Time":"min"}
def get_metric_label():
    return {"Del":"DAUC","Add":"IAUC","Spars":"Sparsity","IIC":"IIC","AD":"AD","ADD":"ADD","DelCorr":"DC","AddCorr":"IC","Time":"Time","Acc":"Accuracy"}
    
def latex_table_figure(exp_id,full=False,with_std=False,img_bckgr=False,suppMet=False):

    arr_dic = {}
    arr_f_dic = {}
    best_dic = {}

    what_is_best = get_what_is_best()

    if not suppMet:
        what_is_best.pop("Spars")
        what_is_best.pop("Time")

    metric_to_label = get_metric_label()
    metric_list = list(what_is_best.keys())

    for metric in metric_list:
        arr_dic[metric],arr_f_dic[metric],best_dic[metric] = loadArr(exp_id,metric,best=what_is_best[metric],img_bckgr=img_bckgr)

    id_to_label = get_id_to_label()

    res_list = []

    for i in range(len(arr_dic["Del"])):
       
        id = id_to_label[arr_dic["Del"][i,0]]

        mean = arr_f_dic["Del"][i].mean()
        dele,dele_std = round(mean,4),round(arr_f_dic["Del"][i].std(),3)
        dele_full_precision = mean
        is_best_dele = (mean==best_dic["Del"])

        res_dic = {"id":id,"Del_full_precision":dele_full_precision,
                    "Del":dele,  "Del_std":dele_std,  "is_best_Del":is_best_dele,}

        for metric in metric_list:
            if metric != "Del":
                mean_rnd,std_rnd,is_best = find_perf(i,metric,arr_dic,arr_f_dic,best_dic)
                res_dic.update({metric:mean_rnd,metric+"_std":std_rnd,"is_best_"+metric:is_best})

        res_list.append(res_dic)

    if full:
        res_list = addAccuracy(res_list,exp_id)
        metric_list.insert(8,"Acc")
        what_is_best["Acc"] = "max"

    res_list = sorted(res_list,key=lambda x:-x["Del_full_precision"])

    csv = "Model&Viz. Method&"+"&".join([metric_to_label[metric] for metric in metric_list])+"\\\\ \n"

    is_post_hoc = get_is_post_hoc()
    label_to_id = get_label_to_id()

    nbPostHoc = 0
    #Counting post hoc
    for i in range(len(res_list)):
        row = formatRow(res_list,i) 
        if is_post_hoc[label_to_id[row["id"]]]:
            nbPostHoc += 1 

    postHocInd =0
    #Adding post hoc
    for i in range(len(res_list)):
        row = formatRow(res_list,i) 
        if is_post_hoc[label_to_id[row["id"]]]:
            if postHocInd == 0:
                csv += "\multirow{"+str(postHocInd)+"}{*}{CNN}&"  
            else:
                csv += "&"
            csv += row["id"] 

            csv = processRow(csv,row,with_std,metric_list,full=full,postHoc=True,postHocInd=postHocInd,nbPostHoc=nbPostHoc)

            postHocInd += 1 

    csv += "\\hline \n"

    #Adding att models
    for i in range(len(res_list)):
        row = formatRow(res_list,i)
        if not is_post_hoc[label_to_id[row["id"]]]:
            csv += row["id"]+"&-"   
            csv = processRow(csv,row,with_std,metric_list)

    suff = "-IB" if img_bckgr else ""

    with open("../results/{}/attMetr_latex_table{}.csv".format(exp_id,suff),"w") as text:
        print(csv,file=text)

    for metric in metric_list:

        res_list = sorted(res_list,key=lambda x:x[metric])

        plt.figure()
        plt.errorbar(np.arange(len(res_list)),[row[metric] for row in res_list],[row[metric+"_std"] for row in res_list],color="darkblue",fmt='o')
        plt.bar(np.arange(len(res_list)),[row[metric] for row in res_list],0.9,color="lightblue",linewidth=0.5,edgecolor="darkblue")
        plt.ylabel(metric_to_label[metric])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(len(res_list)),[row["id"] for row in res_list],rotation=45,ha="right")
        plt.tight_layout()
        plt.savefig("../vis/{}/attMetr_{}{}.png".format(exp_id,metric_to_label[metric],suff))

def latex_fmt(mean,std,is_best,with_std=False,with_start_char=True):

    if with_std:
        if is_best:
            metric_value = "\mathbf{"+str(mean)+"\pm"+str(std)+"}"
        else:
            metric_value = ""+str(mean)+"\pm"+str(std)
    else:
        if is_best:
            metric_value = "\mathbf{"+str(mean)+" }"
        else:
            metric_value = ""+str(mean)

    if with_start_char:
        metric_value = "&$" + metric_value + "$"
    
    return metric_value

def reverseLabDic(id_to_label,exp_id):

    label_to_id = {}

    for id in id_to_label:
        label = id_to_label[id]

        if label == "BR-NPA":
            if exp_id == "CUB10":
                id = "clus_masterClusRed"
            else:
                id = "clus_mast"
        elif id.startswith("noneRed"):
            id = "noneRed"

        label_to_id[label] = id 
    
    return label_to_id

def addAccuracy(res_list,exp_id):
    res_list_with_acc = []
    id_to_label= get_id_to_label()

    label_to_id = reverseLabDic(id_to_label,exp_id)

    acc_list = np.genfromtxt("../results/{}/attMetrics_Acc.csv".format(exp_id),delimiter=",",dtype=str)
    acc_dic = {model_id:{"mean":mean,"std":std} for (model_id,mean,std) in acc_list}

    for row in res_list:

        model_id = label_to_id[row["id"]]
        
        accuracy = float(acc_dic[model_id]["mean"])
        accuracy_std = float(acc_dic[model_id]["std"])

        row["Acc"] = str(round(accuracy,1))
        row["Acc_std"] = str(round(accuracy_std,1))
        row["Acc_full_precision"] = accuracy

        res_list_with_acc.append(row)
    
    bestAcc = max([row["Acc_full_precision"] for row in res_list_with_acc])

    res_list_with_acc_and_best = []
    for row in res_list_with_acc:   
        row["is_best_Acc"] = (row["Acc_full_precision"]==bestAcc)
        res_list_with_acc_and_best.append(row)

    return res_list_with_acc_and_best

def accuracyPerVideo(exp_id,nbClass=16):

    pred_file_paths = glob.glob("../results/{}/*_epoch*_test.csv".format(exp_id))
    pred_file_paths = list(filter(lambda x:os.path.basename(x).find("model") == -1,pred_file_paths))

    all_accuracy_list_csv = ""
    all_accuracy_mean_csv = ""

    for path in pred_file_paths:

        csv = np.genfromtxt(path,delimiter=",",dtype=str)[1:]

        vidNames = list(map(lambda x:os.path.basename(x).split("_")[0],csv[:,0]))

        vid_corr_dic = {}
        vid_frameNb_dic = {}

        predicted_class = csv[:,2:2+nbClass].astype(float).argmax(axis=1)
        gt_class = csv[:,1].astype(int)   
        correct = (predicted_class==gt_class)

        for i in range(len(csv)):
            if vidNames[i] in vid_corr_dic:
                vid_corr_dic[vidNames[i]] += 1*correct[i]
                vid_frameNb_dic[vidNames[i]] += 1
            else:
                vid_corr_dic[vidNames[i]] = 1*correct[i]
                vid_frameNb_dic[vidNames[i]] = 1
        
        vid_acc_dic = {vid:100*vid_corr_dic[vid]/vid_frameNb_dic[vid] for vid in vid_corr_dic}

        model_id = os.path.basename(path).split("_epoch")[0]
        
        acc_list = list(vid_acc_dic.values())

        all_accuracy_list_csv += model_id + ","+",".join(map(str,acc_list))+"\n"
        all_accuracy_mean_csv += model_id + ","+str(np.mean(acc_list))+","+str(np.std(acc_list))+"\n"

    with open("../results/{}/attMetrics_Acc_pop.csv".format(exp_id),"w") as file:
        print(all_accuracy_list_csv,file=file)

    with open("../results/{}/attMetrics_Acc.csv".format(exp_id),"w") as file:
        print(all_accuracy_mean_csv,file=file)

def reformatAttScoreArray(attScores,pairs):

    k = attScores.shape[1]//pairs.shape[1]

    if k > 1:
        newAttScores = []
        for i in range(pairs.shape[0]):
            scoreList = []
            attScores_i = attScores[i].astype("float64")

            for j in range(pairs.shape[1]):
                scores,indices = torch.topk(torch.tensor(attScores_i),k=k)
                scoreList.append(scores.mean())
                attScores_i[indices] = -1

            newAttScores.append(scoreList) 
    else:
        newAttScores = attScores

    return np.array(newAttScores)

def correlation(points):
    corrList = []
    for i in range(len(points)):
        points_i = points[i]
        corrList.append(np.corrcoef(points_i,rowvar=False)[0,1])
    return np.array(corrList)

def attCorrelation(exp_id,img_bckgr=False):

    if not os.path.exists("../vis/{}/correlation/".format(exp_id)):
        os.makedirs("../vis/{}/correlation/".format(exp_id))

    suff = "-IB" if img_bckgr else ""

    for metric in ["Del","Add"]:
        csv_res = []
        csv_res_pop = []

        paths = getResPaths(exp_id,metric,img_bckgr)

        for path in paths:
            points = np.load(path,allow_pickle=True).astype("float")
            
            path_att_score = path.replace("attMetr{}{}".format(metric,suff),"attMetrAttScore")
            if os.path.exists(path_att_score):
                model_id = os.path.basename(path).split(metric+suff+"_")[1].replace(".npy","")    

                attScores = np.load(path_att_score,allow_pickle=True)
                oldShape = attScores.shape
                attScores = reformatAttScoreArray(attScores,points)
                if oldShape != attScores.shape:
                    np.save(path_att_score,attScores)

                points[:,:,0] = attScores

                points_diff = points.copy()
                if metric == "Del":
                    points_diff[:,:-1,1] = points[:,:-1,1]-points[:,1:,1]
                else:
                    points_diff[:,:-1,1] = points[:,1:,1]-points[:,:-1,1]
                points_diff = points_diff[:,:-1]

                all_corr = correlation(points_diff)

                corr = np.nanmean(all_corr)
  
                csv_res += "{},{}\n".format(model_id,corr) 
                csv_res_pop += "{},".format(model_id)+",".join([str(corr) for corr in all_corr])+"\n"

        suff = "-IB" if img_bckgr else ""

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}.csv","w") as f:
            f.writelines(csv_res)

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}_pop.csv","w") as f:
            f.writelines(csv_res_pop)

def attTime(exp_id):
    metric = "Time"

    csv_res = []
    csv_res_pop = []
    paths = sorted(glob.glob("../results/{}/attMetr{}_*.npy".format(exp_id,metric)))
    paths = list(filter(lambda x:os.path.basename(x).find("noise") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("imgBG") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("maskFeat") ==-1,paths))

    for path in paths:
        model_id = os.path.basename(path).split(metric+"_")[1].replace(".npy","")    
        all_times = np.load(path,allow_pickle=True)
        time = all_times.mean()
    
        csv_res += "{},{}\n".format(model_id,time) 
        csv_res_pop += "{},".format(model_id)+",".join([str(corr) for corr in all_times])+"\n"

    with open(f"../results/{exp_id}/attMetrics_{metric}.csv","w") as f:
        f.writelines(csv_res)

    with open(f"../results/{exp_id}/attMetrics_{metric}_pop.csv","w") as f:
        f.writelines(csv_res_pop)


def normalize_metrics(values,metric):
    if metric in ["DelCorr","AddCorr"]:
        values = (values + 1)*0.5
    elif metric in ["AD","ADD","IIC"]:
        values = values*0.01 
    elif metric not in ["Del","Add"]:
        raise ValueError("Can't normalize",metric)
    
    return values 

def loadPerf(exp_id,metric,pop=True,img_bckgr=False,norm=False,reverse_met_to_min=False):

    suff = "-IB" if img_bckgr else ""

    if pop:
        perfs = np.genfromtxt("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),delimiter=",",dtype=str)
    else:
        if metric in ["ADD","AD","IIC"]:
            perfs = np.genfromtxt("../results/{}/attMetrics_Lift{}.csv".format(exp_id,suff),delimiter=",",dtype=str)
        
            if metric == "IIC":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,1:2]),axis=1)
            elif metric == "AD":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,2:3]),axis=1)
            elif metric == "ADD":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,3:4]),axis=1) 
        else:
            perfs = np.genfromtxt("../results/{}/attMetrics_{}{}.csv".format(exp_id,metric,suff),delimiter=",",dtype=str)
    
    if norm:
        perfs_norm = normalize_metrics(perfs[:,1:].astype("float"),metric)
        perfs = np.concatenate((perfs[:,0:1],perfs_norm.astype(str)),axis=1)

    if get_what_is_best()[metric] == "min":
        perfs[:,1:]= (-1*perfs[:,1:].astype("float")).astype("str")

    return perfs

def bar_viz(exp_id,img_bckgr=False):

    what_is_best = get_what_is_best()
    metric_list = list(what_is_best.keys())
    metric_list = list(filter(lambda metric:metric in ["Del","Add","AD","ADD","IIC"],metric_list))

    is_post_hoc = get_is_post_hoc()

    id_to_label = get_id_to_label()

    for metric in metric_list:

        perfs = loadPerf(exp_id,metric,img_bckgr=img_bckgr)

        perfs_att = []
        for perf in perfs:
            if not is_post_hoc[perf[0]]:
                perfs_att.append(perf)
        perfs_att = np.array(perfs_att)

        perfs_post = []
        for perf in perfs:
            if is_post_hoc[perf[0]]:
                perfs_post.append(perf)
        perfs_post = np.array(perfs_post)

        xmin,xmax = perfs[:,1].astype("float").min(),perfs[:,1].astype("float").max()

        fig = plt.figure(figsize=(5,2*len(perfs)))

        for i,model_perf in enumerate(perfs_att):
            subfig_mod = plt.subplot(len(perfs),1,i+1)
            fig.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=20)
            subfig_mod.hist(model_perf[1:].astype("float"),range=(xmin,xmax),color="orange",bins=10,label=id_to_label[model_perf[0]])
            plt.legend(prop={'size':20})

        for i,model_perf in enumerate(perfs_post):
            subfig_mod = plt.subplot(len(perfs),1,i+len(perfs_att)+1)
            fig.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=20)
            subfig_mod.hist(model_perf[1:].astype("float"),range=(xmin,xmax),color="blue",bins=10,label=id_to_label[model_perf[0]])
            plt.legend(prop={'size':20})

        plt.tight_layout()
        suff = "-IB" if img_bckgr else ""
        plt.savefig("../vis/{}/bar_{}{}.png".format(exp_id,metric,suff))

def kendallTauInd(metric_list,exp_id,img_bckgr,k,what_is_best,pop):
    rank_dic = {}
    kendall_tau_mat_k = np.zeros((len(metric_list),len(metric_list)))
    p_val_mat_k = np.zeros((len(metric_list),len(metric_list)))

    for i in range(len(metric_list)):

        metric = metric_list[i]

        if metric == "IIC":
            perfs = loadPerf(exp_id,metric,pop=False,img_bckgr=img_bckgr,reverse_met_to_min=True)
            rank = perfs[:,1].astype("float")
        else:
            perfs = loadPerf(exp_id,metric,pop=pop,img_bckgr=img_bckgr,reverse_met_to_min=True)
            rank = perfs[:,k+1].astype("float")
        rank_dic[metric] = rank 

    for i in range(len(metric_list)):
        for j in range(len(metric_list)):
            kendall_tau_mat_k[i,j],p_val_mat_k[i,j] = kendalltau(rank_dic[metric_list[i]],rank_dic[metric_list[j]])

    return kendall_tau_mat_k,p_val_mat_k

def computeKendallTauMat(metric_list,exp_id,pop,img_bckgr,what_is_best):
    
    kendall_tau_mat = np.zeros((len(metric_list),len(metric_list)))
    p_val_mat = np.zeros((len(metric_list),len(metric_list)))

    if pop:
        nbSamples = loadPerf(exp_id,"Del",pop=True,img_bckgr=img_bckgr).shape[0]

        for k in range(nbSamples):
            kendall_tau_mat_k,p_val_mat_k = kendallTauInd(metric_list,exp_id,img_bckgr,k,what_is_best,pop=True)
            kendall_tau_mat += kendall_tau_mat_k
            p_val_mat += p_val_mat_k

        kendall_tau_mat /= nbSamples
        p_val_mat /= nbSamples 

    else:
        kendall_tau_mat,p_val_mat = kendallTauInd(metric_list,exp_id,img_bckgr,0,what_is_best,pop=False)

    return kendall_tau_mat,p_val_mat

def ranking_similarities(exp_id,img_bckgr=False,pop=False):

    what_is_best = get_what_is_best()
    metric_list = ["Del","Add","DelCorr","AddCorr","AD","ADD","IIC"] 

    suff = "-IB" if img_bckgr else ""
    model_list = np.genfromtxt("../results/{}/attMetrics_Del{}.csv".format(exp_id,suff),delimiter=",",dtype=str)[:,0]

    kendall_tau_mat,p_val = computeKendallTauMat(metric_list,exp_id,pop,img_bckgr,what_is_best)

    cmap = plt.get_cmap("RdBu_r")

    fig = plt.figure()
    ax = fig.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    metric_to_label = get_metric_label()
    label_list = [metric_to_label[metric] for metric in metric_list]

    plt.imshow(kendall_tau_mat*0,cmap="Greys")
    #plt.imshow(p_val_mat*0)
    for i in range(len(metric_list)):
        for j in range(len(metric_list)):
            if i <= j:
                rad = 0.3
                circle = plt.Circle((i, j), rad, color=cmap((kendall_tau_mat[i,j]+1)*0.5))
                ax.add_patch(circle)

    fontSize = 17
    cbar = plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1,1),cmap=cmap))
    cbar.ax.tick_params(labelsize=fontSize)
    plt.xticks(range(len(metric_list)),label_list,rotation=45,fontsize=fontSize)
    plt.yticks(range(len(metric_list)),label_list,fontsize=fontSize)
    #fig, ax = plt.subplots()
    plt.tight_layout()
    suff += "_pop" if pop else ""
    print("../vis/{}/kendall_tau{}.png".format(exp_id,suff))
    plt.savefig("../vis/{}/kendall_tau{}.png".format(exp_id,suff))

#@numba.njit()
def rankDist(x,y):
    tau = kendalltau(x,y)[0]
    if tau == -1:
        tau = -0.999

    return -np.log(0.5*(tau+1))

def run_dimred_or_load(path,allFeat,dimred="umap"):
    #if dimred == "pca":
    #    dimRedFunc = PCA
    #    kwargs = {}
    if dimred == "umap":
        dimRedFunc = umap.UMAP
        kwargs = {}
    elif dimred == "tsne":
        dimRedFunc = TSNE
        kwargs = {"metric":rankDist,"learning_rate":100}
    else:
        raise ValueError("Unknown dimred {}".format(dimred))

    path = path.replace(".npy","_"+dimred+".npy")

    if not os.path.exists(path):
        allFeat = dimRedFunc(n_components=2,**kwargs).fit_transform(allFeat)
        np.save(path,allFeat)
    else:
        allFeat = np.load(path)

    return allFeat 

def dimred_metrics(exp_id,pop=False,dimred="umap",img_bckgr=False):

    metric_list = ["Del","Add","DelCorr","AddCorr","AD","ADD","IIC"] 

    if pop:
        metric_list.pop(-1)

    allPerfs = []
    sample_nb = loadPerf(exp_id,"Del",pop=pop).shape[1]-1
    for metric in metric_list:
        if pop:
                perfs = loadPerf(exp_id,metric,pop=True,img_bckgr=img_bckgr,norm=True,reverse_met_to_min=True)

                perfs = perfs[:,1:].transpose(1,0)
                allPerfs.append(perfs)
        else:
            perfs = loadPerf(exp_id,metric,pop=False,img_bckgr=img_bckgr,norm=True,reverse_met_to_min=True)
            allPerfs.append(perfs[np.newaxis,:,1])

    allPerfs = np.concatenate(allPerfs,axis=0).astype("float")

    path = f"../results/{exp_id}/metrics_dimred_pop{pop}_imgBckr{img_bckgr}.npy"
    allFeat = run_dimred_or_load(path,allPerfs,dimred)  
    
    metric_to_label = get_metric_label()
    cmap = plt.get_cmap("Set1")
    plt.figure()
    colorList = []
    for i,metric in enumerate(metric_list):
        start,end = i*sample_nb,(i+1)*sample_nb
        plt.scatter([allFeat[start,0]],[allFeat[start,1]],label=metric_to_label[metric],color=cmap(i*1.0/8))
        colorList.extend([cmap(i*1.0/8) for _ in range(sample_nb)])

    print(allFeat.shape,np.array(colorList).shape)
    feat_and_color = np.concatenate((allFeat,np.array(colorList)),axis=1)
    np.random.shuffle(feat_and_color)

    allFeat,colorList = feat_and_color[:,:2],feat_and_color[:,2:]

    fontSize = 15
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.scatter([allFeat[:,0]],[allFeat[:,1]],color=colorList)
    plt.legend(fontsize=fontSize)
    plt.savefig(f"../vis/{exp_id}/metrics_{dimred}_pop{pop}_imgBckgr{img_bckgr}.png")
    plt.close()

def vizRepr(exp_id):
    allFeat = []
    for metric in ["Lift","Del","Add"]:
        for suff in ["-IB",""]:
            feat = np.load(f"../results/{exp_id}/attMetrFeat{metric}{suff}_noneRed-gradcam_pp.npy",mmap_mode="r")

            if metric in ["Add","Del"]:
                sample_nb,step_nb = feat.shape[0],feat.shape[1]

            allFeat.append(feat.reshape(feat.shape[0]*feat.shape[1],-1))
    
    allFeat = np.concatenate(allFeat,axis=0)
    umap_path = f"../results/{exp_id}/attMetrFeat_noneRed-gradcam_pp.npy"
    allFeat = run_dimred_or_load(umap_path,allFeat,dimred="umap")

    lifNb = sample_nb*3
    deladd_nb = sample_nb*step_nb

    feat_lift_ib = allFeat[:lifNb] 
    feat_lift = allFeat[lifNb:lifNb*2] 
    feat_del_ib = allFeat[lifNb*2:lifNb*2+deladd_nb] 
    feat_del = allFeat[lifNb*2+deladd_nb:lifNb*2+2*deladd_nb] 
    feat_add_ib = allFeat[lifNb*2+2*deladd_nb:lifNb*2+3*deladd_nb]
    feat_add = allFeat[lifNb*2+3*deladd_nb:]

    for shape in [feat_lift_ib.shape,feat_lift.shape,feat_del_ib.shape,feat_del.shape,feat_add_ib.shape,feat_add.shape]:
        print(shape)

    for metric in ["Add","Del","Lift"]:
        
        for ib in [False,True]:
            plt.figure()
            if ib:
                if metric == "Del":
                    feat = feat_del_ib 
                elif metric == "Add":
                    feat = feat_add_ib 
                else:
                    feat = feat_lift_ib
            else:
                if metric == "Del":
                    feat = feat_del 
                elif metric == "Add":
                    feat = feat_add
                else:
                    feat = feat_lift

            if metric == "Del":
                step_nb_met = step_nb
                cmap = plt.get_cmap("plasma")
                labels = lambda x:"" 
            elif metric == "Add":
                step_nb_met = step_nb
                cmap = lambda x:plt.get_cmap("plasma")(1-x)
                labels = lambda x:""                 
            else:
                step_nb_met = 3
                cmap = lambda x:[plt.get_cmap("plasma")(0),"red","green"][int(x*step_nb_met)]
                labels = lambda x:["Original Data","Mask","Inversed mask"][x]

            feat = feat.reshape(sample_nb,step_nb_met,-1)

            range_i = range(feat.shape[1]) if feat.shape[1] == 3 else range(20,feat.shape[1],40)
            for i in range_i:
                feat_i = feat[:,i]
                if i ==0:
                    plt.scatter(feat_i[:,0],feat_i[:,1],marker="o",color=cmap(i*1.0/feat.shape[1]),alpha=0.5,label=labels(i))
                else:
                    plt.scatter(feat_i[:,0],feat_i[:,1],marker="o",color=cmap(i*1.0/feat.shape[1]),alpha=0.5,label=labels(i))
            plt.xlim(allFeat[:,0].min()-1,allFeat[:,0].max()+1)
            plt.ylim(allFeat[:,1].min()-1,allFeat[:,1].max()+1)
            
            if metric in ["Add","Del"]:
                plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap=plt.get_cmap("plasma")))
            else:
                plt.legend()
            plt.legend()
            plt.savefig(f"../vis/{exp_id}/attMetrFeat{metric}IB={ib}_noneRed-gradcam_pp.png")

def vote_for_best_model(exp_id,metric_list,img_bckgr):

    allRankings = []
    for metric in metric_list:
        perfs = loadPerf(exp_id,metric,img_bckgr=img_bckgr,reverse_met_to_min=True,pop=False)
        argsort = np.argsort(-perfs[:,1].astype("float"))
        
        ranking = np.zeros(len(argsort))

        for i in range(len(ranking)):
            ranking[argsort[i]] = i+1

        allRankings.append(ranking)

    average_ranking = np.stack(allRankings,0)
    average_ranking = average_ranking.mean(axis=0)

    return average_ranking

def find_best_methods(exp_id,img_bckgr):

    model_list = loadPerf(exp_id,"Del")[:,0]

    ranking_del = vote_for_best_model(exp_id,["Del","DelCorr","ADD"],img_bckgr)
    ranking_add = vote_for_best_model(exp_id,["Add","AddCorr","AD","IIC"],img_bckgr)
    ranking_all = vote_for_best_model(exp_id,["Del","DelCorr","ADD","Add","AddCorr","AD","IIC"],img_bckgr)

    rank_list = [ranking_del,ranking_add,ranking_all]
    rank_label = ["Deletion","Addition","Global"]

    id_to_label = get_id_to_label()

    for i in range(len(rank_list)):

        ranking = rank_list[i]

        print(rank_label[i])
        argsort = np.argsort(ranking)

        for ind in argsort:
            print("\t",round(ranking[ind],2),id_to_label[model_list[ind]])

    print(model_list[np.argmin(ranking_del)],model_list[np.argmin(ranking_add)],model_list[np.argmin(ranking)])

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
    argreader.parser.add_argument('--reduction_fact_list',type=float,metavar="INT",nargs="*",help='The list of reduction factor.',default=[])
    argreader.parser.add_argument('--inverse_xy',type=str2bool,nargs="*",metavar="BOOL",help='To inverse x and y',default=[])
    argreader.parser.add_argument('--use_dropped_list',type=str2bool,nargs="*",metavar="BOOL",help='To plot the dropped point instead of all the points',default=[])
    argreader.parser.add_argument('--full_att_map',type=str2bool,nargs="*",metavar="BOOL",help='A list of boolean indicating if the model produces full attention maps or selects points.',default=[])
    argreader.parser.add_argument('--use_threshold',type=str2bool,nargs="*",metavar="BOOL",help='To apply the threshold to filter out points',default=[])

    argreader.parser.add_argument('--mode',type=str,metavar="MODE",help='Can be "val" or "test".',default="test")
    argreader.parser.add_argument('--force_feat',type=str2bool,nargs="*",metavar="BOOL",help='To force feature plotting even when there is attention weights available.',default=[])
    argreader.parser.add_argument('--plot_id',type=str,metavar="ID",help='The plot id',default="")
    argreader.parser.add_argument('--maps_inds',type=int,nargs="*",metavar="INT",help='The index of the attention map to use when there is several. If there only one or if there is none, set this to -1',default=[])
    argreader.parser.add_argument('--receptive_field',type=str2bool,nargs="*",metavar="BOOL",help='To plot with the effective receptive field',default=[])
    argreader.parser.add_argument('--cluster',type=str2bool,nargs="*",metavar="BOOL",help='To cluster points with UMAP',default=[])
    argreader.parser.add_argument('--cluster_attention',type=str2bool,nargs="*",metavar="BOOL",help='To cluster attended points with UMAP',default=[])
    argreader.parser.add_argument('--pond_by_norm',type=str2bool,nargs="*",metavar="BOOL",help='To also show the norm of pixels along with the attention weights.',default=[])
    argreader.parser.add_argument('--only_norm',type=str2bool,nargs="*",metavar="BOOL",help='To only plot the norm of pixels',default=[])

    argreader.parser.add_argument('--gradcam',type=str2bool,nargs="*",metavar="BOOL",help='To plot gradcam instead of attention maps',default=[])
    argreader.parser.add_argument('--gradcam_maps',type=str2bool,nargs="*",metavar="BOOL",help='To plot gradcam w guided backprop instead of default gradcam',default=[])
    argreader.parser.add_argument('--gradcam_pp',type=str2bool,nargs="*",metavar="BOOL",help='To plot gradcam pp instead of default gradcam',default=[])
    argreader.parser.add_argument('--score_map',type=str2bool,nargs="*",metavar="BOOL",help='To plot score_map',default=[])

    argreader.parser.add_argument('--vargrad',type=str2bool,nargs="*",metavar="BOOL",help='To plot the varGrad method',default=[])
    argreader.parser.add_argument('--smoothgrad_sq',type=str2bool,nargs="*",metavar="BOOL",help='To plot the smooth grad sq method',default=[])
    
    argreader.parser.add_argument('--rise',type=str2bool,nargs="*",metavar="BOOL",help='To plot rise maps',default=[])

    argreader.parser.add_argument('--correctness',type=str,metavar="CORRECT",help='Set this to True to only show image where the model has given a correct answer.',default=None)
    argreader.parser.add_argument('--agregate_multi_att',type=str2bool,nargs="*",metavar="BOOL",help='Set this to True to agregate the multiple attention map when there\'s several.',default=[])

    argreader.parser.add_argument('--luminosity',type=str2bool,metavar="BOOL",help='To plot the attention maps not with a cmap but with luminosity',default=False)
    argreader.parser.add_argument('--plot_vec_emb',type=str2bool,nargs="*",metavar="BOOL",help='To plot the vector embeddings computed using UMAP on images from test set',default=[])

    argreader.parser.add_argument('--nrows',type=int,metavar="INT",help='The number of rows',default=4)
    argreader.parser.add_argument('--class_index',type=int,metavar="INT",help='The class index to show')

    argreader.parser.add_argument('--ind_to_keep',type=int,nargs="*",metavar="INT",help='The index of the images to keep')

    argreader.parser.add_argument('--interp',type=str2bool,nargs="*",metavar="BOOL",help='To smoothly interpolate the att map.',default=[])
    argreader.parser.add_argument('--direct_ind',type=str2bool,nargs="*",metavar="BOOL",help='To use direct indices',default=[])
    argreader.parser.add_argument('--no_ref',type=str2bool,nargs="*",metavar="BOOL",help='Set to True for model not refining vectors',default=[])

    argreader.parser.add_argument('--viz_id',type=str,help='The viz ID to plot gradcam like viz',default="")

    ######################################## Find failure cases #########################################""

    argreader.parser.add_argument('--list_best_pred',action="store_true",help='To create a file listing the prediction for all models at their best epoch')
    argreader.parser.add_argument('--find_hard_image',action="store_true",help='To find the hard image indexs')
    argreader.parser.add_argument('--dataset_size',type=int,metavar="INT",help='Size of the dataset (not the whole dataset, but the concerned part)')
    argreader.parser.add_argument('--threshold',type=float,metavar="INT",help='Accuracy threshold above which a model is taken into account')
    argreader.parser.add_argument('--dataset_name',type=str,metavar="NAME",help='Name of the dataset')
    argreader.parser.add_argument('--nb_class',type=int,metavar="NAME",help='Nb of big classes')

    argreader.parser.add_argument('--print_ind',type=str2bool,metavar="BOOL",help='To print image index',default=False)

    ####################################### Efficiency plot #########################################"""

    argreader.parser.add_argument('--efficiency_plot',action="store_true",help='to plot accuracy vs latency/model size. --exp_id, --model_ids and --epoch_list must be set.')

    ######################################## Compile test performance ##################################

    argreader.parser.add_argument('--compile_test',action="store_true",help='To compile the test performance of all model of an experiment. The --exp_id arg must be set. \
                                    The --model_ids can be set to put only some models in the table')

    argreader.parser.add_argument('--table_id',type=str,metavar="NAME",help='Name of the table file')

    ####################################### UMAP ############################################

    argreader.parser.add_argument('--umap',action="store_true",help='To plot features using UMAP')

    ###################################### Latency  ########################################

    argreader.parser.add_argument('--latency',action="store_true",help='To create a table with all the latencies')
    argreader.parser.add_argument('--param_nb',action="store_true",help='To create a table with all the parameter number')

    ###################################### Agregated vectors plot #####################################################

    argreader.parser.add_argument('--agr_vec',action="store_true",help='To plot all the agregated vector from test set in a 2D graph using UMAP.')
    argreader.parser.add_argument('--class_min',type=int,metavar="NAME",help='Minimum class index to plot.',default=None)
    argreader.parser.add_argument('--class_max',type=int,metavar="NAME",help='Maximum class index to plot.',default=None)

    ###################################### Importance plot #################################################

    argreader.parser.add_argument('--importance_plot',action="store_true",help='To plot the average relative norm of pixels chosen by each map.')

    ###################################### Representative vectors vs global vector ##############################

    argreader.parser.add_argument('--rep_vs_glob',type=str,metavar="PATH",help='To plot the importance of the representative vector features vs the ones from the global vector.')

    ######################################### Efficiency plot ########################

    argreader.parser.add_argument('--eff_plot',action="store_true",help='Efficiency plot')
    argreader.parser.add_argument('--red',action="store_true",help='To plot the reduced feature map models.')

    ######################################## att maps number plot ###############################

    argreader.parser.add_argument('--att_maps_nb_plot',action="store_true",help='Att maps nb plot')

    ####################################### Grad exp ##############################################

    argreader.parser.add_argument('--grad_exp',action="store_true",help='Grad exp plot')
    argreader.parser.add_argument('--grad_exp_test',action="store_true",help='Grad exp test plot')
    argreader.parser.add_argument('--grad_exp2',action="store_true",help='Grad exp 2 plot')

    ####################################### Attention metrics #################################################

    argreader.parser.add_argument('--att_metrics',action="store_true") 
    argreader.parser.add_argument('--att_metrics_stats',action="store_true") 
    argreader.parser.add_argument('--not_ignore_model',action="store_true") 
    argreader.parser.add_argument('--all_att_metrics',action="store_true") 
    argreader.parser.add_argument('--with_std',action="store_true") 
    argreader.parser.add_argument('--img_bckgr',action="store_true") 
    argreader.parser.add_argument('--pop',action="store_true")   
    argreader.parser.add_argument('--vis_repr',action="store_true") 
    argreader.parser.add_argument('--dimred_metrics',action="store_true") 
    argreader.parser.add_argument('--dimred_func',type=str,default="tsne") 
    argreader.parser.add_argument('--ranking_similarities',action="store_true") 
    argreader.parser.add_argument('--find_best_methods',action="store_true") 

    ###################################### Accuracy per video ############################################""

    argreader.parser.add_argument('--accuracy_per_video',action="store_true") 
    
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
        if len(args.gradcam_maps) == 0:
            args.gradcam_maps = [False for _ in range(len(args.model_ids))]
        if len(args.gradcam_pp) == 0:
            args.gradcam_pp = [False for _ in range(len(args.model_ids))]
        if len(args.score_map) == 0:
            args.score_map = [False for _ in range(len(args.model_ids))]  
        if len(args.vargrad) == 0:
            args.vargrad = [False for _ in range(len(args.model_ids))]   
        if len(args.smoothgrad_sq) == 0:
            args.smoothgrad_sq = [False for _ in range(len(args.model_ids))]           
        if len(args.rise) == 0:
            args.rise = [False for _ in range(len(args.model_ids))]
        if len(args.cluster) == 0:
            args.cluster = [False for _ in range(len(args.model_ids))]
        if len(args.cluster_attention) == 0:
            args.cluster_attention = [False for _ in range(len(args.model_ids))]
        if len(args.pond_by_norm) == 0:
            args.pond_by_norm = [True for _ in range(len(args.model_ids))]
        if len(args.agregate_multi_att) == 0:
            args.agregate_multi_att = [False for _ in range(len(args.model_ids))]
        if len(args.plot_vec_emb) ==  0:
            args.plot_vec_emb = [False for _ in range(len(args.model_ids))]
        if len(args.only_norm) ==  0:
            args.only_norm = [False for _ in range(len(args.model_ids))]
        if len(args.inverse_xy) ==  0:
            args.inverse_xy = [False for _ in range(len(args.model_ids))]
        if len(args.use_dropped_list) ==  0:
            args.use_dropped_list = [False for _ in range(len(args.model_ids))]
        if len(args.use_threshold) ==  0:
            args.use_threshold = [False for _ in range(len(args.model_ids))]
        if len(args.force_feat) ==  0:
            args.force_feat = [False for _ in range(len(args.model_ids))]
        if len(args.reduction_fact_list) ==  0:
            args.reduction_fact_list = [False for _ in range(len(args.model_ids))]
        if len(args.maps_inds) ==  0:
            args.maps_inds = [-1 for _ in range(len(args.model_ids))]
        if len(args.full_att_map) ==  0:
            args.full_att_map = [True for _ in range(len(args.model_ids))]
        if len(args.interp) == 0:
            args.interp = [False for _ in range(len(args.model_ids))]
        if len(args.direct_ind) == 0:
            args.direct_ind = [False for _ in range(len(args.model_ids))]
        if len(args.no_ref) == 0:
            args.no_ref = [False for _ in range(len(args.model_ids))]

        plotPointsImageDatasetGrid(args.exp_id,args.image_nb,args.epoch_list,args.model_ids,args.reduction_fact_list,args.inverse_xy,args.mode,\
                                    args.class_nb,args.use_dropped_list,args.force_feat,args.full_att_map,args.use_threshold,args.maps_inds,args.plot_id,\
                                    args.luminosity,args.receptive_field,args.cluster,args.cluster_attention,args.pond_by_norm,args.gradcam,args.gradcam_maps,args.gradcam_pp,\
                                    args.score_map,args.vargrad,args.smoothgrad_sq,args.rise,\
                                    args.nrows,args.correctness,args.agregate_multi_att,args.plot_vec_emb,args.only_norm,args.class_index,args.ind_to_keep,\
                                    args.interp,args.direct_ind,args.no_ref,args.viz_id,args)
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
    if args.latency:
        latency(args.exp_id)
    if args.param_nb:
        param_nb(args.exp_id)
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
    if args.agr_vec:
        agrVec(args.exp_id,args.model_id,args,args.class_min,args.class_max)
    if args.importance_plot:
        importancePlot(args.exp_id,args.model_id,debug=args.debug)
    if args.rep_vs_glob:
        repVSGlob(args.rep_vs_glob)
    if args.eff_plot:
        effPlot(args.red)
    if args.att_maps_nb_plot:
        attMapsNbPlot()
    if args.grad_exp:
        gradExp()
    if args.grad_exp_test:
        gradExp_test()
    if args.grad_exp2:
        gradExp2()
    if args.att_metrics:

        suff = "-IB" if args.img_bckgr else ""

        for metric in ["Add","Del","Spars","Lift"]:
            attMetrics(args.exp_id,metric=metric,ignore_model=not args.not_ignore_model,img_bckgr=args.img_bckgr)
        
        #if not os.path.exists("../results/{}/attMetrics_AddCorr{}.csv".format(args.exp_id,suff)) or not os.path.exists("../results/{}/attMetrics_DelCorr{}.csv".format(args.exp_id,suff)):
        attCorrelation(args.exp_id,img_bckgr=args.img_bckgr)

        #if not os.path.exists("../results/{}/attMetrics_Time.csv".format(args.exp_id)):
        attTime(args.exp_id)
   
        for metric in ["Add","Del","Spars","IIC","AD","ADD","DelCorr","AddCorr","Time"]:
            ttest_attMetr(args.exp_id,metric=metric,img_bckgr=args.img_bckgr)

        bar_viz(args.exp_id,args.img_bckgr)

        latex_table_figure(args.exp_id,full=args.all_att_metrics,with_std=args.with_std,img_bckgr=args.img_bckgr)

    if args.att_metrics_stats:
        attMetricsStats(args.exp_id)
    if args.accuracy_per_video:
        accuracyPerVideo(args.exp_id)
        ttest_attMetr(args.exp_id,metric="Acc")
    if args.vis_repr:
        vizRepr(args.exp_id)
    if args.dimred_metrics:
        dimred_metrics(args.exp_id,args.pop,args.dimred_func,args.img_bckgr)
    if args.ranking_similarities:
        ranking_similarities(args.exp_id,img_bckgr=args.img_bckgr,pop=args.pop)
    if args.find_best_methods:
        find_best_methods(args.exp_id,args.img_bckgr)

if __name__ == "__main__":
    main()
