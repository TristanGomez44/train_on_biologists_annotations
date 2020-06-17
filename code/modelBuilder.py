import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel

import args
import sys

import glob
from skimage.transform import resize
import matplotlib.pyplot as plt

from models import deeplab
from models import resnet
from models import pointnet2
import torchvision

try:
    import torch_geometric
except ModuleNotFoundError:
    pass

import skimage.feature
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import cv2
from scipy import ndimage

_EPSILON = 10e-7

from  torch.nn.modules.upsampling import Upsample
import time

import matplotlib.pyplot as plt
plt.switch_backend('agg')

import random

def buildFeatModel(featModelName, pretrainedFeatMod, featMap=True, bigMaps=False, layerSizeReduce=False, stride=2,dilation=1,deeplabv3_outchan=64, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("deeplabv3") != -1:
        featModel = deeplab._segm_resnet("deeplabv3", featModelName[featModelName.find("resnet"):], \
                                         outChan=deeplabv3_outchan,\
                                         pretrained=pretrainedFeatMod, featMap=featMap, layerSizeReduce=layerSizeReduce,
                                         **kwargs)
    elif featModelName.find("resnet") != -1:
        featModel = getattr(resnet, featModelName)(pretrained=pretrainedFeatMod, featMap=featMap,
                                                   layerSizeReduce=layerSizeReduce, **kwargs)
    else:
        raise ValueError("Unknown model type : ", featModelName)

    return featModel

def mapToList(map, abs, ord):
    # This extract the desired pixels in a map
    indices = tuple([torch.arange(map.size(0), dtype=torch.long).unsqueeze(1).unsqueeze(1),
                     torch.arange(map.size(1), dtype=torch.long).unsqueeze(1).unsqueeze(0),
                     ord.long().unsqueeze(1), abs.long().unsqueeze(1)])
    list = map[indices].permute(0, 2, 1)
    return list

# This class is just the class nn.DataParallel that allow running computation on multiple gpus
# but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallelModel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(DataParallelModel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self, firstModel, secondModel,nbFeat=512,drop_and_crop=False,zoom=False,zoom_max_sub_clouds=2,zoom_merge_preds=False,passOrigImage=False,reducedImgSize=1):
        super(Model, self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.zoom = zoom
        self.passOrigImage = passOrigImage
        self.reducedImgSize = reducedImgSize
        self.subcloudNb = zoom_max_sub_clouds
        self.zoom_merge_preds = zoom_merge_preds
        self.nbFeat = nbFeat
        self.drop_and_crop = drop_and_crop
        if drop_and_crop:
            self.bn = nn.BatchNorm2d(nbFeat, eps=0.001)

    def forward(self, origImgBatch):

        if self.reducedImgSize == origImgBatch.size(-1):
            imgBatch = origImgBatch
        else:
            imgBatch = F.interpolate(origImgBatch,size=(self.reducedImgSize,self.reducedImgSize))

        visResDict = self.firstModel(imgBatch)

        secModKwargs = {}
        if self.passOrigImage:
            secModKwargs["origImgBatch"] = origImgBatch

        resDict = self.secondModel(visResDict,**secModKwargs)

        resDict = merge(visResDict,resDict)

        if self.zoom:

            if len(visResDict["x"].size()) == 4:
                xSize = visResDict["x"].size()
            else:
                xSize = visResDict["x_size"]

            subCloudInd,countList = self.splitCloud(resDict['points'],xSize)

            croppedImg,_,_,_,_,bboxNbs = self.computeZoom(origImgBatch,xSize,resDict['points'],subCloudInd,countList)

            cumBboxNbs = torch.cumsum(torch.tensor([0]+bboxNbs).to(origImgBatch),dim=0).long()
            visResDict_zoom = self.firstModel(croppedImg,zoom=True)

            predBatch = visResDict_zoom["x"]

            predList = [[predBatch[cumBboxNbs[i]+j].unsqueeze(0) for j in range(bboxNbs[i])] for i in range(len(origImgBatch))]

            #Padding for image in which there is less than the required number of sub cloud
            predList = [torch.cat((torch.cat(predList[i],dim=-1),torch.zeros((1,self.nbFeat*(self.subcloudNb-bboxNbs[i]))).to(origImgBatch.device)),dim=-1) for i in range(len(origImgBatch))]
            predBatch = torch.cat(predList,dim=0)
            visResDict_zoom["x"] = predBatch

            resDict_zoom = self.secondModel(visResDict_zoom)
            resDict_zoom = merge(visResDict_zoom,resDict_zoom)
            resDict = merge(resDict_zoom,resDict,"zoom")

            resDict.pop('x_size_zoom', None)

            if self.zoom_merge_preds:
                resDict["pred"] = 0.5*resDict["pred"]+0.5*resDict["pred_zoom"]
                resDict.pop('pred_zoom', None)

        elif self.drop_and_crop:

            if self.firstModel.topk:
                features = resDict["features"]
                features = self.bn(features)
                attMaps = torch.mean(F.relu(features, inplace=True), dim=1, keepdim=True)
            else:
                attMaps = resDict["attMaps"]

            with torch.no_grad():
                crop_images = batch_augment(origImgBatch, resDict["attMaps"], mode='crop', theta=(0.4, 0.6), padding_ratio=0.1)
            resDict["pred_crop"] = self.secondModel(self.firstModel(crop_images))["pred"]

            resDict["pred_rawcrop"] = (resDict["pred_crop"]+resDict["pred"])/2

            with torch.no_grad():
                drop_images = batch_augment(origImgBatch, resDict["attMaps"], mode='drop', theta=(0.2, 0.5))
            resDict["pred_drop"] = self.secondModel(self.firstModel(drop_images))["pred"]

        resDict.pop('x_size', None)
        return resDict

    def computeZoom(self,origImg,xSize,pts,subCloudInd,countList):
        imgSize = torch.tensor(origImg.size())[-2:].to(origImg.device)
        xSize = torch.tensor(xSize)[-2:].to(origImg.device)

        bboxNbs = list(map(lambda x:min(len(x)-1,self.subcloudNb),countList))

        bboxCount = 0
        bboxs = torch.zeros((sum(bboxNbs),4)).to(origImg.device)
        pts = pts.view(origImg.size(0),-1,pts.size(-1))
        for i in range(len(pts)):
            sortedInds = torch.argsort(countList[i],descending=True)
            for j in range(1,1+min(len(countList[i])-1,self.subcloudNb)):
                ptsCoord = (subCloudInd[i,0] == sortedInds[j]).nonzero()
                ptsCoord *= (imgSize/xSize).unsqueeze(0)
                yMin,yMax = ptsCoord[:,0].min(dim=0)[0],ptsCoord[:,0].max(dim=0)[0]
                xMin,xMax = ptsCoord[:,1].min(dim=0)[0],ptsCoord[:,1].max(dim=0)[0]
                bboxs[bboxCount] = torch.cat((xMin.unsqueeze(0),xMax.unsqueeze(0),yMin.unsqueeze(0),yMax.unsqueeze(0)),dim=0)
                bboxCount += 1

        theta = bboxToTheta(bboxs,imgSize)

        origImg = torch.repeat_interleave(origImg, torch.tensor(bboxNbs).to(origImg.device), dim=0)
        pts = torch.repeat_interleave(pts, torch.tensor(bboxNbs).to(origImg.device), dim=0)
        #origImg = debugCrop(pts,origImg,xSize)

        flowField = F.affine_grid(theta, origImg.size(),align_corners=False).to(origImg.device)
        croppedImg = F.grid_sample(origImg, flowField,align_corners=False)

        #torchvision.utils.save_image(croppedImg, "../vis/cropped.png")

        return croppedImg,xMin,xMax,yMin,yMax,bboxNbs

    def splitCloud(self,points,xSize,maxSubCloud=3):

        ended = False
        randTens = torch.rand((points.size(0),1,xSize[-2],xSize[-1])).to(points.device)

        tensor = torch.zeros_like(randTens).to(points.device)

        inds = tuple([torch.arange(tensor.size(0), dtype=torch.long).unsqueeze(1).unsqueeze(1),
                         torch.arange(tensor.size(1), dtype=torch.long).unsqueeze(1).unsqueeze(0),
                         points[:,:,1].long().unsqueeze(1), points[:,:,0].long().unsqueeze(1)])

        tensor[inds] = randTens[inds]
        stepCount = 0
        tensList = [tensor]

        #Spliting the clouds
        while not ended and stepCount<40:

            newTensor = F.max_pool2d(tensor,kernel_size=3,stride=1,padding=1)

            newTensorPadded = torch.zeros_like(newTensor).to(points.device)
            newTensorPadded[inds] = newTensor[inds]
            newTensor = newTensorPadded

            ended=torch.all(torch.eq(newTensor, tensor))
            if not ended:
                tensor = newTensor.clone()

            tensList.append(newTensor)
            stepCount += 1

        #debugSplitCloud(tensList)

        #For each cloud, collecting the biggest subclouds
        ids = tensList[-1][inds].squeeze(1)

        subCloudInd = []
        countList = []
        for i in range(len(tensList[-1])):
            _,revInds,counts = torch.unique(tensList[-1][i],return_inverse=True,return_counts=True)
            subCloudInd.append(revInds.unsqueeze(0))
            countList.append(counts)

        subCloudInd = torch.cat(subCloudInd,dim=0)

        return subCloudInd,countList

def bboxToTheta(bboxs,imgSize):

    xMin,xMax,yMin,yMax = bboxs[:,0],bboxs[:,1],bboxs[:,2],bboxs[:,3]

    theta = torch.eye(3)[:2].unsqueeze(0).expand(len(xMin),-1,-1)

    #Zoom
    theta = theta.permute(1,2,0).clone()

    zoomX = (xMax-xMin)/imgSize[-2]
    zoomY = (yMax-yMin)/imgSize[-1]
    theta[0,0] = torch.max(zoomX,zoomY)
    theta[1,1] = torch.max(zoomX,zoomY)
    theta = theta.permute(2,0,1)

    #Translation
    theta = theta.permute(2,0,1).clone()
    theta[2,:,0] = 2*((xMax+xMin)/2)/imgSize[0]-1
    theta[2,:,1] = 2*((yMax+yMin)/2)/imgSize[1]-1
    theta = theta.permute(1,2,0)
    return theta

def debugCrop(pts,origImg,xSize):
    ptsValues = torch.abs(pts[:,:,4:]).sum(axis=-1)
    ptsValues = ptsValues/ptsValues.max()
    ptsValues = torch.pow(ptsValues,2)
    ptsWeights = (ptsValues-ptsValues.min(dim=0)[0])/(ptsValues.max(dim=0)[0]-ptsValues.min(dim=0)[0])
    cmPlasma = plt.get_cmap('plasma')
    ptsWeights = torch.tensor(cmPlasma(ptsWeights.cpu().detach().numpy())).to(ptsValues.device)[:,:,:3].float()
    ptsCoord = pts[:,:,:2]
    ptsImage = torch.zeros((origImg.size(0),origImg.size(1),xSize[0],xSize[1])).to(ptsValues.device)
    ptsImage[torch.arange(origImg.size(0)).unsqueeze(1),:,ptsCoord[:,:,1].long(),ptsCoord[:,:,0].long()] = ptsWeights
    ptsImage = F.interpolate(ptsImage, size=(origImg.size(-2),origImg.size(-1)))
    origImg = 0.5*origImg+0.5*ptsImage
    torchvision.utils.save_image(origImg, "../vis/orig.png")
    return origImg

def debugSplitCloud(tensList):
    cmPlasma = plt.get_cmap('rainbow')

    colorTensBatch = []
    for i,tensor in enumerate(tensList[-1]):
        values = torch.unique(tensor)
        tensor = tensor[0]
        colorTens = torch.ones((tensor.size(0),tensor.size(1),3)).to(tensList[-1].device)
        for j,value in enumerate(values):
            if value != 0:
                color = cmPlasma(j/len(values))[:3]
                colorTens[tensor == value] = torch.tensor(color).to(tensList[-1].device)
        colorTens = colorTens.permute(2,0,1).unsqueeze(0)
        colorTensBatch.append(colorTens)

    colorTensBatch = torch.cat(colorTensBatch,dim=0)
    torchvision.utils.save_image(F.interpolate(colorTensBatch,size=(224,224)), "../vis/cloudSplit_color.png")

    for i,tensor in enumerate(tensList):
        torchvision.utils.save_image(F.interpolate(tensor,size=(224,224)), "../vis/cloudSplit{}.png".format(i))

def debugSubClouds(subCloudsList,origImgBatch,resDict):
    tensor = torch.zeros((origImgBatch.size(0),56,56,3)).to(origImgBatch.device)
    for i,subClouds in enumerate(subCloudsList):
        for j,subCloud in enumerate(subClouds):
            tensor[i][subCloud[:,0],subCloud[:,1]] = 2

    tensor = tensor.permute(0,3,1,2)
    tensor = 0.5*F.interpolate(tensor,size=(224,224))+0.5*origImgBatch
    torchvision.utils.save_image(tensor, "../vis/cloudSplit_kepsSubs.png")

def merge(dictA,dictB,suffix=""):
    for key in dictA.keys():
        if key in dictB:
            dictB[key+"_"+suffix] = dictA[key]
        else:
            dictB[key] = dictA[key]
    return dictB

def plotBox(mask,xMin,xMax,yMin,yMax,chan):
    mask[chan][max(xMin,0):min(xMax,mask.size(1)-1),max(yMin,0)] = 1
    mask[chan][max(xMin,0):min(xMax,mask.size(1)-1),min(yMax,mask.size(2)-1)] = 1
    mask[chan][max(xMin,0),max(yMin,0):min(yMax,mask.size(2)-1)] = 1
    mask[chan][min(xMax,mask.size(1)-1),max(yMin,0):min(yMax,mask.size(2)-1)] = 1
    return mask
################################# Visual Model ##########################

def batch_augment(images, attention_map, mode='crop', theta=0.5, padding_ratio=0.1):
    #This comes from https://github.com/GuYuc/WS-DAN.PyTorch/blob/87779124f619ceeb445ddfb0246c8a22ff324db4/utils.py

    batches, _, imgH, imgW = images.size()

    if mode == 'crop':
        crop_images = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_c = random.uniform(*theta) * atten_map.max()
            else:
                theta_c = theta * atten_map.max()

            crop_mask = F.interpolate(atten_map, size=(imgH, imgW)) >= theta_c
            nonzero_indices = torch.nonzero(crop_mask[0, 0, ...])
            height_min = max(int(nonzero_indices[:, 0].min().item() - padding_ratio * imgH), 0)
            height_max = min(int(nonzero_indices[:, 0].max().item() + padding_ratio * imgH), imgH)
            width_min = max(int(nonzero_indices[:, 1].min().item() - padding_ratio * imgW), 0)
            width_max = min(int(nonzero_indices[:, 1].max().item() + padding_ratio * imgW), imgW)

            crop_images.append(
                F.interpolate(images[batch_index:batch_index + 1, :, height_min:height_max, width_min:width_max],
                                    size=(imgH, imgW)))
        crop_images = torch.cat(crop_images, dim=0)
        return crop_images

    elif mode == 'drop':
        drop_masks = []
        for batch_index in range(batches):
            atten_map = attention_map[batch_index:batch_index + 1]
            if isinstance(theta, tuple):
                theta_d = random.uniform(*theta) * atten_map.max()
            else:
                theta_d = theta * atten_map.max()

            drop_masks.append(F.interpolate(atten_map, size=(imgH, imgW)) < theta_d)
        drop_masks = torch.cat(drop_masks, dim=0)
        drop_images = images * drop_masks.float()
        return drop_images

    else:
        raise ValueError('Expected mode in [\'crop\', \'drop\'], but received unsupported augmentation method %s' % mode)

class FirstModel(nn.Module):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, **kwargs):
        super(FirstModel, self).__init__()

        self.featMod = buildFeatModel(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)

        self.featMap = featMap
        self.bigMaps = bigMaps

    def forward(self, x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False,aux_model=False,**kwargs):
        super(CNN2D, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps,**kwargs)

        self.aux_model= aux_model

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        res = self.featMod(x)
        features = res["x"]

        spatialWeights = torch.pow(features, 2).sum(dim=1, keepdim=True)
        retDict = {}
        retDict["attMaps"] = spatialWeights
        retDict["x"] = features

        if self.aux_model:
            retDict["auxFeat"] = features

        return retDict

def buildImageAttention(inFeat,blockNb,outChan=1):
    attention = []
    for i in range(blockNb):
        attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

class SoftMax(nn.Module):
    def __init__(self,norm=True,dim=-1):
        super(SoftMax,self).__init__()
        self.norm = norm
        self.dim = dim
    def forward(self,x):
        if self.dim == -1:
            origSize = x.size()
            x = torch.softmax(x.view(x.size(0),-1),dim=-1).view(origSize)
        elif self.dim == 1:
            x = torch.softmax(x.permute(0,2,3,1),dim=-1).permute(0,3,1,2)
        if self.norm:
            x_min,x_max = x.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0],x.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
            x = (x-x_min)/(x_max-x_min)
        return x

class CNN2D_simpleAttention(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, chan=64, attBlockNb=2,
                 attChan=16, \
                 topk=False, topk_pxls_nb=256, topk_enc_chan=64,inFeat=512,sagpool=False,sagpool_drop=False,sagpool_drop_ratio=0.5,\
                 norm_points=False,predictScore=False,score_pred_act_func="sigmoid",aux_model=False,zoom_tied_models=False,zoom_model_no_topk=False,**kwargs):

        super(CNN2D_simpleAttention, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.predictScore = predictScore
        if predictScore:
            self.attention = buildImageAttention(inFeat,attBlockNb)
            if score_pred_act_func == "sigmoid":
                self.attention_activation = torch.sigmoid
            elif score_pred_act_func == "softmax":
                self.attention_activation = SoftMax()
            elif score_pred_act_func == "relu":
                self.attention_activation = torch.relu
            else:
                raise ValueError("Unknown activation function")

        self.topk = topk
        if topk:
            self.topk_pxls_nb = topk_pxls_nb

            if not type(self.topk_pxls_nb) is list:
                self.topk_pxls_nb = [self.topk_pxls_nb]

        self.topk_enc_chan = topk_enc_chan
        if self.topk_enc_chan != -1:
            self.conv1x1 = nn.Conv2d(inaux_modelFeat,self.topk_enc_chan,1)
        else:
            self.conv1x1 = None

        self.sagpool = sagpool
        if sagpool:
            self.sagpoolModule = sagpoolLayer(topk_enc_chan if self.conv1x1 else inFeat)
            self.sagpool_drop = sagpool_drop
            self.sagpool_drop_ratio = sagpool_drop_ratio
            self.norm_points = norm_points

        self.predictScore = predictScore

        self.aux_model = aux_model

        self.zoom_tied_models = zoom_tied_models
        if self.zoom_tied_models:
            self.featMod_zoom = self.featMod
        else:
            self.featMod_zoom = buildFeatModel(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)

        self.zoom_model_no_topk = zoom_model_no_topk

    def forward(self, x,zoom=False):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        if zoom:
            output = self.featMod_zoom(x)
        else:
            output = self.featMod(x)

        if type(output) is dict:
            features = output["x"]
        else:
            features = output

        if self.topk_enc_chan != -1:
            features = self.conv1x1(features)

        retDict = {}

        if self.predictScore:
            spatialWeights = self.attention_activation(self.attention(features))
            features_weig = spatialWeights * features
        else:
            spatialWeights = torch.pow(features, 2).sum(dim=1, keepdim=True)
            features_weig = features

        if self.topk and (not (zoom and self.zoom_model_no_topk)):
            featVecList = []

            flatSpatialWeights = spatialWeights.view(spatialWeights.size(0), -1)
            allFlatVals, allFlatInds = torch.topk(flatSpatialWeights, max(self.topk_pxls_nb), dim=-1, largest=True)

            for i in range(len(self.topk_pxls_nb)):
                flatVals, flatInds = allFlatVals[:,:self.topk_pxls_nb[i]], allFlatInds[:,:self.topk_pxls_nb[i]]
                abs, ord = (flatInds % spatialWeights.shape[-1], flatInds // spatialWeights.shape[-1])
                depth = torch.zeros(abs.size(0), abs.size(1), 1).to(x.device)
                featureList = mapToList(features_weig, abs, ord)

                points = torch.cat((abs.unsqueeze(2).float(), ord.unsqueeze(2).float(), depth, featureList), dim=-1).float()
                addOrCat(retDict,'points',points,dim=1)

                if self.sagpool:
                    abs, ord = abs.unsqueeze(-1).float(), ord.unsqueeze(-1).float()
                    points = torch.cat((abs, ord, depth), dim=-1).float()
                    ptsDict = {"batch" : torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(-1).to(points.device),
                               "pos" : points.reshape(points.size(0) * points.size(1), points.size(2)),
                               "pointfeatures" : featureList.reshape(featureList.size(0) * featureList.size(1), featureList.size(2))}
                    if self.norm_points:
                        ptsDict["pos"][:,:2] = (2*ptsDict["pos"][:,:2]/(x.size(-1)-1))-1
                    ptsDict = applySagPool(self.sagpoolModule,ptsDict,x.size(0),self.sagpool_drop,self.sagpool_drop_ratio)
                    finalPtsNb = int(featureList.size(1)*self.sagpool_drop_ratio) if self.sagpool_drop else featureList.size(1)
                    featureList = ptsDict["pointfeatures"].reshape(featureList.size(0),finalPtsNb, featureList.size(2))

                addOrCat(retDict,"pointfeatures",featureList,dim=1)

                features_agr = featureList.mean(dim=1)
                featVecList.append(features_agr)
            features_agr = torch.cat(featVecList,dim=-1)
        else:
            features_agr = self.avgpool(features_weig)
            features_agr = features_agr.view(features.size(0), -1)

        retDict["x"] = features_agr
        retDict["x_size"] = features_weig.size()
        retDict["attMaps"] = spatialWeights
        retDict["features"] = features

        if self.aux_model:
            retDict["auxFeat"] = features

        return retDict

class CNN2D_bilinearAttPool(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, chan=64, attBlockNb=2,
                 attChan=16,inFeat=512,nb_parts=3,aux_model=False,**kwargs):

        super(CNN2D_bilinearAttPool, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.attention = buildImageAttention(inFeat,attBlockNb,nb_parts+1)
        self.nb_parts = nb_parts
        self.attention_activation = SoftMax(norm=False,dim=1)

        self.aux_model = aux_model

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        output = self.featMod(x)

        if type(output) is dict:
            features = output["x"]
        else:
            features = output

        retDict = {}

        spatialWeights = self.attention_activation(self.attention(features))
        features_weig = (spatialWeights[:,:self.nb_parts].unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),features.size(1)*(spatialWeights.size(1)-1),features.size(2),features.size(3))
        features_agr = self.avgpool(features_weig)

        features_agr = features_agr.view(features.size(0), -1)

        retDict["x"] = features_agr
        retDict["x_size"] = features_weig.size()
        retDict["attMaps"] = spatialWeights

        if self.aux_model:
            retDict["auxFeat"] = features

        return retDict


def addOrCat(dict,key,tensor,dim):
    if not key in dict:
        dict[key] = tensor
    else:
        dict[key] = torch.cat((dict[key],tensor),dim=dim)

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super(SecondModel, self).__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbFeatAux,nbClass, dropout,aux_model=False,zoom=False,zoom_max_sub_clouds=2):
        super(LinearSecondModel, self).__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat, self.nbClass)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_model = aux_model
        if self.aux_model:
            self.aux_model = nn.Linear(nbFeatAux, self.nbClass)

        self.zoom = zoom
        if self.zoom:
            self.zoom_model = nn.Linear(zoom_max_sub_clouds*nbFeat,self.nbClass)

    def forward(self, visResDict):

        x = visResDict["x"]

        if len(x.size()) == 4:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)

        x = self.dropout(x)
        if x.size(-1) == self.nbFeat:
            x = self.linLay(x)
        else:
            x = self.zoom_model(x)

        retDict = {"pred": x}

        if self.aux_model:
            retDict["auxPred"] = self.aux_model(visResDict["auxFeat"].mean(dim=-1).mean(dim=-1))

        return retDict

def sagpoolLayer(inChan):
    return pointnet2.SAModule(1, 0.2, pointnet2.MLP_linEnd([inChan+3,64,1]))

def applySagPool(module,retDict,batchSize,drop,dropRatio):

    nodeWeight,pos,batch = module(retDict['pointfeatures'],retDict["pos"],retDict["batch"])
    nodeWeight = torch.tanh(nodeWeight)
    ptsNb = nodeWeight.size(0)//batchSize

    retDict["pointWeights"] = nodeWeight.reshape(batchSize,ptsNb)

    if drop:

        ptsNb_afterpool = int(ptsNb*dropRatio)

        nodeWeight = nodeWeight.reshape(batchSize,ptsNb)
        retDict["pointfeatures"] = retDict["pointfeatures"].unsqueeze(0).reshape(batchSize,ptsNb,-1)
        retDict["pos"] = retDict["pos"].unsqueeze(0).reshape(batchSize,ptsNb,-1)
        retDict["batch"] = retDict["batch"].unsqueeze(0).reshape(batchSize,ptsNb)

        #Finding most important pixels
        nodeWeight, inds = torch.topk(nodeWeight, int(nodeWeight.size(1)*dropRatio), dim=-1, largest=True)

        #Selecting those pixels
        retDict["pointfeatures"] = retDict["pointfeatures"][torch.arange(batchSize).unsqueeze(1),inds]
        retDict["pos"] = retDict["pos"][torch.arange(batchSize).unsqueeze(1),inds]
        retDict["batch"] = retDict["batch"][torch.arange(batchSize).unsqueeze(1),inds]

        retDict["points_dropped"] = torch.cat((retDict["pos"],retDict["pointfeatures"],nodeWeight.unsqueeze(-1)),dim=-1)

        #Reshaping
        nodeWeight = nodeWeight.reshape(batchSize*ptsNb_afterpool).unsqueeze(1)
        retDict["pointfeatures"] = retDict["pointfeatures"].reshape(batchSize*ptsNb_afterpool,-1)
        retDict["pos"] = retDict["pos"].reshape(batchSize*ptsNb_afterpool,-1)
        retDict["batch"] = retDict["batch"].reshape(batchSize*ptsNb_afterpool)

    retDict["pointfeatures"] = nodeWeight*retDict["pointfeatures"]
    return retDict

class ReinforcePointExtractor(nn.Module):

    def __init__(self, cuda, nbFeat, point_nb, encoderChan, hasLinearProb, use_baseline, reinf_linear_only):

        super(ReinforcePointExtractor, self).__init__()

        self.conv1x1 = nn.Conv2d(nbFeat, encoderChan, kernel_size=1, stride=1)
        self.point_extracted = point_nb
        self.hasLinearProb = hasLinearProb
        self.use_baseline = use_baseline

        if self.hasLinearProb:
            self.lin_prob = nn.Conv2d(encoderChan, 1, kernel_size=1, stride=1)

        self.reinf_linear_only = reinf_linear_only
        if use_baseline:
            self.baseline_linear = nn.Linear(nbFeat, 1)

    def forward(self, featureMaps):

        # Because of zero padding, the border are very active, so we remove it.
        featureMaps = featureMaps[:, :, 3:-3, 3:-3]
        pointFeaturesMap = self.conv1x1(featureMaps)

        if self.hasLinearProb:
            if self.reinf_linear_only:
                x = self.lin_prob(pointFeaturesMap.detach())
            else:
                x = self.lin_prob(pointFeaturesMap)
            flatX = x.view(x.size(0), -1)
            # probs = F.softmax(flatX, dim=(1))+_EPSILON
            flatX = torch.sigmoid(flatX)
            probs = flatX / (flatX.sum(dim=1, keepdim=True) +_EPSILON)
            # probs = flatX / flatX.sum(dim=-1, keepdim=True)[0]
        else:
            x = torch.pow(pointFeaturesMap, 2).sum(dim=1, keepdim=True)

        # flatInds = torch.distributions.categorical.Categorical(probs=probs).sample(torch.tensor([self.point_extracted]))
        _, flatInds = torch.topk(probs, self.point_extracted, dim=-1, largest=True)


        abs, ord = (flatInds % x.shape[-1], flatInds // x.shape[-1])

        retDict = {}

        pointFeat = mapToList(pointFeaturesMap, abs, ord)

        depth = torch.zeros(abs.size(0), abs.size(1), 1).to(x.device)

        abs, ord = abs.unsqueeze(-1).float(), ord.unsqueeze(-1).float()
        points = torch.cat((abs, ord, depth), dim=-1).float()
        retDict['points'] = torch.cat((abs, ord, depth, pointFeat), dim=-1).float()
        retDict["batch"] = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(-1).to(points.device)
        retDict["pos"] = points.reshape(points.size(0) * points.size(1), points.size(2))
        retDict["pointfeatures"] = pointFeat.reshape(pointFeat.size(0) * pointFeat.size(1), pointFeat.size(2))
        retDict["probs"] = probs
        retDict["flatInds"] = flatInds

        if self.use_baseline:
            retDict["baseFeat"] = featureMaps.mean(dim=-1).mean(dim=-1)
            retDict["baseline"] = F.relu((self.baseline_linear(retDict['baseFeat'].detach())))

        return retDict

    def updateDict(self, device):

        if not device in self.ordKerDict.keys():
            self.ordKerDict[device] = self.ordKer.to(device)
            self.absKerDict[device] = self.absKer.to(device)
            self.spatialWeightKerDict[device] = self.spatialWeightKer.to(device)

class ClusterModel(nn.Module):
    def __init__(self, nb_cluster):
        super(ClusterModel, self).__init__()
        pass

    def forward(self, retDict):
        return retDict

def getLayerNb(backbone_name):
    if backbone_name.find("9") != -1:
        return 2
    else:
        return 4

def getResnetFeat(backbone_name, backbone_inplanes,deeplabv3_outchan):
    if backbone_name == "resnet50" or backbone_name == "resnet101" or backbone_name == "resnet151":
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("deeplab") != -1:
        nbFeat = deeplabv3_outchan
    elif backbone_name.find("resnet34") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet14") != -1:
        nbFeat = backbone_inplanes * 2 ** (3 - 1)
    elif backbone_name.find("resnet9") != -1:
        nbFeat = backbone_inplanes * 2 ** (2 - 1)
    elif backbone_name.find("resnet4") != -1:
        nbFeat = backbone_inplanes * 2 ** (1 - 1)
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args):
    ############### Visual Model #######################
    if args.first_mod.find("resnet") != -1:

        if not args.multi_level_feat:
            nbFeat = getResnetFeat(args.first_mod, args.resnet_chan,args.deeplabv3_outchan)
        else:
            nbFeat = args.multi_level_feat_outchan

        if args.resnet_bilinear:
            CNNconst = CNN2D_bilinearAttPool
            kwargs = {"inFeat":nbFeat,"aux_model":args.aux_model,"nb_parts":args.resnet_bil_nb_parts}
            nbFeatAux = nbFeat
            nbFeat *= args.resnet_bil_nb_parts
        elif args.resnet_simple_att:
            CNNconst = CNN2D_simpleAttention
            kwargs = {"inFeat":nbFeat,
                      "topk": args.resnet_simple_att_topk,
                      "topk_pxls_nb": args.resnet_simple_att_topk_pxls_nb,
                      "topk_enc_chan":args.resnet_simple_att_topk_enc_chan,
                      "sagpool":args.resnet_simple_att_topk_sagpool,
                      "sagpool_drop":args.resnet_simple_att_topk_sagpool_drop,
                      "sagpool_drop_ratio":args.resnet_simple_att_topk_sagpool_ratio,
                      "norm_points":args.norm_points,\
                      "predictScore":args.resnet_simple_att_pred_score,
                      "score_pred_act_func":args.resnet_simple_att_score_pred_act_func,
                      "aux_model":args.aux_model,\
                      "zoom_tied_models":args.zoom_tied_models,\
                      "zoom_model_no_topk":args.zoom_model_no_topk}
            if args.resnet_simple_att_topk_enc_chan != -1:
                nbFeat = args.resnet_simple_att_topk_enc_chan

            nbFeatAux = nbFeat
            if type(args.resnet_simple_att_topk_pxls_nb) is list and len(args.resnet_simple_att_topk_pxls_nb) > 1:
                nbFeat *= len(args.resnet_simple_att_topk_pxls_nb)

        else:
            CNNconst = CNN2D
            kwargs = {"aux_model":args.aux_model}
            nbFeatAux = nbFeat

        firstModel = CNNconst(args.first_mod, args.pretrained_visual, featMap=True,chan=args.resnet_chan, stride=args.resnet_stride,
                              dilation=args.resnet_dilation, \
                              attChan=args.resnet_att_chan, attBlockNb=args.resnet_att_blocks_nb,
                              attActFunc=args.resnet_att_act_func, \
                              num_classes=args.class_nb, \
                              layerSizeReduce=args.resnet_layer_size_reduce,
                              preLayerSizeReduce=args.resnet_prelay_size_reduce, \
                              applyStrideOnAll=args.resnet_apply_stride_on_all, \
                              replaceBy1x1=args.resnet_replace_by_1x1,\
                              reluOnLast=args.relu_on_last_layer,
                              multiLevelFeat=args.multi_level_feat,\
                              multiLev_outChan=args.multi_level_feat_outchan,\
                              multiLev_cat=args.multi_level_feat_cat,\
                              deeplabv3_outchan=args.deeplabv3_outchan,\
                              **kwargs)
    else:
        raise ValueError("Unknown visual model type : ", args.first_mod)

    if args.freeze_visual:
        for param in firstModel.parameters():
            param.requires_grad = False

    ############### Second Model #######################

    zoomArgs= {"zoom":args.zoom,"zoom_max_sub_clouds":args.zoom_max_sub_clouds}

    if args.zoom and args.second_mod != "linear":
        raise ValueError("zoom must be used with linear second model")

    if args.second_mod == "linear":
        secondModel = LinearSecondModel(nbFeat,nbFeatAux, args.class_nb, args.dropout,args.aux_model,**zoomArgs)
    else:
        raise ValueError("Unknown temporal model type : ", args.second_mod)

    ############### Whole Model ##########################

    net = Model(firstModel, secondModel,drop_and_crop=args.drop_and_crop,zoom=args.zoom,zoom_max_sub_clouds=args.zoom_max_sub_clouds,\
                zoom_merge_preds=args.zoom_merge_preds,\
                passOrigImage=False,reducedImgSize=args.reduced_img_size)

    if args.cuda:
        net.cuda()

    if args.multi_gpu:
        net = DataParallelModel(net)

    return net


def addArgs(argreader):
    argreader.parser.add_argument('--first_mod', type=str, metavar='MOD',
                                  help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float, metavar='D',
                                  help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--second_mod', type=str, metavar='MOD',
                                  help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--freeze_visual', type=args.str2bool, metavar='BOOL',
                                  help='To freeze the weights of the visual model during training.')

    argreader.parser.add_argument('--pretrained_visual', type=args.str2bool, metavar='BOOL',
                                  help='To have a visual feature extractor pretrained on ImageNet.')

    argreader.parser.add_argument('--zoom', type=args.str2bool, metavar='BOOL',
                                  help='To use with a model that generates points. To zoom on the parts of the images where the points are focused an apply the model a second time on it.')

    argreader.parser.add_argument('--zoom_max_sub_clouds', type=int, metavar='NB',
                                  help='The maximum number of subclouds to use.')

    argreader.parser.add_argument('--zoom_merge_preds', type=args.str2bool, metavar='BOOL',
                                  help='To merge the predictions produced by the first model and by the model using crops.')

    argreader.parser.add_argument('--zoom_tied_models', type=args.str2bool, metavar='BOOL',
                                  help='To tie the weights of the global and the zoom model.')

    argreader.parser.add_argument('--zoom_model_no_topk', type=args.str2bool, metavar='BOOL',
                                  help='To force the zoom model to not use only the top-K pixels but all of them when the global model is a top-K model.')

    argreader.parser.add_argument('--aux_model', type=args.str2bool, metavar='INT',
                                  help='To train an auxilliary model that will apply average pooling and a dense layer on the feature map\
                        to make a prediction alongside the principal model\'s one.')

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_dilation', type=int, metavar='INT',
                                  help='The dilation for the visual model when resnet is used')

    argreader.parser.add_argument('--resnet_layer_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the layer 2,3 and 4 when the resnet model is used.')
    argreader.parser.add_argument('--resnet_prelay_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the convolution and the maxpooling before the layer 1.')
    argreader.parser.add_argument('--resnet_simple_att', type=args.str2bool, metavar='INT',
                                  help='To apply a simple attention on top of the resnet model.')
    argreader.parser.add_argument('--resnet_simple_att_topk', type=args.str2bool, metavar='BOOL',
                                  help='To use top-k feature as attention model with resnet. Ignored when --resnet_simple_att is False.')
    argreader.parser.add_argument('--resnet_simple_att_topk_pxls_nb', type=int, metavar='INT',
                                  nargs="*",help='The value of k when using top-k selection for resnet simple attention. Can be a list of values. Ignored when --resnet_simple_att_topk is False.')
    argreader.parser.add_argument('--resnet_simple_att_topk_enc_chan', type=int, metavar='NB',
                                  help='For the resnet_simple_att_topk model. This is the number of output channel of the encoder. Ignored when --resnet_simple_att_topk is False.')

    argreader.parser.add_argument('--resnet_simple_att_topk_sagpool', type=args.str2bool, metavar='BOOL',
                                  help='To use sagpool.')
    argreader.parser.add_argument('--resnet_simple_att_topk_sagpool_drop', type=args.str2bool, metavar='BOOL',
                                  help='To use sagpool with point dropping.')
    argreader.parser.add_argument('--resnet_simple_att_topk_sagpool_ratio', type=float, metavar='BOOL',
                                  help='The ratio of point dropped.')
    argreader.parser.add_argument('--resnet_simple_att_pred_score', type=args.str2bool, metavar='BOOL',
                                  help='To predict the score of each pixel, instead of using their norm to select them.')
    argreader.parser.add_argument('--resnet_simple_att_score_pred_act_func', type=str, metavar='STR',
                                  help='The activation function of the attention module.')

    argreader.parser.add_argument('--resnet_apply_stride_on_all', type=args.str2bool, metavar='NB',
                                  help='Apply stride on every non 3x3 convolution')
    argreader.parser.add_argument('--resnet_replace_by_1x1', type=args.str2bool, metavar='NB',
                                  help='Replace the second 3x3 conv of BasicBlock by a 1x1 conv')

    argreader.parser.add_argument('--resnet_att_chan', type=int, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The number of channels in the attention module.')
    argreader.parser.add_argument('--resnet_att_blocks_nb', type=int, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The number of blocks in the attention module.')
    argreader.parser.add_argument('--resnet_att_act_func', type=str, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The activation function for the attention weights. Can be "sigmoid", "relu" or "tanh+relu".')

    argreader.parser.add_argument('--reduced_img_size', type=int, metavar='BOOL',
                                  help="The size at which the image is reduced at the begining of the process")

    argreader.parser.add_argument('--norm_points', type=args.str2bool, metavar='BOOL',
                                  help="To normalize the points before passing them to pointnet")

    argreader.parser.add_argument('--relu_on_last_layer', type=args.str2bool, metavar='BOOL',
                                  help="To apply relu on the last layer of the feature extractor.")

    argreader.parser.add_argument('--multi_level_feat', type=args.str2bool, metavar='BOOL',
                                  help="To extract multi-level features by combining features maps at every layers.")
    argreader.parser.add_argument('--multi_level_feat_outchan', type=int, metavar='BOOL',
                                  help="The number of channels of the multi level feature maps.")
    argreader.parser.add_argument('--multi_level_feat_cat', type=args.str2bool, metavar='BOOL',
                                  help="To concatenate the features instead of computing the mean")

    argreader.parser.add_argument('--deeplabv3_outchan', type=int, metavar='BOOL',
                                  help="The number of output channel of deeplabv3")

    argreader.parser.add_argument('--resnet_bil_nb_parts', type=int, metavar='INT',
                                  help="The number of parts for the bilinear model.")
    argreader.parser.add_argument('--resnet_bilinear', type=args.str2bool, metavar='BOOL',
                                  help="To use bilinear attention")


    argreader.parser.add_argument('--drop_and_crop', type=args.str2bool, metavar='BOOL',
                                  help="To crop and drop part of the images where the attention is focused.")



    return argreader
