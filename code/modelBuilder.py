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

    def __init__(self, firstModel, secondModel,zoom=False,passOrigImage=False,reducedImgSize=1):
        super(Model, self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.zoom = zoom
        self.passOrigImage = passOrigImage
        self.reducedImgSize = reducedImgSize

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
            croppedImg,xMinCr,xMaxCr,yMinCr,yMaxCr = self.computeZoom(origImgBatch,visResDict["x"],resDict)
            visResDict_zoom = self.firstModel(croppedImg)
            resDict_zoom = self.secondModel(visResDict_zoom["x"])
            resDict_zoom = merge(visResDict_zoom,resDict_zoom)
            resDict = merge(resDict_zoom,resDict,"zoom")

        return resDict

    def computeZoom(self,origImg,x,retDict):
        imgSize = torch.tensor(origImg.size())[-2:].to(x.device)
        xSize = torch.tensor(x.size())[-2:].to(x.device)

        pts = retDict['points']
        pts = pts.view(x.size(0),-1,pts.size(-1))

        ptsValues = torch.abs(pts[:,:,4:]).sum(axis=-1)
        ptsValues = ptsValues/ptsValues.max()
        ptsValues = torch.pow(ptsValues,2)
        means = (pts[:,:,:2]*ptsValues.unsqueeze(2)).sum(dim=1)/ptsValues.sum(dim=1).unsqueeze(1)
        stds = torch.sqrt((torch.pow(pts[:,:,:2]-pts[:,:,:2].mean(dim=1).unsqueeze(1),2)*ptsValues.unsqueeze(2)).sum(dim=1)/ptsValues.sum(dim=1).unsqueeze(1))

        means *= (imgSize/xSize).unsqueeze(0)
        stds *= (imgSize/xSize).unsqueeze(0)

        xMin,xMax,yMin,yMax = means[:,1]-2*stds[:,1],means[:,1]+2*stds[:,1],means[:,0]-2*stds[:,0],means[:,0]+2*stds[:,0]
        xMin = xMin.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1,origImg.size(1),-1,-1)
        xMax = xMax.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1,origImg.size(1),-1,-1)
        yMin = yMin.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1,origImg.size(1),-1,-1)
        yMax = yMax.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1,origImg.size(1),-1,-1)

        indsX = torch.arange(origImg.size(-1)).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(-1,origImg.size(1),-1,-1).to(origImg.device)
        indsY = torch.arange(origImg.size(-2)).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(-1,origImg.size(1),-1,-1).to(origImg.device)

        maskedImage = torch.zeros_like(origImg)

        mask = (yMin < indsY)*(indsY < yMax)*(xMin < indsX)*(indsX < xMax)
        maskedImage[mask] = origImg[mask]

        theta = torch.eye(3)[:2].unsqueeze(0).expand(x.size(0),-1,-1)

        flowField = F.affine_grid(theta, origImg.size(),align_corners=False).to(x.device)
        croppedImg = F.grid_sample(maskedImage, flowField,align_corners=False,padding_mode='border')

        return croppedImg,xMin,xMax,yMin,yMax

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

class FirstModel(nn.Module):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, **kwargs):
        super(FirstModel, self).__init__()

        self.featMod = buildFeatModel(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)

        self.featMap = featMap
        self.bigMaps = bigMaps

    def forward(self, x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False,**kwargs):
        super(CNN2D, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps,**kwargs)

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        res = self.featMod(x)

        # N x D
        if type(res) is dict:
            # Some feature model can return a dictionnary instead of a tensor
            return res
        else:
            return {'x': res}

def buildImageAttention(inFeat,blockNb):
    attention = []
    for i in range(blockNb):
        attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, 1))
    return nn.Sequential(*attention)

class CNN2D_simpleAttention(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, chan=64, attBlockNb=2,
                 attChan=16, \
                 topk=False, topk_pxls_nb=256, topk_enc_chan=64,inFeat=512,sagpool=False,sagpool_drop=False,sagpool_drop_ratio=0.5,\
                 norm_points=False,predictScore=False,aux_model=False,**kwargs):

        super(CNN2D_simpleAttention, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.predictScore = predictScore
        if predictScore:
            self.attention = buildImageAttention(inFeat,attBlockNb)

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

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        output = self.featMod(x)

        if type(output) is dict:
            features = output["x"]
        else:
            features = output

        if self.topk_enc_chan != -1:
            features = self.conv1x1(features)

        retDict = {}

        if self.predictScore:
            spatialWeights = torch.sigmoid(self.attention(features))
            features_weig = spatialWeights * features
        else:
            spatialWeights = torch.pow(features, 2).sum(dim=1, keepdim=True)
            features_weig = features

        if self.topk:

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

    def __init__(self, nbFeat, nbFeatAux,nbClass, dropout,aux_model=False):
        super(LinearSecondModel, self).__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat, self.nbClass)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.aux_model = aux_model
        if self.aux_model:
            self.aux_model = nn.Linear(nbFeatAux, self.nbClass)

    def forward(self, visResDict):

        x = visResDict["x"]

        if len(x.size()) == 4:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)

        x = self.dropout(x)
        x = self.linLay(x)

        retDict = {"pred": x}

        if self.aux_model:
            retDict["auxPred"] = self.aux_model(visResDict["auxFeat"].mean(dim=-1).mean(dim=-1))

        return retDict

def sagpoolLayer(inChan):
    return pointnet2.SAModule(1, 0.2, pointnet2.MLP_linEnd([inChan+3,64,1]))

class TopkPointExtractor(nn.Module):

    def __init__(self, cuda, nbFeat,point_nb,encoderChan, \
                 furthestPointSampling, furthestPointSampling_nb_pts,\
                 auxModel,auxModelTopk,topk_euclinorm,predictPixelScore,imageAttBlockNb,cannyedge,cannyedge_sigma,orbedge,sobel,patchsim,\
                 patchsim_patchsize,patchsim_groupNb,patchsim_neiSimRefin,nms,no_feat,patchsim_mod,norm_points,sagpool,\
                 sagpool_drop,sagpool_drop_ratio,bottomK):

        super(TopkPointExtractor, self).__init__()

        if nbFeat == encoderChan or encoderChan == 0:
            self.conv1x1 = None
            encoderChan = nbFeat
        else:
            self.conv1x1 = nn.Conv2d(nbFeat, encoderChan, kernel_size=1, stride=1)

        self.point_extracted = point_nb
        self.furthestPointSampling = furthestPointSampling
        self.furthestPointSampling_nb_pts = furthestPointSampling_nb_pts
        self.sagpool = sagpool

        if self.sagpool:
            #self.sagpoolModule = torch_geometric.nn.SAGPooling(in_channels=encoderChan if self.conv1x1 else nbFeat, ratio=sagpool_pts_nb/point_nb)
            self.sagpoolModule = sagpoolLayer(encoderChan if self.conv1x1 else nbFeat)
        else:
            self.sagpoolModule = None

        self.topk_euclinorm = topk_euclinorm

        self.auxModel = auxModel
        self.auxModelTopk = auxModelTopk

        self.predictPixelScore = predictPixelScore
        if self.predictPixelScore:
            self.imageAttention = buildImageAttention(encoderChan,imageAttBlockNb)

        self.cannyedge = cannyedge
        self.cannyedge_sigma = cannyedge_sigma
        self.orbedge = orbedge
        self.sobel = sobel
        self.nms = nms

        self.patchsim = patchsim
        if self.patchsim:
            self.patchSimCNN = PatchSimCNN("resnet9",False,patchsim_groupNb,chan=32,patchSize=patchsim_patchsize,neiSimRefin=patchsim_neiSimRefin,featMod=patchsim_mod,nms=nms)
            self.patchSim_useAllLayers = (patchsim_mod is None)

        if (self.cannyedge or self.orbedge or self.sobel) and self.patchsim:
            raise ValueError("cannyedge and patchsim can't be True at the same time.")

        self.no_feat = no_feat
        self.norm_points = norm_points
        self.bottomK = bottomK
        self.sagpool_drop = sagpool_drop
        self.sagpool_drop_ratio = sagpool_drop_ratio

    def forward(self, featureMaps,pointFeaturesMap=None,x=None,**kwargs):
        retDict = {}

        # Because of zero padding, the border are very active, so we remove it.
        if pointFeaturesMap is None:
            if (not self.cannyedge) and (not self.orbedge) and (not self.sobel) and (not self.patchsim):
                featureMaps = featureMaps[:, :, 3:-3, 3:-3]

            if self.conv1x1:
                pointFeaturesMap = self.conv1x1(featureMaps)
            else:
                pointFeaturesMap = featureMaps

        if (not self.cannyedge) and (not self.orbedge):
            if x is None:
                if self.topk_euclinorm:
                    x = torch.pow(pointFeaturesMap, 2).sum(dim=1, keepdim=True)
                elif self.patchsim:
                    with torch.no_grad():
                        x = -self.patchSimCNN(kwargs["origImgBatch"],returnLayer="last" if self.patchSim_useAllLayers else '2')
                    x_min = x.view(pointFeaturesMap.size(0),-1).min(dim=-1,keepdim=True)[0].unsqueeze(1).unsqueeze(1)
                    x_max = x.view(pointFeaturesMap.size(0),-1).max(dim=-1,keepdim=True)[0].unsqueeze(1).unsqueeze(1)
                    x = (x-x_min)/(x_max-x_min)

                    pointFeaturesMap = F.interpolate(pointFeaturesMap,size=(x.size(-2),x.size(-1)))
                    pointFeaturesMap = pointFeaturesMap*x
                elif self.predictPixelScore:
                    x = torch.sigmoid(self.imageAttention(pointFeaturesMap))
                    pointFeaturesMap = (pointFeaturesMap*x).float()
                elif self.sobel:
                    with torch.no_grad():
                        x = -computeSobel(kwargs["origImgBatch"],self.nms)
                    pointFeaturesMap = F.interpolate(pointFeaturesMap,size=(x.size(-2),x.size(-1)))
                    pointFeaturesMap = pointFeaturesMap*x
                else:
                    x = F.relu(pointFeaturesMap).sum(dim=1, keepdim=True)
            else:
                x = F.interpolate(torch.pow(pointFeaturesMap, 2).sum(dim=1, keepdim=True),size=(x.size(-2),x.size(-1)))

            pointFeaturesMap = pointFeaturesMap.float()
            retDict["featVolume"] = pointFeaturesMap
            retDict["prob_map"] = x

            flatX = x.view(x.size(0), -1)

            _, flatInds = torch.topk(flatX, self.point_extracted, dim=-1, largest=not self.bottomK)

            abs, ord = (flatInds % x.shape[-1], flatInds // x.shape[-1])

        else:
            edges = computeEdges(kwargs["origImgBatch"],self.cannyedge_sigma,self.point_extracted,featureMaps.size(-1),self.orbedge,self.point_extracted)
            ord, abs = edges[:,:,0],edges[:,:,1]

        if self.furthestPointSampling and self.furthestPointSampling_nb_pts / abs.size(1) < 1:
            points = torch.cat((abs.unsqueeze(-1), ord.unsqueeze(-1)), dim=-1).float()

            exampleInds = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(-1).to(points.device)
            selectedPointInds = torch_geometric.nn.fps(points.view(-1, 2), exampleInds,ratio=self.furthestPointSampling_nb_pts / abs.size(1))
            selectedPointInds = selectedPointInds.reshape(points.size(0), -1)
            selectedPointInds = selectedPointInds % abs.size(1)

            abs = abs[torch.arange(abs.size(0)).unsqueeze(1).long(), selectedPointInds.long()]
            ord = ord[torch.arange(ord.size(0)).unsqueeze(1).long(), selectedPointInds.long()]

        depth = torch.zeros(abs.size(0), abs.size(1), 1).to(featureMaps.device)

        if self.no_feat:
            pointFeat = None
        else:
            pointFeat = mapToList(pointFeaturesMap, abs.int(), ord.int())

        abs, ord = abs.unsqueeze(-1).float(), ord.unsqueeze(-1).float()
        points = torch.cat((abs, ord, depth), dim=-1).float()

        retDict["batch"] = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(-1).to(points.device)
        retDict["pos"] = points.reshape(points.size(0) * points.size(1), points.size(2))

        if self.auxModel:
            if self.auxModelTopk:
                retDict["auxFeat"] = F.relu(pointFeat).mean(dim=1)
            else:
                retDict["auxFeat"] = F.relu(featureMaps).mean(dim=-1).mean(dim=-1)

        if self.no_feat:
            retDict['points'] = torch.cat((abs, ord, depth), dim=-1).float()
            retDict["pointfeatures"] = None
        else:
            retDict['points'] = torch.cat((abs, ord, depth, pointFeat), dim=-1).float()
            retDict["pointfeatures"] = pointFeat.reshape(pointFeat.size(0) * pointFeat.size(1), pointFeat.size(2))

        if self.norm_points:
            if self.cannyedge or self.orbedge:
                retDict["pos"] = (2*retDict["pos"]/(featureMaps.size(-1)-1))-1
            else:
                retDict["pos"][:,:2] = (2*retDict["pos"][:,:2]/(x.size(-1)-1))-1

        if self.sagpool:
            retDict = applySagPool(self.sagpoolModule,retDict,featureMaps.size(0),self.sagpool_drop,self.sagpool_drop_ratio)

        return retDict

    def updateDict(self, device):

        if not device in self.ordKerDict.keys():
            self.ordKerDict[device] = self.ordKer.to(device)
            self.absKerDict[device] = self.absKer.to(device)
            self.spatialWeightKerDict[device] = self.spatialWeightKer.to(device)

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

def sobelFunc(img):

    img = (255*(img-img.min())/(img.max()-img.min()))
    img = resize(img, (100,100),anti_aliasing=True,mode="constant",order=3)*255

    img = img.astype('int32')
    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    #mag = dx.astype("float64")
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)

    return mag

def computeSobel(origImgBatch,nms):
    allSobel = []
    for i in range(len(origImgBatch)):
        img_np= origImgBatch[i].detach().cpu().permute(1,2,0).numpy().mean(axis=-1)
        allSobel.append(torch.tensor(sobelFunc(img_np)).unsqueeze(0).unsqueeze(0))
    allSobel = torch.cat(allSobel,dim=0).to(origImgBatch.device)

    allSobel = (allSobel-allSobel.min())/(allSobel.max()-allSobel.min())

    if nms:
        simMap = computeNMS(1-allSobel)
    else:
        simMap = 1-allSobel

    return simMap

def computeORB(origImgBatch,nbPoints):
    kpsCoord_list = []
    for i in range(len(origImgBatch)):
        img_np= origImgBatch[i].detach().cpu().permute(1,2,0).numpy()
        kps, _ = cv2.ORB_create(nfeatures=1500).detectAndCompute(img_np, None)
        if len(kps) > 0:
            kpsCoord = np.concatenate([np.concatenate((np.array(i)[np.newaxis,np.newaxis],np.array(kp.pt)[::-1][np.newaxis]),axis=-1) for kp in kps],axis=0)
            kpsCoord_list.append(torch.tensor(kpsCoord))
    if len(kpsCoord_list):
        kpsCoord = torch.cat(kpsCoord_list,dim=0)
    else:
        kpsCoord = torch.zeros(0,3)
    return kpsCoord.to(origImgBatch.device).long()

def computeCanny(origImgBatch,sigma):
    edgeTensBatch = []
    for i in range(len(origImgBatch)):
        edges_np = skimage.feature.canny(origImgBatch[i].detach().cpu().mean(dim=0).numpy(),sigma=sigma)
        edges_tens = torch.tensor(edges_np).unsqueeze(0)
        edgeTensBatch.append(edges_tens)
    edgeTensBatch = torch.cat(edgeTensBatch,dim=0)
    edgesCoord = torch.nonzero(edgeTensBatch).to(origImgBatch.device)
    return edgesCoord

def computeEdges(origImgBatch,sigma,pts_nb,featMapSize,orbedge,nbPoints):

    kpFunc = (lambda x:computeCanny(x,sigma)) if not orbedge else (lambda x:computeORB(x,nbPoints))

    edgesCoord = kpFunc(origImgBatch)

    selEdgeCoordBatch = []
    for i in range(len(origImgBatch)):
        selEdgeCoord = edgesCoord[edgesCoord[:,0] == i]
        if selEdgeCoord.size(0) > 0:
            if selEdgeCoord.size(0) <= pts_nb:
                res = torch.randint(selEdgeCoord.size(0),size=(pts_nb,))
            else:
                res = torch_geometric.nn.fps(selEdgeCoord[:,1:].float(),ratio=pts_nb/selEdgeCoord.size(0))

            selEdgeCoordBatch.append(selEdgeCoord[res][:,1:].unsqueeze(0))
        else:
            abs = torch.randint(origImgBatch.size(-1),size=(pts_nb,)).unsqueeze(1)
            ord = torch.randint(origImgBatch.size(-2),size=(pts_nb,)).unsqueeze(1)
            coord = torch.cat((abs,ord),dim=-1).to(origImgBatch.device)

            selEdgeCoordBatch.append(coord.unsqueeze(0))

    edges = torch.cat(selEdgeCoordBatch,dim=0).float()
    edges /= (origImgBatch.size(-1)/featMapSize)
    edges = torch.clamp(edges,0,featMapSize-1).to(origImgBatch.device)

    return edges

def compositeShiftFeat(coord,features):

    if coord[0] != 0:
        if coord[0] > 0:
            shiftFeatV,shiftMaskV = shiftFeat("top",features,coord[0])
        else:
            shiftFeatV,shiftMaskV = shiftFeat("bot",features,-coord[0])
    else:
        shiftFeatV,shiftMaskV = shiftFeat("none",features)

    if coord[1] != 0:
        if coord[1] > 0:
            shiftFeatH,shiftMaskH = shiftFeat("right",shiftFeatV,coord[1])
        else:
            shiftFeatH,shiftMaskH = shiftFeat("left",shiftFeatV,-coord[1])
    else:
        shiftFeatH,shiftMaskH = shiftFeat("none",shiftFeatV)

    return shiftFeatH,shiftMaskH*shiftMaskV

def shiftFeat(where,features,dilation=None):

    mask = torch.ones_like(features)

    if where=="left":
        #x,y = 0,1
        padd = features[:,:,:,-1:].expand(-1,-1,-1,dilation)
        paddMask = torch.zeros((features.size(0),features.size(1),features.size(2),dilation)).to(features.device)+0.0001
        featuresShift = torch.cat((features[:,:,:,dilation:],padd),dim=-1)
        maskShift = torch.cat((mask[:,:,:,dilation:],paddMask),dim=-1)
    elif where=="right":
        #x,y= 2,1
        padd = features[:,:,:,:1].expand(-1,-1,-1,dilation)
        paddMask = torch.zeros((features.size(0),features.size(1),features.size(2),dilation)).to(features.device)+0.0001
        featuresShift = torch.cat((padd,features[:,:,:,:-dilation]),dim=-1)
        maskShift = torch.cat((paddMask,mask[:,:,:,:-dilation]),dim=-1)
    elif where=="bot":
        #x,y = 1,0
        padd = features[:,:,:1].expand(-1,-1,dilation,-1)
        paddMask = torch.zeros((features.size(0),features.size(1),dilation,features.size(3))).to(features.device)+0.0001
        featuresShift = torch.cat((padd,features[:,:,:-dilation,:]),dim=-2)
        maskShift = torch.cat((paddMask,mask[:,:,:-dilation,:]),dim=-2)
    elif where=="top":
        #x,y = 1,2
        padd = features[:,:,-1:].expand(-1,-1,dilation,-1)
        paddMask = torch.zeros((features.size(0),features.size(1),dilation,features.size(3))).to(features.device)+0.0001
        featuresShift = torch.cat((features[:,:,dilation:,:],padd),dim=-2)
        maskShift = torch.cat((mask[:,:,dilation:,:],paddMask),dim=-2)
    elif where=="none":
        featuresShift = features
        maskShift = mask
    else:
        raise ValueError("Unkown position")

    maskShift = maskShift.mean(dim=1,keepdim=True)
    return featuresShift,maskShift

def applyDiffKer_CosSimi(direction,features,dilation=1):
    origFeatSize = features.size()
    featNb = origFeatSize[1]

    if type(direction) is str:
        if direction == "horizontal":
            featuresShift1,maskShift1 = shiftFeat("right",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("left",features,dilation)
        elif direction == "vertical":
            featuresShift1,maskShift1 = shiftFeat("top",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("bot",features,dilation)
        elif direction == "top":
            featuresShift1,maskShift1 = shiftFeat("top",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "bot":
            featuresShift1,maskShift1 = shiftFeat("bot",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "left":
            featuresShift1,maskShift1 = shiftFeat("left",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "right":
            featuresShift1,maskShift1 = shiftFeat("right",features,dilation)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "none":
            featuresShift1,maskShift1 = shiftFeat("none",features)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        else:
            raise ValueError("Unknown direction : ",direction)
    else:
        featuresShift1,maskShift1 = compositeShiftFeat(direction,features)
        featuresShift2,maskShift2 = shiftFeat("none",features)

    sim = (featuresShift1*featuresShift2*maskShift1*maskShift2).sum(dim=1,keepdim=True)
    sim /= torch.sqrt(torch.pow(maskShift1*featuresShift1,2).sum(dim=1,keepdim=True))*torch.sqrt(torch.pow(maskShift2*featuresShift2,2).sum(dim=1,keepdim=True))

    return sim,featuresShift1,featuresShift2,maskShift1,maskShift2

def computeTotalSim(features,dilation):
    horizDiff,_,_,_,_ = applyDiffKer_CosSimi("horizontal",features,dilation)
    vertiDiff,_,_,_,_ = applyDiffKer_CosSimi("vertical",features,dilation)
    totalDiff = (horizDiff + vertiDiff)/2
    return totalDiff

class PatchSimCNN(torch.nn.Module):
    def __init__(self,resType,pretr,nbGroup,patchSize,neiSimRefin,featMod=None,nms=False,**kwargs):
        super(PatchSimCNN,self).__init__()

        if featMod is None:
            self.featMod = buildFeatModel(resType, pretr, True, False,**kwargs)
        else:
            self.featMod = featMod
        self.nbGroup = nbGroup

        self.neiSimRefin = neiSimRefin
        if not neiSimRefin is None:
            self.refiner = NeighSim(neiSimRefin["cuda"],nbGroup,neiSimRefin["nbIter"],neiSimRefin["softmax"],neiSimRefin["softmax_fact"],\
                                    neiSimRefin["weightByNeigSim"],neiSimRefin["neighRadius"])
        else:
            self.refiner = None

        self.patchSize = patchSize
        self.nms = nms

    def forward(self,data,returnLayer):

        startTime = time.time()
        patch = data.unfold(2, self.patchSize, self.patchSize).unfold(3, self.patchSize, self.patchSize).permute(0,2,3,1,4,5)
        origPatchSize = patch.size()
        patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4),patch.size(5))
        featVolume = self.featMod(patch,returnLayer)["x"]
        origFeatVolSize = featVolume.size()
        featVolume = featVolume.unfold(1, featVolume.size(1)//self.nbGroup, featVolume.size(1)//self.nbGroup).permute(0,1,4,2,3)
        featVolume = featVolume.view(featVolume.size(0)*featVolume.size(1),featVolume.size(2),featVolume.size(3),featVolume.size(4))

        featVolume = featVolume.reshape(featVolume.size(0),featVolume.size(1),featVolume.size(2)*featVolume.size(3))
        feat = featVolume.sum(dim=-1)

        feat = feat.view(origFeatVolSize[0],self.nbGroup,feat.size(-1))
        feat = feat.view(data.size(0),-1,self.nbGroup,feat.size(2))
        feat = feat.permute(0,2,1,3)

        origFeatSize = feat.size()

        #N x NbGroup x nbPatch x C
        feat = feat.reshape(feat.size(0)*feat.size(1),feat.size(2),feat.size(3))
        # (NxNbGroup) x nbPatch x C
        feat = feat.permute(0,2,1)
        # (NxNbGroup) x C x nbPatch
        feat = feat.unsqueeze(-1)
        # (NxNbGroup) x C x nbPatch x 1
        feat = feat.reshape(feat.size(0),feat.size(1),int(np.sqrt(feat.size(2))),int(np.sqrt(feat.size(2))))
        # (NxNbGroup) x C x sqrt(nbPatch) x sqrt(nbPatch)

        if self.refiner is None:
            simMap = computeTotalSim(feat,1).unsqueeze(1)
            # (NxNbGroup) x 1 x 1 x sqrt(nbPatch) x sqrt(nbPatch)
            simMap = simMap.reshape(origFeatSize[0],origFeatSize[1],simMap.size(2),simMap.size(3),simMap.size(4))
            # N x NbGroup x 1 x sqrt(nbPatch) x sqrt(nbPatch)
            simMap = simMap.mean(dim=1)
            # N x 1 x sqrt(nbPatch) x sqrt(nbPatch)
        else:
            startTime = time.time()
            simMap = self.refiner(feat)
            # N x 1 x sqrt(nbPatch) x sqrt(nbPatch)

        if self.nms:
            startTime = time.time()
            simMap = computeNMS(simMap)

        return simMap

def computeNMS(simMap):
    simMapMinH = -F.max_pool2d(-simMap,(1,3), stride=(1,1), padding=(0,1))
    simMapMinV = -F.max_pool2d(-simMap,(3,1), stride=(1,1), padding=(1,0))
    minima = ((simMap == simMapMinH) | (simMap == simMapMinV))
    simMap = 1-((1-simMap)*minima)
    return simMap

def computeNeighborsCoord(neighRadius):

    coord = torch.arange(neighRadius*2+1)-neighRadius

    y = coord.unsqueeze(1).expand(coord.size(0),coord.size(0)).unsqueeze(-1)
    x = coord.unsqueeze(0).expand(coord.size(0),coord.size(0)).unsqueeze(-1)

    coord = torch.cat((x,y),dim=-1).view(coord.size(0)*coord.size(0),2)
    coord = coord[~((coord[:,0] == 0)*(coord[:,1] == 0))]

    return coord

class NeighSim(torch.nn.Module):
    def __init__(self,cuda,groupNb,nbIter,softmax,softmax_fact,weightByNeigSim,neighRadius):
        super(NeighSim,self).__init__()

        self.directions = computeNeighborsCoord(neighRadius)

        self.sumKer = torch.ones((1,len(self.directions),1,1))
        self.sumKer = self.sumKer.cuda() if cuda else self.sumKer
        self.groupNb = groupNb
        self.nbIter = nbIter
        self.softmax = softmax
        self.softmax_fact = softmax_fact
        self.weightByNeigSim = weightByNeigSim

        if not self.weightByNeigSim and self.softmax:
            raise ValueError("Can't have weightByNeigSim=False and softmax=True")

    def forward(self,features):

        simMap = computeTotalSim(features,1)
        #simMapList = [simMap]
        for j in range(self.nbIter):

            allSim = []
            allFeatShift = []
            allPondFeatShift = []
            for direction in self.directions:
                if self.weightByNeigSim:
                    sim,featuresShift1,_,maskShift1,_ = applyDiffKer_CosSimi(direction,features,1)
                    allSim.append(sim*maskShift1)
                else:
                    featuresShift1,maskShift1 = shiftFeat(direction,features,1)
                    allSim.append(maskShift1)

                allFeatShift.append(featuresShift1*maskShift1)

            allSim = torch.cat(allSim,dim=1)

            if self.weightByNeigSim and self.softmax:
                allSim = torch.softmax(self.softmax_fact*allSim,dim=1)

            for i in range(len(self.directions)):
                sim = allSim[:,i:i+1]
                featuresShift1 = allFeatShift[i]
                allPondFeatShift.append((sim*featuresShift1).unsqueeze(1))

            newFeatures = torch.cat(allPondFeatShift,dim=1).sum(dim=1)
            simSum = torch.nn.functional.conv2d(allSim,self.sumKer.to(allSim.device))
            newFeatures /= simSum
            features = newFeatures

        simMap = computeTotalSim(features,1)

        simMap = simMap.unsqueeze(1)

        simMap = simMap.reshape(simMap.size(0)//self.groupNb,self.groupNb,1,simMap.size(3),simMap.size(4))
        # N x NbGroup x 1 x sqrt(nbPatch) x sqrt(nbPatch)
        simMap = simMap.mean(dim=1)
        # N x 1 x sqrt(nbPatch) x sqrt(nbPatch)


        return simMap

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


class PointNet2(SecondModel):

    def __init__(self, cuda, classNb, nbFeat, nbFeatAux,pn_model, topk=False, reinfExct=False, \
                 point_nb=256,encoderChan=1,topk_fps=False,
                 topk_fps_nb_pts=64, auxModel=False,auxModelTopk=False,predictScore=False,imageAttBlockNb=1,
                 use_baseline=False,topk_euclinorm=True,reinf_linear_only=False, pn_clustering=False,\
                 cannyedge=False,cannyedge_sigma=2,orbedge=False,sobel=False,patchsim=False,patchsim_patchsize=5,patchsim_groupNb=4,patchsim_neiSimRefin=None,\
                 nms=False,no_feat=False,patchsim_mod=None,norm_points=False,topk_sagpool=False,topk_sagpool_drop=False,topk_sagpool_drop_ratio=0.5,\
                 pureTextModel=False,pureTextPtsNbFact=4,smoothFeat=False,smoothFeatStartSize=10):

        super(PointNet2, self).__init__(nbFeat, classNb)

        self.point_nb = point_nb
        if topk:
            self.pointExtr = TopkPointExtractor(cuda, nbFeat,point_nb, encoderChan, \
                                                topk_fps, topk_fps_nb_pts,auxModel,auxModelTopk, \
                                                topk_euclinorm,predictScore,imageAttBlockNb,\
                                                cannyedge,cannyedge_sigma,orbedge,sobel,patchsim,patchsim_patchsize,patchsim_groupNb,patchsim_neiSimRefin,nms,no_feat,\
                                                patchsim_mod,norm_points,topk_sagpool,topk_sagpool_drop,topk_sagpool_drop_ratio,False)
        elif reinfExct:
            self.pointExtr = ReinforcePointExtractor(cuda, nbFeat,point_nb,encoderChan,predictScore, use_baseline, reinf_linear_only)
        else:
            raise ValueError("Please set topk or reinfExct to True")

        self.pn2 = pn_model
        self.clustering = pn_clustering
        if self.clustering:
            self.cluster_model = ClusterModel(nb_cluster=4)

        self.auxModel = auxModel
        if auxModel:
            self.auxModel = nn.Linear(nbFeatAux, classNb)

        if pureTextModel:

            self.textModPtsNb = point_nb//pureTextPtsNbFact
            self.pureText_pointExtr = TopkPointExtractor(cuda, nbFeat,self.textModPtsNb, encoderChan, \
                                                False, point_nb,False,False, \
                                                True,False,imageAttBlockNb, \
                                                False,cannyedge_sigma,False,False,False,patchsim_patchsize,patchsim_groupNb,False,False,False,\
                                                False,norm_points,False,False,topk_sagpool_drop_ratio,False)
        else:
            self.pureText_pointExtr = None



        self.smoothFeat=smoothFeat
        self.smoothFeatStartSize=smoothFeatStartSize
        if smoothFeatStartSize % 2 == 0:
            raise ValueError("Initial smoohting kernel size must be odd.")

        if smoothFeat:
            self.smoothKer = torch.ones(nbFeat,1,1,1)
            self.smoothKer = self.smoothKer.to("cuda") if cuda else self.smoothKer


    def setSmoothKer(self,newSize):
        oldSize = self.smoothKer
        device=self.smoothKer.device
        self.smoothKer = torch.ones(oldSize.size(0),oldSize.size(1),newSize,newSize)
        self.smoothKer = self.smoothKer.to(device)

    def forward(self, visResDict,**kwargs):

        featureMaps = visResDict["x"]

        if self.smoothFeat:
            featureMaps = F.conv2d(featureMaps,self.smoothKer.to(featureMaps.device),groups=self.smoothKer.size(0),padding=self.smoothKer.size(-1)//2)

        retDict = self.pointExtr(featureMaps,**kwargs)

        if self.clustering:
            retDict = self.cluster_model(retDict)

        if self.pureText_pointExtr:
            puretext_retDict = self.pureText_pointExtr(featureMaps,pointFeaturesMap=retDict["featVolume"],x=retDict["prob_map"],**kwargs)

            retDict = merge(puretext_retDict,retDict,suffix="puretext")

            #Adding feature to indicate if point comes from texture border or center
            indicator = torch.ones(featureMaps.size(0)*self.textModPtsNb,1).to(featureMaps.device)
            puretext_retDict['pointfeatures'] = torch.cat((puretext_retDict['pointfeatures'],indicator),dim=-1)
            indicator = torch.ones(featureMaps.size(0)*self.point_nb,1).to(featureMaps.device)
            retDict['pointfeatures'] = torch.cat((retDict['pointfeatures'],-indicator),dim=-1)

            #Merging the two point clouds
            retDict['pointfeatures'] = torch.cat((retDict['pointfeatures'],puretext_retDict['pointfeatures']),dim=0)
            retDict['pos'] = torch.cat((retDict['pos'],puretext_retDict["pos"]),dim=0)
            retDict['batch'] = torch.cat((retDict['batch'],puretext_retDict["batch"]),dim=0)

            inds = retDict['batch'].argsort()
            retDict['pointfeatures'] = retDict['pointfeatures'][inds].contiguous()
            retDict['pos'] = retDict['pos'][inds].contiguous()
            retDict['batch'] = retDict['batch'][inds].contiguous()

        pred = self.pn2(retDict['pointfeatures'], retDict['pos'], retDict['batch'])
        retDict['pred'] = pred

        if self.auxModel:
            retDict["auxPred"] = self.auxModel(retDict['auxFeat'])

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

def computeEncChan(nbFeat,pn_enc_chan,pn_topk_no_feat,pn_topk_puretext):
    if pn_topk_no_feat:
        pointnetInputChannels = 0
    elif pn_enc_chan == 0:
        pointnetInputChannels = nbFeat
    else:
        pointnetInputChannels = pn_enc_chan

    if pn_topk_puretext:
        pointnetInputChannels += 1

    return pointnetInputChannels

def netBuilder(args):
    ############### Visual Model #######################
    if args.first_mod.find("resnet") != -1:

        if not args.multi_level_feat:
            nbFeat = getResnetFeat(args.first_mod, args.resnet_chan,args.deeplabv3_outchan)
        else:
            nbFeat = args.multi_level_feat_outchan

        if not args.resnet_simple_att:
            CNNconst = CNN2D
            kwargs = {}
        else:
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
                      "aux_model":args.aux_model}
            if args.resnet_simple_att_topk_enc_chan != -1:
                nbFeat = args.resnet_simple_att_topk_enc_chan

            nbFeatAux = nbFeat
            if type(args.resnet_simple_att_topk_pxls_nb) is list and len(args.resnet_simple_att_topk_pxls_nb) > 1:
                nbFeat *= len(args.resnet_simple_att_topk_pxls_nb)

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

    ############### Temporal Model #######################
    if args.second_mod == "linear":
        secondModel = LinearSecondModel(nbFeat,nbFeatAux, args.class_nb, args.dropout,args.aux_model)
    elif args.second_mod == "pointnet2" or args.second_mod == "edgenet":

        pointnetInputChannels = computeEncChan(nbFeat,args.pn_enc_chan,args.pn_topk_no_feat,args.pn_topk_puretext)

        if args.second_mod == "pointnet2":
            const = pointnet2.Net
        else:
            const = pointnet2.EdgeNet

        pn_model = const(num_classes=args.class_nb,input_channels=pointnetInputChannels)

        if args.pn_patchsim_neiref:
            neiSimRefinDict = {"cuda":args.cuda,"nbIter":args.pn_patchsim_neiref_nbiter,"softmax":args.pn_patchsim_neiref_softm,\
                            "softmax_fact":args.pn_patchsim_neiref_softmfact,"weightByNeigSim":args.pn_patchsim_neiref_weibysim,\
                            "neighRadius":args.pn_patchsim_neiref_neirad}

        else:
            neiSimRefinDict = None

        secondModel = PointNet2(args.cuda, args.class_nb, nbFeat=nbFeat, nbFeatAux=nbFeatAux,pn_model=pn_model,\
                                topk=args.pn_topk, reinfExct=args.pn_reinf, point_nb=args.pn_point_nb,
                                encoderChan=args.pn_enc_chan, \
                                topk_fps=args.pn_topk_farthest_pts_sampling, topk_fps_nb_pts=args.pn_topk_fps_nb_points,
                                auxModel=args.aux_model,auxModelTopk=args.pn_aux_model_topk,predictScore=args.pn_predict_score,
                                imageAttBlockNb=args.pn_image_attention_block_nb,
                                use_baseline=args.pn_reinf_use_baseline, \
                                topk_euclinorm=args.pn_topk_euclinorm, reinf_linear_only=args.pn_train_reinf_linear_only,
                                pn_clustering=args.pn_clustering,\
                                cannyedge=args.pn_cannyedge,cannyedge_sigma=args.pn_cannyedge_sigma,orbedge=args.pn_orbedge,sobel=args.pn_sobel,\
                                patchsim=args.pn_patchsim,patchsim_patchsize=args.pn_patchsim_patchsize,patchsim_groupNb=args.pn_patchsim_groupnb,patchsim_neiSimRefin=neiSimRefinDict,\
                                nms=args.pn_nms,no_feat=args.pn_topk_no_feat,patchsim_mod=None if args.pn_patchsim_randmod else firstModel.featMod,\
                                norm_points=args.norm_points,topk_sagpool=args.pn_topk_sagpool,topk_sagpool_drop=args.pn_topk_sagpool_drop,\
                                topk_sagpool_drop_ratio=args.pn_topk_sagpool_drop_ratio,\
                                pureTextModel=args.pn_topk_puretext,pureTextPtsNbFact=args.pn_topk_puretext_pts_nb_fact,\
                                smoothFeat=args.smooth_features,smoothFeatStartSize=args.smooth_features_start_size)
    else:
        raise ValueError("Unknown temporal model type : ", args.second_mod)

    ############### Whole Model ##########################

    net = Model(firstModel, secondModel,zoom=args.zoom,passOrigImage=args.pn_cannyedge or args.pn_patchsim or args.pn_orbedge or args.pn_sobel,reducedImgSize=args.reduced_img_size)

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

    argreader.parser.add_argument('--pn_topk', type=args.str2bool, metavar='BOOL',
                                  help='To feed the pointnet model with points extracted using torch.topk and not a direct coordinate predictor. Ignored if the model \
                        doesn\'t use pointnet.')
    argreader.parser.add_argument('--pn_reinf', type=args.str2bool, metavar='BOOL',
                                  help='To feed the pointnet model with points extracted using reinforcement learning. Ignored if the model \
                            doesn\'t use pointnet.')
    argreader.parser.add_argument('--pn_predict_score', type=args.str2bool, metavar='BOOL',
                                  help='To use linear layer to compute probabilities for the reinforcement model')
    argreader.parser.add_argument('--pn_reinf_use_baseline', type=args.str2bool, metavar='BOOL',
                                  help='To use linear layer to compute baseline for the reinforcement model training')
    argreader.parser.add_argument('--pn_train_reinf_linear_only', type=args.str2bool, metavar='BOOL',
                                  help='To prevent reinforcement loss to propagate in firstModel ')
    argreader.parser.add_argument('--pn_clustering', type=args.str2bool, metavar='BOOL',
                                  help='To introduce a clustering model between point extractor and pointNet network ')


    argreader.parser.add_argument('--pn_topk_farthest_pts_sampling', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To apply furthest point sampling when extracting points.')
    argreader.parser.add_argument('--pn_topk_fps_nb_points', type=int, metavar='INT',
                                  help='For the pointnet2 model. The number of point extracted by furthest point sampling.')

    argreader.parser.add_argument('--pn_topk_euclinorm', type=args.str2bool, metavar='BOOL',
                                  help='For the topk point net model. To use the euclidean norm to compute pixel importance instead of using their raw value \
                                  filtered by a Relu.')

    argreader.parser.add_argument('--zoom', type=args.str2bool, metavar='BOOL',
                                  help='To use with a model that generates points. To zoom on the part of the images where the points are focused an apply the model a second time on it.')

    argreader.parser.add_argument('--pn_point_nb', type=int, metavar='NB',
                                  help='For the topk point net model. This is the number of point extracted for each image.')

    argreader.parser.add_argument('--pn_enc_chan', type=int, metavar='NB',
                                  help='For the topk point net model. This is the number of output channel of the encoder')

    argreader.parser.add_argument('--aux_model', type=args.str2bool, metavar='INT',
                                  help='To train an auxilliary model that will apply average pooling and a dense layer on the feature map\
                        to make a prediction alongside pointnet\'s one.')
    argreader.parser.add_argument('--pn_aux_model_topk', type=args.str2bool, metavar='INT',
                                  help='To make the aux model a topk one.')



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


    argreader.parser.add_argument('--pn_cannyedge', type=args.str2bool, metavar='BOOL',
                                  help='To use canny edge to extract points.')
    argreader.parser.add_argument('--pn_cannyedge_sigma', type=float, metavar='FLOAT',
                                  help='The sigma hyper-parameter of the canny edge detection.')
    argreader.parser.add_argument('--pn_orbedge', type=args.str2bool, metavar='BOOL',
                                  help='To use ORB to extract points.')
    argreader.parser.add_argument('--pn_sobel', type=args.str2bool, metavar='BOOL',
                                  help='To use sobel filtering to extract points.')

    argreader.parser.add_argument('--pn_patchsim', type=args.str2bool, metavar='BOOL',
                                  help='To use patch similarity to extract points.')
    argreader.parser.add_argument('--pn_patchsim_patchsize', type=int, metavar='BOOL',
                                  help='The patch size.')
    argreader.parser.add_argument('--pn_patchsim_groupnb', type=int, metavar='BOOL',
                                  help='The number of groups of features.')

    argreader.parser.add_argument('--pn_patchsim_neiref', type=args.str2bool, metavar='BOOL',
                                  help='When using patch similarity, to refine results using neighbor similarity')
    argreader.parser.add_argument('--pn_patchsim_neiref_nbiter', type=int, metavar='BOOL',
                                  help='The number of iterations for neihbor refining')
    argreader.parser.add_argument('--pn_patchsim_neiref_softm', type=args.str2bool, metavar='BOOL',
                                  help='Whether to use softmax for neihbor refining')
    argreader.parser.add_argument('--pn_patchsim_neiref_softmfact', type=int, metavar='BOOL',
                                  help='The softmax temperature (lower give smoother) for neihbor refining')
    argreader.parser.add_argument('--pn_patchsim_neiref_weibysim', type=args.str2bool, metavar='BOOL',
                                  help='Whether to weight neighbors using similarity for neihbor refining')
    argreader.parser.add_argument('--pn_patchsim_neiref_neirad', type=int, metavar='BOOL',
                                  help='The radius of the neighborhood for neihbor refining')
    argreader.parser.add_argument('--pn_nms', type=args.str2bool, metavar='BOOL',
                                  help='To apply non-min suppresion on the result of patch sim or sobel.')
    argreader.parser.add_argument('--pn_image_attention_block_nb', type=int, metavar='INT',
                                  help='The number of block to use to predict pixel scores.')



    argreader.parser.add_argument('--reduced_img_size', type=int, metavar='BOOL',
                                  help="The size at which the image is reduced at the begining of the process")
    argreader.parser.add_argument('--pn_topk_no_feat', type=args.str2bool, metavar='BOOL',
                                  help="Set to True to not pass the point features to the pointnet model and only the point position")
    argreader.parser.add_argument('--pn_patchsim_randmod', type=args.str2bool, metavar='BOOL',
                                  help="To use a random resnet9 to extract key points instead of the feature extractor being trained.")

    argreader.parser.add_argument('--norm_points', type=args.str2bool, metavar='BOOL',
                                  help="To normalize the points before passing them to pointnet")

    argreader.parser.add_argument('--pn_topk_sagpool', type=args.str2bool, metavar='BOOL',
                                  help="To use SAG pooling to reduce the number of points that will be passed to pointnet.")
    argreader.parser.add_argument('--pn_topk_sagpool_drop', type=args.str2bool, metavar='BOOL',
                                  help="To drop points which given weights are low.")
    argreader.parser.add_argument('--pn_topk_sagpool_drop_ratio', type=float, metavar='BOOL',
                                  help="The ratio of points dropped.")

    argreader.parser.add_argument('--pn_topk_puretext', type=args.str2bool, metavar='BOOL',
                                  help="To use a second topk model that will takes as input the bottom K pixels intead of the topk")
    argreader.parser.add_argument('--pn_topk_puretext_pts_nb_fact', type=int, metavar='INT',
                                  help="The factor between the number of point first selected by pure texture model and the number of points extracted by the topk model.")

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

    argreader.parser.add_argument('--smooth_features', type=args.str2bool, metavar='BOOL',
                                  help="To smooth features using a simple mean.")
    argreader.parser.add_argument('--smooth_features_sched_step', type=int, metavar='BOOL',
                                  help="The number of epoch to wait before reducing the size of the kernel used to smooth")
    argreader.parser.add_argument('--smooth_features_start_size', type=int, metavar='BOOL',
                                  help="The initial size of the kernel used to smooth.")

    return argreader
