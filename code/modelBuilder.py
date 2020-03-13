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
import torch_geometric

import matplotlib.pyplot as plt
plt.switch_backend('agg')

_EPSILON = 10e-7

from  torch.nn.modules.upsampling import Upsample


def buildFeatModel(featModelName, pretrainedFeatMod, featMap=True, bigMaps=False, layerSizeReduce=False, stride=2,
                   dilation=1, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("deeplabv3") != -1:
        featModel = deeplab._segm_resnet("deeplabv3", featModelName[featModelName.find("resnet"):], \
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

    def __init__(self, firstModel, secondModel,zoom=False):
        super(Model, self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.zoom = zoom

    def forward(self, origImg):
        visResDict = self.firstModel(origImg)
        resDict = self.secondModel(visResDict["x"])

        resDict = self.merge(visResDict,resDict)

        if self.zoom:
            croppedImg,xMinCr,xMaxCr,yMinCr,yMaxCr = self.computeZoom(origImg,visResDict["x"],resDict)
            visResDict_zoom = self.firstModel(croppedImg)
            resDict_zoom = self.secondModel(visResDict_zoom["x"])
            resDict_zoom = self.merge(visResDict_zoom,resDict_zoom)
            resDict = self.merge(resDict_zoom,resDict,"zoom")

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
        #theta = torch.zeros((x.size(0),2,3))
        #theta[:,0,0] = (4*stds.max(dim=-1)[0])/imgSize[0]
        #theta[:,1,1] = (4*stds.max(dim=-1)[0])/imgSize[0]
        #theta[:,:,2] = (means - imgSize.unsqueeze(0)/2)/imgSize.unsqueeze(0)

        flowField = F.affine_grid(theta, origImg.size(),align_corners=False).to(x.device)
        croppedImg = F.grid_sample(maskedImage, flowField,align_corners=False,padding_mode='border')

        return croppedImg,xMin,xMax,yMin,yMax

    def merge(self,dictA,dictB,suffix=""):
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

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, **kwargs):
        super(CNN2D, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)

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


class CNN2D_simpleAttention(FirstModel):

    def __init__(self, featModelName, pretrainedFeatMod=True, featMap=True, bigMaps=False, chan=64, attBlockNb=2,
                 attChan=16, \
                 topk=False, topk_pxls_nb=256, topk_enc_chan=64,**kwargs):

        super(CNN2D_simpleAttention, self).__init__(featModelName, pretrainedFeatMod, featMap, bigMaps, **kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        inFeat = getResnetFeat(featModelName, chan)

        attention = []
        for i in range(attBlockNb):
            attention.append(resnet.BasicBlock(inFeat, inFeat))
        attention.append(resnet.conv1x1(inFeat, 1))

        self.topk = topk
        if not topk:
            self.attention = nn.Sequential(*attention)
            self.topk_pxls_nb = None
        else:
            self.attention = None
            self.topk_pxls_nb = topk_pxls_nb

        self.topk_enc_chan = topk_enc_chan
        if self.topk_enc_chan != -1:
            self.conv1x1 = nn.Conv2d(inFeat,self.topk_enc_chan,1)

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        features = self.featMod(x)

        if self.topk_enc_chan != -1:
            features = self.conv1x1(features)

        retDict = {}

        if not self.topk:
            spatialWeights = torch.sigmoid(self.attention(features))
            features = spatialWeights * features
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)
        else:
            spatialWeights = torch.zeros((features.size(0), 1, features.size(2), features.size(3)))
            # Compute the mean between the k most active pixels
            featNorm = torch.pow(features, 2).sum(dim=1, keepdim=True)
            flatFeatNorm = featNorm.view(featNorm.size(0), -1)
            flatVals, flatInds = torch.topk(flatFeatNorm, self.topk_pxls_nb, dim=-1, largest=True)
            abs, ord = (flatInds % featNorm.shape[-1], flatInds // featNorm.shape[-1])
            featureList = mapToList(features, abs, ord)
            features = featureList.mean(dim=1)
            indices = tuple([torch.arange(spatialWeights.size(0), dtype=torch.long).unsqueeze(1).unsqueeze(1),
                             torch.arange(spatialWeights.size(1), dtype=torch.long).unsqueeze(1).unsqueeze(0),
                             ord.long().unsqueeze(1), abs.long().unsqueeze(1)])
            spatialWeights[indices] = 1

            depth = torch.zeros(abs.size(0), abs.size(1), 1).to(x.device)
            retDict['points'] = torch.cat((abs.unsqueeze(2).float(), ord.unsqueeze(2).float(), depth, featureList), dim=-1).float()

        retDict["x"] = features
        retDict["attMaps"] = spatialWeights

        return retDict

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super(SecondModel, self).__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError


class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbClass, dropout):
        super(LinearSecondModel, self).__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat, self.nbClass)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        if len(x.size()) == 4:
            x = self.avgpool(x).squeeze(-1).squeeze(-1)

        # N x D
        x = self.dropout(x)
        x = self.linLay(x)
        # N x classNb

        # N x classNb
        return {"pred": x}


class Identity(SecondModel):

    def __init__(self, nbFeat, nbClass):
        super(Identity, self).__init__(nbFeat, nbClass)

    def forward(self, x, batchSize):
        return {"pred": x}


class DirectPointExtractor(nn.Module):

    def __init__(self, point_nb, nbFeat):
        super(DirectPointExtractor, self).__init__()
        self.conv1x1 = nn.Conv2d(nbFeat, 32, kernel_size=1, stride=1)
        self.size_red = nn.AdaptiveAvgPool2d((8, 8))
        self.dense = nn.Linear(8 * 8 * 32, point_nb * 2)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.size_red(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)
        points = x.view(x.size(0), x.size(1) // 2, 2)
        points = torch.cat((points, torch.zeros(points.size(0), points.size(1), 1).to(x.device)), dim=-1)

        return {"points": points,
                "batch": torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(
                    -1).to(points.device),
                "pos": points.reshape(points.size(0) * points.size(1), points.size(2)),
                "pointfeatures": None}


class TopkPointExtractor(nn.Module):

    def __init__(self, cuda, nbFeat, softCoord, softCoord_kerSize, softCoord_secOrder, point_nb, reconst, encoderChan, \
                 predictDepth, softcoord_shiftpred, furthestPointSampling, furthestPointSampling_nb_pts, dropout,
                 auxModel, \
                 topkRandSamp, topkRSUnifWeight, topk_euclinorm,hasLinearProb):

        super(TopkPointExtractor, self).__init__()

        self.conv1x1 = nn.Conv2d(nbFeat, encoderChan, kernel_size=1, stride=1)
        self.point_extracted = point_nb
        self.softCoord = softCoord
        self.softCoord_kerSize = softCoord_kerSize
        self.softCoord_secOrder = softCoord_secOrder
        self.softcoord_shiftpred = softcoord_shiftpred
        self.furthestPointSampling = furthestPointSampling
        self.furthestPointSampling_nb_pts = furthestPointSampling_nb_pts
        self.dropout = dropout
        self.topk_euclinorm = topk_euclinorm

        self.reconst = reconst
        if reconst:
            self.decoder = nn.Sequential(resnet.conv1x1(encoderChan, 3, stride=1), \
                                         resnet.resnet4(pretrained=True, chan=64, featMap=True, layerSizeReduce=False,
                                                        preLayerSizeReduce=False, stride=1), \
                                         resnet.conv1x1(64, 3, stride=1))

        self.predictDepth = predictDepth
        if predictDepth:
            self.conv1x1_depth = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1)

        if self.softcoord_shiftpred and self.softCoord:
            raise ValueError("softcoord_shiftpred and softCoord can't be true at the same time.")

        if self.softcoord_shiftpred:
            self.shiftpred_x, self.shiftpred_y = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1), nn.Conv2d(nbFeat, 1,
                                                                                                          kernel_size=1,
                                                                                                          stride=1)

        if self.softCoord:

            if self.softCoord_kerSize % 2 == 0:
                raise ValueError("Kernel size of soft coordinate extractor must not be a multiple of 2.")

            ordKer = (torch.arange(self.softCoord_kerSize) - self.softCoord_kerSize // 2).unsqueeze(1).unsqueeze(
                0).unsqueeze(0).expand(1, 1, self.softCoord_kerSize, self.softCoord_kerSize).float()
            absKer = (torch.arange(self.softCoord_kerSize) - self.softCoord_kerSize // 2).unsqueeze(0).unsqueeze(
                0).unsqueeze(0).expand(1, 1, self.softCoord_kerSize, self.softCoord_kerSize).float()
            self.ordKer, self.absKer = (ordKer.cuda(), absKer.cuda()) if cuda else (ordKer, absKer)

            self.spatialWeightKer = self.softCoord_kerSize - (torch.abs(self.ordKer) + torch.abs(self.absKer))
            if self.softCoord_secOrder:
                self.spatialWeightKer = self.spatialWeightKer * self.spatialWeightKer

            self.ordKerDict, self.absKerDict, self.spatialWeightKerDict = {}, {}, {}

        self.auxModel = auxModel
        self.topkRandSamp = topkRandSamp
        self.topkRSUnifWeight = topkRSUnifWeight
        self.hasLinearProb = hasLinearProb
        self.linearProb = nn.Conv2d(encoderChan, 1, kernel_size=1, stride=1)

    def forward(self, featureMaps):

        # Because of zero padding, the border are very active, so we remove it.
        featureMaps = featureMaps[:, :, 3:-3, 3:-3]

        pointFeaturesMap = self.conv1x1(featureMaps)

        if self.topk_euclinorm:
            x = torch.pow(pointFeaturesMap, 2).sum(dim=1, keepdim=True)
        elif self.hasLinearProb:
            x = torch.sigmoid(self.linearProb(pointFeaturesMap))
        else:
            x = F.relu(pointFeaturesMap).sum(dim=1, keepdim=True)

        flatX = x.view(x.size(0), -1)
        if (not self.topkRandSamp) or (self.topkRandSamp and not self.training):
            _, flatInds = torch.topk(flatX, self.point_extracted, dim=-1, largest=True)
        else:
            probs = self.topkRSUnifWeight * (1.0 / flatX.size(1)) + (1 - self.topkRSUnifWeight) * flatX / \
                    flatX.max(dim=-1, keepdim=True)[0]
            flatInds = torch.distributions.categorical.Categorical(probs=probs).sample(
                torch.tensor([self.point_extracted]))
            flatInds = flatInds.permute(1, 0)

        abs, ord = (flatInds % x.shape[-1], flatInds // x.shape[-1])

        if self.furthestPointSampling:
            points = torch.cat((abs.unsqueeze(-1), ord.unsqueeze(-1)), dim=-1).float()

            exampleInds = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(
                -1).to(points.device)
            selectedPointInds = torch_geometric.nn.fps(points.view(-1, 2), exampleInds,
                                                       ratio=self.furthestPointSampling_nb_pts / abs.size(1))
            selectedPointInds = selectedPointInds.reshape(points.size(0), -1)
            selectedPointInds = selectedPointInds % abs.size(1)

            abs = abs[torch.arange(abs.size(0)).unsqueeze(1).long(), selectedPointInds.long()]
            ord = ord[torch.arange(ord.size(0)).unsqueeze(1).long(), selectedPointInds.long()]

        if self.dropout and self.training:
            idx = torch.randperm(abs.size(1))[:int((1 - self.dropout) * abs.size(1))].unsqueeze(0)
            abs = abs[torch.arange(abs.size(0)).unsqueeze(1).long(), idx]
            ord = ord[torch.arange(ord.size(0)).unsqueeze(1).long(), idx]

        retDict = {}

        if self.softCoord:
            abs, ord = self.fastSoftCoordRefiner(x, abs, ord, kerSize=self.softCoord_kerSize,
                                                 secondOrderSpatWeight=self.softCoord_secOrder)
        if self.reconst:
            flatVals = mapToList(x, abs, ord)
            sparseX = pointFeaturesMap * ((x - flatVals[:, -1].unsqueeze(1).unsqueeze(2).unsqueeze(3)) > 0).float()
            reconst = self.decoder(sparseX)
            # Keeping only one channel (the three channels are just copies)
            retDict['reconst'] = reconst[:, 0:1]

        pointFeat = mapToList(pointFeaturesMap, abs, ord)

        if self.predictDepth:
            depthMap = torch.tanh(self.conv1x1_depth(featureMaps)) * max(featureMaps.size(-2),
                                                                         featureMaps.size(-1)) // 10
            depth = mapToList(depthMap, abs, ord)
        else:
            depth = torch.zeros(abs.size(0), abs.size(1), 1).to(x.device)

        if self.softcoord_shiftpred:
            xShiftMap, yShiftMap = torch.tanh(self.shiftpred_x(featureMaps)), torch.tanh(self.shiftpred_y(featureMaps))
            xShift, yShift = mapToList(xShiftMap, abs, ord).squeeze(-1), mapToList(yShiftMap, abs, ord).squeeze(-1)
            abs = xShift + abs.float()
            ord = yShift + ord.float()

        abs, ord = abs.unsqueeze(-1).float(), ord.unsqueeze(-1).float()
        points = torch.cat((abs, ord, depth), dim=-1).float()
        retDict['points'] = torch.cat((abs, ord, depth, pointFeat), dim=-1).float()
        retDict["batch"] = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(
            -1).to(points.device)
        retDict["pos"] = points.reshape(points.size(0) * points.size(1), points.size(2))
        retDict["pointfeatures"] = pointFeat.reshape(pointFeat.size(0) * pointFeat.size(1), pointFeat.size(2))

        if self.auxModel:
            retDict["auxFeat"] = featureMaps.mean(dim=-1).mean(dim=-1)
        return retDict

    def fastSoftCoordRefiner(self, x, abs, ord, kerSize=5, secondOrderSpatWeight=False):

        self.updateDict(x.device)

        ordTens = torch.arange(x.size(-2)).unsqueeze(1).expand(x.size(-2), x.size(-1)).float().to(x.device)
        absTens = torch.arange(x.size(-1)).unsqueeze(0).expand(x.size(-2), x.size(-1)).float().to(x.device)

        unorm_ordShiftMean = F.conv2d(x, self.ordKerDict[x.device] * self.spatialWeightKerDict[x.device], stride=1,
                                      padding=kerSize // 2)
        unorm_absShiftMean = F.conv2d(x, self.absKerDict[x.device] * self.spatialWeightKerDict[x.device], stride=1,
                                      padding=kerSize // 2)

        weightSum = F.conv2d(x, self.spatialWeightKerDict[x.device], stride=1, padding=kerSize // 2)

        startOrd, endOrd = ord[0, 0].int().item() - kerSize // 2, ord[0, 0].int().item() + kerSize // 2 + 1
        startAbs, endAbs = abs[0, 0].int().item() - kerSize // 2, abs[0, 0].int().item() + kerSize // 2 + 1

        ordMean = ordTens.unsqueeze(0).unsqueeze(0) + unorm_ordShiftMean / (weightSum + 0.001)
        absMean = absTens.unsqueeze(0).unsqueeze(0) + unorm_absShiftMean / (weightSum + 0.001)

        ordList = ordMean[:, 0][torch.arange(x.size(0), dtype=torch.long).unsqueeze(1), ord.long(), abs.long()]
        absList = absMean[:, 0][torch.arange(x.size(0), dtype=torch.long).unsqueeze(1), ord.long(), abs.long()]

        return absList, ordList

    def updateDict(self, device):

        if not device in self.ordKerDict.keys():
            self.ordKerDict[device] = self.ordKer.to(device)
            self.absKerDict[device] = self.absKer.to(device)
            self.spatialWeightKerDict[device] = self.spatialWeightKer.to(device)


class ReinforcePointExtractor(nn.Module):

    def __init__(self, cuda, nbFeat, softCoord, softCoord_kerSize, softCoord_secOrder, point_nb, reconst, encoderChan, \
                 predictDepth, softcoord_shiftpred, furthestPointSampling, furthestPointSampling_nb_pts, dropout,
                 auxModel, \
                 topkRandSamp, topkRSUnifWeight, hasLinearProb, use_baseline, reinf_linear_only):

        super(ReinforcePointExtractor, self).__init__()

        self.conv1x1 = nn.Conv2d(nbFeat, encoderChan, kernel_size=1, stride=1)
        self.point_extracted = point_nb
        self.softCoord = softCoord
        self.softCoord_kerSize = softCoord_kerSize
        self.softCoord_secOrder = softCoord_secOrder
        self.softcoord_shiftpred = softcoord_shiftpred
        self.furthestPointSampling = furthestPointSampling
        self.furthestPointSampling_nb_pts = furthestPointSampling_nb_pts
        self.dropout = dropout
        self.hasLinearProb = hasLinearProb
        self.use_baseline = use_baseline

        self.reconst = reconst
        if reconst:
            self.decoder = nn.Sequential(resnet.conv1x1(encoderChan, 3, stride=1), \
                                         resnet.resnet4(pretrained=True, chan=64, featMap=True, layerSizeReduce=False,
                                                        preLayerSizeReduce=False, stride=1), \
                                         resnet.conv1x1(64, 3, stride=1))

        self.predictDepth = predictDepth
        if predictDepth:
            self.conv1x1_depth = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1)

        if self.softcoord_shiftpred and self.softCoord:
            raise ValueError("softcoord_shiftpred and softCoord can't be true at the same time.")

        if self.softcoord_shiftpred:
            self.shiftpred_x, self.shiftpred_y = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1), nn.Conv2d(nbFeat, 1,
                                                                                                          kernel_size=1,
                                                                                                          stride=1)

        if self.hasLinearProb:
            self.lin_prob = nn.Conv2d(encoderChan, 1, kernel_size=1, stride=1)

        if self.softCoord:

            if self.softCoord_kerSize % 2 == 0:
                raise ValueError("Kernel size of soft coordinate extractor must not be a multiple of 2.")

            ordKer = (torch.arange(self.softCoord_kerSize) - self.softCoord_kerSize // 2).unsqueeze(1).unsqueeze(
                0).unsqueeze(0).expand(1, 1, self.softCoord_kerSize, self.softCoord_kerSize).float()
            absKer = (torch.arange(self.softCoord_kerSize) - self.softCoord_kerSize // 2).unsqueeze(0).unsqueeze(
                0).unsqueeze(0).expand(1, 1, self.softCoord_kerSize, self.softCoord_kerSize).float()
            self.ordKer, self.absKer = (ordKer.cuda(), absKer.cuda()) if cuda else (ordKer, absKer)

            self.spatialWeightKer = self.softCoord_kerSize - (torch.abs(self.ordKer) + torch.abs(self.absKer))
            if self.softCoord_secOrder:
                self.spatialWeightKer = self.spatialWeightKer * self.spatialWeightKer

            self.ordKerDict, self.absKerDict, self.spatialWeightKerDict = {}, {}, {}

        self.auxModel = auxModel
        self.topkRandSamp = topkRandSamp
        self.topkRSUnifWeight = topkRSUnifWeight
        self.reinf_linear_only = reinf_linear_only
        if use_baseline:
            self.baseline_linear = nn.Linear(nbFeat, 1)

    def forward(self, featureMaps):

        # Because of zero padding, the border are very active, so we remove it.
        featureMaps = featureMaps[:, :, 3:-3, 3:-3]


        import numpy as np
        pointFeaturesMap = self.conv1x1(featureMaps)

        if self.hasLinearProb:
            if self.reinf_linear_only:
                x = self.lin_prob(pointFeaturesMap.detach())
            else:
                x = self.lin_prob(pointFeaturesMap)
            flatX = x.view(x.size(0), -1)
            # probs = F.softmax(flatX, dim=(1))+_EPSILON
            flatX = torch.sigmoid(flatX)
            probs = flatX / flatX.sum(dim=1, keepdim=True) +_EPSILON
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
        retDict["batch"] = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0), points.size(1)).reshape(
            -1).to(points.device)
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


class PointNet2(SecondModel):

    def __init__(self, cuda, classNb, nbFeat, pn_model, topk=False, reinfExct=False, \
                 topk_softcoord=False, topk_softCoord_kerSize=2, topk_softCoord_secOrder=False, point_nb=256,
                 reconst=False, \
                 encoderChan=1, encoderHidChan=64, predictDepth=False, topk_softcoord_shiftpred=False, topk_fps=False,
                 topk_fps_nb_pts=64, \
                 topk_dropout=0, auxModel=False, topkRandSamp=False, topkRSUnifWeight=0, hasLinearProb=False,
                 use_baseline=False, \
                 topk_euclinorm=True , reinf_linear_only=False):

        super(PointNet2, self).__init__(nbFeat, classNb)

        if topk:
            self.pointExtr = TopkPointExtractor(cuda, nbFeat, topk_softcoord, topk_softCoord_kerSize, \
                                                topk_softCoord_secOrder, point_nb, reconst, encoderChan, predictDepth, \
                                                topk_softcoord_shiftpred, topk_fps, topk_fps_nb_pts, topk_dropout,
                                                auxModel, \
                                                topkRandSamp, topkRSUnifWeight, topk_euclinorm,hasLinearProb)
        elif reinfExct:
            self.pointExtr = ReinforcePointExtractor(cuda, nbFeat, topk_softcoord, topk_softCoord_kerSize, \
                                                     topk_softCoord_secOrder, point_nb, reconst, encoderChan,
                                                     predictDepth, \
                                                     topk_softcoord_shiftpred, topk_fps, topk_fps_nb_pts, topk_dropout,
                                                     auxModel, \
                                                     topkRandSamp, topkRSUnifWeight, hasLinearProb, use_baseline, reinf_linear_only)
        else:
            self.pointExtr = DirectPointExtractor(point_nb, nbFeat)
            if auxModel:
                raise ValueError("Can't use aux model with direct point extractor")
        self.pn2 = pn_model

        self.auxModel = auxModel
        if auxModel:
            self.auxModel = nn.Linear(nbFeat, classNb)

    def forward(self, x):
        retDict = self.pointExtr(x)

        x = self.pn2(retDict['pointfeatures'], retDict['pos'], retDict['batch'])
        retDict['pred'] = x

        if self.auxModel:
            retDict["auxPred"] = self.auxModel(retDict['auxFeat'])

        return retDict


def getResnetFeat(backbone_name, backbone_inplanes):
    if backbone_name == "resnet50" or backbone_name == "resnet101" or backbone_name == "resnet151":
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("deeplab") != -1:
        nbFeat = 256
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
        nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)

        if not args.resnet_simple_att:
            CNNconst = CNN2D
            kwargs = {}
        else:
            CNNconst = CNN2D_simpleAttention
            kwargs = {"featMap": True, "topk": args.resnet_simple_att_topk,
                      "topk_pxls_nb": args.resnet_simple_att_topk_pxls_nb,
                      "topk_enc_chan":args.resnet_simple_att_topk_enc_chan}
            if args.resnet_simple_att_topk_enc_chan != -1:
                nbFeat = args.resnet_simple_att_topk_enc_chan

        firstModel = CNNconst(args.first_mod, args.pretrained_visual, chan=args.resnet_chan, stride=args.resnet_stride,
                              dilation=args.resnet_dilation, \
                              attChan=args.resnet_att_chan, attBlockNb=args.resnet_att_blocks_nb,
                              attActFunc=args.resnet_att_act_func, \
                              multiModel=args.resnet_multi_model, \
                              multiModSparseConst=args.resnet_multi_model_sparse_const, num_classes=args.class_nb, \
                              layerSizeReduce=args.resnet_layer_size_reduce,
                              preLayerSizeReduce=args.resnet_prelay_size_reduce, \
                              applyStrideOnAll=args.resnet_apply_stride_on_all, \
                              **kwargs)
    else:
        raise ValueError("Unknown visual model type : ", args.first_mod)

    if args.freeze_visual:
        for param in firstModel.parameters():
            param.requires_grad = False

    ############### Temporal Model #######################
    if args.second_mod == "linear":
        secondModel = LinearSecondModel(nbFeat, args.class_nb, args.dropout)
    elif args.second_mod == "pointnet2" or args.second_mod == "edgenet":
        if args.second_mod == "pointnet2":
            pn_model = pointnet2.Net(num_classes=args.class_nb,
                                     input_channels=args.pn_enc_chan if args.pn_topk or args.pn_reinf else 0)
        else:
            pn_model = pointnet2.EdgeNet(num_classes=args.class_nb,
                                         input_channels=args.pn_enc_chan if args.pn_topk else 3)

        secondModel = PointNet2(args.cuda, args.class_nb, nbFeat=nbFeat, pn_model=pn_model,
                                encoderHidChan=args.pn_topk_hid_chan, \
                                topk=args.pn_topk, reinfExct=args.pn_reinf, topk_softcoord=args.pn_topk_softcoord,
                                topk_softCoord_kerSize=args.pn_topk_softcoord_kersize, \
                                topk_softCoord_secOrder=args.pn_topk_softcoord_secorder, point_nb=args.pn_point_nb,
                                reconst=args.pn_topk_reconst, topk_softcoord_shiftpred=args.pn_topk_softcoord_shiftpred, \
                                encoderChan=args.pn_enc_chan, predictDepth=args.pn_topk_pred_depth, \
                                topk_fps=args.pn_topk_farthest_pts_sampling, topk_fps_nb_pts=args.pn_topk_fps_nb_points,
                                topk_dropout=args.pn_topk_dropout, \
                                auxModel=args.pn_aux_model, topkRandSamp=args.pn_topk_rand_sampling,
                                topkRSUnifWeight=args.pn_topk_rs_unif_weight,
                                hasLinearProb=args.pn_has_linear_prob, use_baseline=args.pn_reinf_use_baseline, \
                                topk_euclinorm=args.pn_topk_euclinorm, reinf_linear_only=args.pn_train_reinf_linear_only)
    else:
        raise ValueError("Unknown temporal model type : ", args.second_mod)

    ############### Whole Model ##########################

    net = Model(firstModel, secondModel,zoom=args.zoom)

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

    argreader.parser.add_argument('--lstm_lay', type=int, metavar='N',
                                  help='Number of layers for the lstm temporal model')

    argreader.parser.add_argument('--lstm_hid_size', type=int, metavar='N',
                                  help='Size of hidden layers for the lstm temporal model')

    argreader.parser.add_argument('--freeze_visual', type=args.str2bool, metavar='BOOL',
                                  help='To freeze the weights of the visual model during training.')

    argreader.parser.add_argument('--use_time', type=args.str2bool, metavar='BOOL',
                                  help='To use the time elapsed of each image as a feature')

    argreader.parser.add_argument('--feat_attention_ker_size', type=int, metavar='BOOL',
                                  help='The kernel size of the feature attention.')

    argreader.parser.add_argument('--feat_attention_att_type', type=str, metavar='TYPE',
                                  help="The attention type. Can be 'shallow' or 'deep'.")

    argreader.parser.add_argument('--feat_attention_grouped_att', type=args.str2bool, metavar='BOOl',
                                  help="To use grouped convolution in the attention module.")

    argreader.parser.add_argument('--pretrained_visual', type=args.str2bool, metavar='BOOL',
                                  help='To have a visual feature extractor pretrained on ImageNet.')

    argreader.parser.add_argument('--class_bias_model', type=args.str2bool, metavar='BOOL',
                                  help='To have a global feature model (ignored when not using a feature attention model.)')

    argreader.parser.add_argument('--spat_transf', type=args.str2bool, metavar='BOOL',
                                  help='To have a spatial transformer network (STN) before the visual model')

    argreader.parser.add_argument('--spat_transf_img_size', type=int, metavar='BOOL',
                                  help='To size to which the image will be resized after the STN.')
    argreader.parser.add_argument('--pn_topk', type=args.str2bool, metavar='BOOL',
                                  help='To feed the pointnet model with points extracted using torch.topk and not a direct coordinate predictor. Ignored if the model \
                        doesn\'t use pointnet.')
    argreader.parser.add_argument('--pn_reinf', type=args.str2bool, metavar='BOOL',
                                  help='To feed the pointnet model with points extracted using reinforcement learning. Ignored if the model \
                            doesn\'t use pointnet.')
    argreader.parser.add_argument('--pn_has_linear_prob', type=args.str2bool, metavar='BOOL',
                                  help='To use linear layer to compute probabilities for the reinforcement model')
    argreader.parser.add_argument('--pn_reinf_use_baseline', type=args.str2bool, metavar='BOOL',
                                  help='To use linear layer to compute baseline for the reinforcement model training')
    argreader.parser.add_argument('--pn_train_reinf_linear_only', type=args.str2bool, metavar='BOOL',
                                  help='To prevent reinforcement loss to propagate in firstModel ')


    argreader.parser.add_argument('--pn_topk_softcoord', type=args.str2bool, metavar='BOOL',
                                  help='For the topk point net model. The point coordinate will be computed using soft argmax.')
    argreader.parser.add_argument('--pn_topk_softcoord_kersize', type=int, metavar='KERSIZE',
                                  help='For the topk point net model. This is the kernel size of the soft argmax.')
    argreader.parser.add_argument('--pn_topk_softcoord_secorder', type=args.str2bool, metavar='BOOL',
                                  help='For the topk point net model. The spatial weight kernel is squarred to make the central elements weight more.')
    argreader.parser.add_argument('--pn_topk_softcoord_shiftpred', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To predict the coordinate shift of each point instead of using weighted means.')
    argreader.parser.add_argument('--pn_topk_farthest_pts_sampling', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To apply furthest point sampling when extracting points.')
    argreader.parser.add_argument('--pn_topk_fps_nb_points', type=int, metavar='INT',
                                  help='For the pointnet2 model. The number of point extracted by furthest point sampling.')
    argreader.parser.add_argument('--pn_topk_rand_sampling', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To sample point in a probabilistic way instead of topk during training.')
    argreader.parser.add_argument('--pn_topk_rs_unif_weight', type=float, metavar='INT',
                                  help='For the stochastic sampling of pixels (when --pn_topk_rand_sampling is True). The distribution of probability \
                        is an interpolation between the uniform distribution and the distribution based purely on pixel feature norm. Set this arg \
                        respectively to 0 and 1 to make only the feature norm and the uniform distribution matter. This arg must be comprised between 0 and 1.')

    argreader.parser.add_argument('--pn_topk_euclinorm', type=args.str2bool, metavar='BOOL',
                                  help='For the topk point net model. To use the euclidean norm to compute pixel importance instead of using their raw value \
                                  filtered by a Relu.')

    argreader.parser.add_argument('--zoom', type=args.str2bool, metavar='BOOL',
                                  help='To use with a model that generates points. To zoom on the part of the images where the points are focused an apply the model a second time on it.')

    argreader.parser.add_argument('--pn_point_nb', type=int, metavar='NB',
                                  help='For the topk point net model. This is the number of point extracted for each image.')
    argreader.parser.add_argument('--pn_topk_reconst', type=args.str2bool, metavar='BOOL',
                                  help='For the topk point net model. An input image reconstruction term will added to the loss function if True.')
    argreader.parser.add_argument('--pn_enc_chan', type=int, metavar='NB',
                                  help='For the topk point net model. This is the number of output channel of the encoder')
    argreader.parser.add_argument('--pn_topk_hid_chan', type=int, metavar='NB',
                                  help='For the topk point net model. This is the number of hidden channel of the encoder')
    argreader.parser.add_argument('--pn_topk_pred_depth', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To predict the depth of chosen points.')
    argreader.parser.add_argument('--pn_use_xyz', type=args.str2bool, metavar='INT',
                                  help='For the pointnet2 model. To use the point coordinates as feature.')
    argreader.parser.add_argument('--pn_topk_dropout', type=float, metavar='FLOAT',
                                  help='The proportion of point to randomly drop to decrease overfitting.')
    argreader.parser.add_argument('--pn_aux_model', type=args.str2bool, metavar='INT',
                                  help='To train an auxilliary model that will apply average pooling and a dense layer on the feature map\
                        to make a prediction alongside pointnet\'s one.')

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_dilation', type=int, metavar='INT',
                                  help='The dilation for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_multi_model', type=args.str2bool, metavar='INT',
                                  help='To apply average pooling and a dense layer to each feature map. This leads to one model \
                        per scale. The final scores is the average of the scores provided by each model.')
    argreader.parser.add_argument('--resnet_multi_model_sparse_const', type=args.str2bool, metavar='INT',
                                  help='For the resnet attention block. Forces the attention map of higher resolution to be sparsier \
                        than the lower resolution attention maps.')
    argreader.parser.add_argument('--resnet_layer_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the layer 2,3 and 4 when the resnet model is used.')
    argreader.parser.add_argument('--resnet_prelay_size_reduce', type=args.str2bool, metavar='INT',
                                  help='To apply a stride of 2 in the convolution and the maxpooling before the layer 1.')
    argreader.parser.add_argument('--resnet_simple_att', type=args.str2bool, metavar='INT',
                                  help='To apply a simple attention on top of the resnet model.')
    argreader.parser.add_argument('--resnet_simple_att_topk', type=args.str2bool, metavar='BOOL',
                                  help='To use top-k feature as attention model with resnet. Ignored when --resnet_simple_att is False.')
    argreader.parser.add_argument('--resnet_simple_att_topk_pxls_nb', type=int, metavar='INT',
                                  help='The value of k when using top-k selection for resnet simple attention. Ignored when --resnet_simple_att_topk is False.')
    argreader.parser.add_argument('--resnet_simple_att_topk_enc_chan', type=int, metavar='NB',
                                  help='For the resnet_simple_att_topk model. This is the number of output channel of the encoder. Ignored when --resnet_simple_att_topk is False.')

    argreader.parser.add_argument('--resnet_apply_stride_on_all', type=args.str2bool, metavar='NB',
                                  help='Apply stride on every non 3x3 convolution')


    argreader.parser.add_argument('--resnet_att_chan', type=int, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The number of channels in the attention module.')
    argreader.parser.add_argument('--resnet_att_blocks_nb', type=int, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The number of blocks in the attention module.')
    argreader.parser.add_argument('--resnet_att_act_func', type=str, metavar='INT',
                                  help='For the \'resnetX_att\' feat models. The activation function for the attention weights. Can be "sigmoid", "relu" or "tanh+relu".')


    return argreader
