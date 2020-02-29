import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet
import resnet3D
import vgg
import args
import sys
import cv2
import deeplab
import glob
from skimage.transform import resize
import matplotlib.pyplot as plt

#pointnet2 and torch_cluster can only be installed on a machine that has a gpu
#But to avoid program stopping when doing test on a no-gpu machine
#the error is catched
try:
    import torch_geometric
    import pointnet2
except ModuleNotFoundError:
    pass

def buildFeatModel(featModelName,pretrainedFeatMod,featMap=False,bigMaps=False,layerSizeReduce=False,stride=2,dilation=1,**kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("deeplabv3") != -1:
        featModel = deeplab._segm_resnet("deeplabv3", featModelName[featModelName.find("resnet"):],\
                                        pretrained=pretrainedFeatMod,featMap=featMap,layerSizeReduce=layerSizeReduce,**kwargs)
    elif featModelName.find("resnet") != -1:
        featModel = getattr(resnet,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,layerSizeReduce=layerSizeReduce,**kwargs)
    elif featModelName == "r2plus1d_18":
        featModel = getattr(resnet3D,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    elif featModelName.find("vgg") != -1:
        featModel = getattr(vgg,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    else:
        raise ValueError("Unknown model type : ",featModelName)

    return featModel

def mapToList(map,abs,ord):
    #This extract the desired pixels in a map

    indices = tuple([torch.arange(map.size(0), dtype=torch.long).unsqueeze(1).unsqueeze(1),
                     torch.arange(map.size(1), dtype=torch.long).unsqueeze(1).unsqueeze(0),
                     ord.long().unsqueeze(1),abs.long().unsqueeze(1)])
    list = map[indices].permute(0,2,1)
    return list

#This class is just the class nn.DataParallel that allow running computation on multiple gpus
#but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super(DataParallelModel, self).__init__(model)

    def __getattr__(self, name):
        try:
            return super(DataParallelModel, self).__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self,firstModel,secondModel,spatTransf=None):
        super(Model,self).__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel
        self.spatTransf = spatTransf

        self.transMat = torch.zeros((self.secondModel.nbClass,self.secondModel.nbClass))
        self.priors = torch.zeros((self.secondModel.nbClass))

    def forward(self,x,timeElapsed=None):
        if self.spatTransf:
            x = self.spatTransf(x)["x"]

        visResDict = self.firstModel(x)
        x = visResDict["x"]

        resDict = self.secondModel(x,self.firstModel.batchSize,timeElapsed)

        for key in visResDict.keys():
            resDict[key] = visResDict[key]

        return resDict

    def computeVisual(self,x):
        if self.spatTransf:
            resDict = self.spatTransf(x)
            x = resDict["x"]
            theta = resDict["theta"]

        resDict = self.firstModel(x)

        if self.spatTransf:
            resDict["theta"] = theta
        return resDict

    def setTransMat(self,transMat):
        self.transMat = transMat
    def setPriors(self,priors):
        self.priors = priors

################################# Spatial Transformer #########################

class SpatialTransformer(nn.Module):

    def __init__(self,inSize,outSize):

        super(SpatialTransformer,self).__init__()

        self.outSize = outSize
        postVisFeatSize = outSize//4

        self.visFeat = buildFeatModel("resnet4",False,featMap=True,layerSizeReduce=False)
        self.conv1x1 = nn.Conv2d(8,1,1)
        self.mlp = nn.Sequential(nn.Linear(postVisFeatSize*postVisFeatSize,512),nn.ReLU(True),\
                                 nn.Linear(512,6))

        #Initialising
        self.mlp[2].weight.data.zero_()
        self.mlp[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self,inImage):

        batchSize = inImage.size(0)

        inImage = inImage.view(inImage.size(0)*inImage.size(1),inImage.size(2),inImage.size(3),inImage.size(4))

        x = torch.nn.functional.interpolate(inImage, size=self.outSize,mode='bilinear', align_corners=False)
        x = self.visFeat(x)
        x = self.conv1x1(x)
        x = x.view(x.size(0),-1)
        x = self.mlp(x)
        theta = x.view(x.size(0),2,3)
        grid = F.affine_grid(theta, inImage.size())
        x = F.grid_sample(inImage, grid)
        x = torch.nn.functional.interpolate(x, size=self.outSize,mode='bilinear', align_corners=False)

        x = x.view(batchSize,x.size(0)//batchSize,x.size(1),x.size(2),x.size(3))

        return {"x":x,"theta":theta}

################################# Visual Model ##########################

class FirstModel(nn.Module):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False,**kwargs):
        super(FirstModel,self).__init__()

        self.featMod = buildFeatModel(featModelName,pretrainedFeatMod,featMap,bigMaps,**kwargs)

        self.featMap = featMap
        self.bigMaps = bigMaps
        self.videoMode = videoMode
    def forward(self,x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False,**kwargs):
        super(CNN2D,self).__init__(videoMode,featModelName,pretrainedFeatMod,featMap,bigMaps,**kwargs)

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)).contiguous() if self.videoMode else x
        # NT x C x H x L
        res = self.featMod(x)

        # NT x D
        if type(res) is dict:
            #Some feature model can return a dictionnary instead of a tensor
            return res
        else:
            return {'x':res}

class CNN2D_simpleAttention(FirstModel):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False,chan=64,attBlockNb=2,attChan=16,\
                topk=False,topk_pxls_nb=256,**kwargs):

        super(CNN2D_simpleAttention,self).__init__(videoMode,featModelName,pretrainedFeatMod,featMap,bigMaps,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        inFeat = getResnetFeat(featModelName,chan)

        attention = []
        for i in range(attBlockNb):
            attention.append(resnet.BasicBlock(inFeat, inFeat))
        attention.append(resnet.conv1x1(inFeat,1))

        self.topk = topk
        if not topk:
            self.attention = nn.Sequential(*attention)
            self.topk_pxls_nb = None
        else:
            self.attention = None
            self.topk_pxls_nb = topk_pxls_nb

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)).contiguous() if self.videoMode else x
        # NT x C x H x L
        features = self.featMod(x)

        if not self.topk:
            spatialWeights = torch.sigmoid(self.attention(features))
            features = spatialWeights*features
            features = self.avgpool(features)
            features = features.view(features.size(0), -1)
        else:
            spatialWeights = torch.zeros((features.size(0),1,features.size(2),features.size(3)))
            #Compute the mean between the k most active pixels
            featNorm = torch.pow(features,2).sum(dim=1,keepdim=True)
            flatFeatNorm = featNorm.view(featNorm.size(0),-1)
            flatVals,flatInds = torch.topk(flatFeatNorm, self.topk_pxls_nb, dim=-1, largest=True)
            abs,ord = (flatInds%featNorm.shape[-1],flatInds//featNorm.shape[-1])
            featureList = mapToList(features,abs,ord)
            features = featureList.mean(dim=1)
            indices = tuple([torch.arange(spatialWeights.size(0), dtype=torch.long).unsqueeze(1).unsqueeze(1),
                             torch.arange(spatialWeights.size(1), dtype=torch.long).unsqueeze(1).unsqueeze(0),
                            ord.long().unsqueeze(1),abs.long().unsqueeze(1)])
            spatialWeights[indices] = 1

        return {'x':features,'attMaps':spatialWeights}

class CNN3D(FirstModel):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False):
        super(CNN3D,self).__init__(videoMode,featModelName,pretrainedFeatMod,featMap,bigMaps)

        if not self.videoMode:
            raise NotImplementedError("A CNN3D can't be used when video_mode is False")

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.permute(0,2,1,3,4)
        # N x C x T x H x L

        x = self.featMod(x)

        if self.featMap:
            # N x D x T x H x L
            x = x.permute(0,2,1,3,4)
            # N x T x D x H x L
            x = x.contiguous().view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
            # NT x D x H x L
        else:
            # N x D x T
            x = x.permute(0,2,1)
            # N x T x D
            x = x.contiguous().view(x.size(0)*x.size(1),-1)
            # NT x D
        return {'x':x}

class ClassBias(nn.Module):

    def __init__(self,nbFeat,nbClass):

        super(ClassBias,self).__init__()

        self.hidFeat = 128
        self.inFeat = nbFeat
        self.nbClass = nbClass

        self.mlp = nn.Sequential(nn.Linear(self.inFeat,self.hidFeat),nn.ReLU(),
                    nn.Linear(self.hidFeat,self.hidFeat))

        self.classFeat = nn.Parameter(torch.zeros((self.hidFeat,self.nbClass)).uniform_(-1/self.hidFeat,1/self.hidFeat))

    def forward(self,x):

        x = x.mean(dim=-1).mean(dim=-1)

        #Computing context vector
        x = self.mlp(x)
        #N x 512
        x = x.unsqueeze(2).expand(x.size(0),x.size(1),self.nbClass)
        #N x 512 x class_nb
        x = (self.classFeat.unsqueeze(0)*x).sum(dim=1)
        #N x class_nb
        x = x.unsqueeze(-1).unsqueeze(-1)
        #N x class_nb x 1 x 1

        return x

class Attention(nn.Module):

    def __init__(self,type,inFeat,nbClass,grouped=True):
        super(Attention,self).__init__()

        nbGroups = 1 if not grouped else nbClass
        hidFeat = inFeat

        if type == "shallow":
            self.attention = nn.Conv2d(inFeat,nbClass,3,padding=1,groups=nbGroups)
        elif type == "deep":

            if hidFeat != nbClass:
                downsample = nn.Sequential(resnet.conv1x1(hidFeat, nbClass),nn.BatchNorm2d(nbClass))
            else:
                downsample = None

            self.attention = nn.Sequential(resnet.BasicBlock(hidFeat, hidFeat,groups=nbGroups),resnet.BasicBlock(hidFeat, nbClass,groups=nbGroups,downsample=downsample,feat=True))
        else:
            raise ValueError("Unknown attention type :",type)

    def forward(self,x):
        return self.attention(x)

class AttentionModel(FirstModel):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod,nbFeat,nbClass,classBiasMod=None,attType="shallow",groupedAtt=True,**kwargs):
        super(AttentionModel,self).__init__(videoMode,featModelName,pretrainedFeatMod,True,**kwargs)

        self.classConv = nn.Conv2d(nbFeat,nbClass,1)
        self.attention = Attention(attType,nbClass,nbClass,groupedAtt)
        self.nbClass = nbClass
        self.classBiasMod = classBiasMod

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)) if self.videoMode else x
        featureVolume = self.featMod(x)

        classFeatMaps = self.classConv(featureVolume)

        attWeight = self.attention(classFeatMaps)

        if self.classBiasMod:
            attWeight += self.classBiasMod(featureVolume)

        attWeight = torch.sigmoid(attWeight)

        x = classFeatMaps*attWeight
        # N x class_nb x 7 x 7

        x = x.mean(dim=-1).mean(dim=-1)

        return {"x":x,"attention":attWeight,"features":classFeatMaps}

class AttentionFullModel(FirstModel):

    def __init__(self,videoMode,featModelName,pretrainedFeatMod,nbFeat,nbClass,classBiasMod=None,attType="shallow",groupedAtt=True,**kwargs):
        super(AttentionFullModel,self).__init__(videoMode,featModelName,pretrainedFeatMod,True,**kwargs)

        #self.attention = nn.Conv2d(nbFeat,nbClass,attKerSize,padding=attKerSize//2)
        self.attention = Attention(attType,nbFeat,nbClass,groupedAtt)

        self.lin = nn.Linear(nbFeat*nbClass,nbClass)
        self.nbClass = nbClass
        self.classBiasMod = classBiasMod

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)) if self.videoMode else x
        x = self.featMod(x)
        # N*T x D x H x W

        attWeight = self.attention(x)

        if self.classBiasMod:
            attWeight += self.classBiasMod(x)

        attWeight = torch.sigmoid(attWeight)

        x = x.unsqueeze(2).expand(x.size(0),x.size(1),self.nbClass,x.size(2),x.size(3))
        # N*T x D x class nb x H x W

        attWeightExp = attWeight.unsqueeze(1).expand(attWeight.size(0),x.size(1),attWeight.size(1),attWeight.size(2),attWeight.size(3))

        x = x*attWeightExp
        x = x.mean(dim=-1).mean(dim=-1)
        # N*T x D x class nb
        x = x.permute(0,2,1)
        # N*T x class nb x D
        x = x.contiguous().view(x.size(0),-1)
        # N*T x (class nb*D)
        x = self.lin(x)
        # N*T x class_nb

        return {"x":x,"attention":attWeight}

class DirectPointExtractor(nn.Module):

    def __init__(self,point_nb):
        super(DirectPointExtractor,self).__init__()
        self.feat = resnet.resnet4(pretrained=True,chan=64,featMap=True)
        self.conv1x1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.size_red = nn.AdaptiveAvgPool2d((8, 8))
        self.dense = nn.Linear(8*8*32,point_nb*2)

    def forward(self,x):
        x = self.feat(x)
        x = self.conv1x1(x)
        x = self.size_red(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        points = x.view(x.size(0),x.size(1)//2,2)
        points = torch.cat((points,torch.zeros(points.size(0),points.size(1),1).to(x.device)),dim=-1)

        return {"points":points,
                "batch":torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0),points.size(1)).reshape(-1).to(points.device),
                "pos":points.reshape(points.size(0)*points.size(1),points.size(2)),
                "pointfeatures":None}

class TopkPointExtractor(nn.Module):

    def __init__(self,cuda,nbFeat,featMod,softCoord,softCoord_kerSize,softCoord_secOrder,point_nb,reconst,encoderChan,\
                    predictDepth,softcoord_shiftpred,furthestPointSampling,furthestPointSampling_nb_pts,dropout):

        super(TopkPointExtractor,self).__init__()

        self.feat = featMod
        self.conv1x1 = nn.Conv2d(nbFeat, encoderChan, kernel_size=1, stride=1)
        self.point_extracted = point_nb
        self.softCoord = softCoord
        self.softCoord_kerSize = softCoord_kerSize
        self.softCoord_secOrder = softCoord_secOrder
        self.softcoord_shiftpred = softcoord_shiftpred
        self.furthestPointSampling = furthestPointSampling
        self.furthestPointSampling_nb_pts = furthestPointSampling_nb_pts
        self.dropout = dropout

        self.reconst = reconst
        if reconst:
            self.decoder = nn.Sequential(resnet.conv1x1(encoderChan, 3, stride=1),\
                           resnet.resnet4(pretrained=True,chan=64,featMap=True,layerSizeReduce=False,preLayerSizeReduce=False,stride=1),\
                           resnet.conv1x1(64, 3, stride=1))

        self.predictDepth = predictDepth
        if predictDepth:
            self.conv1x1_depth = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1)

        if self.softcoord_shiftpred and self.softCoord:
            raise ValueError("softcoord_shiftpred and softCoord can't be true at the same time.")

        if self.softcoord_shiftpred:
            self.shiftpred_x,self.shiftpred_y = nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1),nn.Conv2d(nbFeat, 1, kernel_size=1, stride=1)

        if self.softCoord:

            if self.softCoord_kerSize%2==0:
                raise ValueError("Kernel size of soft coordinate extractor must not be a multiple of 2.")

            ordKer = (torch.arange(self.softCoord_kerSize)-self.softCoord_kerSize//2).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(1,1,self.softCoord_kerSize,self.softCoord_kerSize).float()
            absKer = (torch.arange(self.softCoord_kerSize)-self.softCoord_kerSize//2).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1,1,self.softCoord_kerSize,self.softCoord_kerSize).float()
            self.ordKer,self.absKer = (ordKer.cuda(),absKer.cuda()) if cuda else (ordKer,absKer)

            self.spatialWeightKer = self.softCoord_kerSize-(torch.abs(self.ordKer)+torch.abs(self.absKer))
            if self.softCoord_secOrder:
                self.spatialWeightKer = self.spatialWeightKer*self.spatialWeightKer

            self.ordKerDict,self.absKerDict,self.spatialWeightKerDict = {},{},{}

    def forward(self,imgBatch):

        featureMaps = self.feat(imgBatch)

        #Because of zero padding, the border are very active, so we remove it.
        featureMaps = featureMaps[:,:,3:-3,3:-3]

        pointFeaturesMap = self.conv1x1(featureMaps)
        x = torch.pow(pointFeaturesMap,2).sum(dim=1,keepdim=True)

        flatX = x.view(x.size(0),-1)
        flatVals,flatInds = torch.topk(flatX, self.point_extracted, dim=-1, largest=True)
        abs,ord = (flatInds%x.shape[-1],flatInds//x.shape[-1])

        if self.furthestPointSampling:
            points = torch.cat((abs.unsqueeze(-1),ord.unsqueeze(-1)),dim=-1).float()

            # exampleInds is a list that indicates the example index for each point in the batch
            # If batch_size==1, exampleInds is just a list of zeros.
            # If batch_size==2, the first half of exampleInds if filled with 0's and the rest with 1's.
            exampleInds = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0),points.size(1)).reshape(-1).to(points.device)
            selectedPointInds = torch_geometric.nn.fps(points.view(-1,2),exampleInds,ratio=self.furthestPointSampling_nb_pts/abs.size(1))
            selectedPointInds = selectedPointInds.reshape(points.size(0),-1)
            selectedPointInds = selectedPointInds%abs.size(1)

            abs = abs[torch.arange(abs.size(0)).unsqueeze(1).long(),selectedPointInds.long()]
            ord = ord[torch.arange(ord.size(0)).unsqueeze(1).long(),selectedPointInds.long()]

        if self.dropout and self.training:
            idx = torch.randperm(abs.size(1))[:int((1-self.dropout)*abs.size(1))].unsqueeze(0)
            abs = abs[torch.arange(abs.size(0)).unsqueeze(1).long(),idx]
            ord = ord[torch.arange(ord.size(0)).unsqueeze(1).long(),idx]

        retDict={}

        if self.softCoord:
            abs,ord = self.fastSoftCoordRefiner(x,abs,ord,kerSize=self.softCoord_kerSize,secondOrderSpatWeight=self.softCoord_secOrder)
        if self.reconst:
            sparseX = pointFeaturesMap*((x-flatVals[:,-1].unsqueeze(1).unsqueeze(2).unsqueeze(3))>0).float()
            reconst = self.decoder(sparseX)
            #Keeping only one channel (the three channels are just copies)
            retDict['reconst'] = reconst[:,0:1]

        pointFeat = mapToList(pointFeaturesMap,abs,ord)

        if self.predictDepth:
            depthMap = torch.tanh(self.conv1x1_depth(featureMaps))*max(featureMaps.size(-2),featureMaps.size(-1))//10
            depth = mapToList(depthMap,abs,ord)
        else:
            depth = torch.zeros(abs.size(0),abs.size(1),1).to(x.device)

        if self.softcoord_shiftpred:
            xShiftMap,yShiftMap = torch.tanh(self.shiftpred_x(featureMaps)),torch.tanh(self.shiftpred_y(featureMaps))
            xShift,yShift = mapToList(xShiftMap,abs,ord).squeeze(-1),mapToList(yShiftMap,abs,ord).squeeze(-1)
            abs = xShift + abs.float()
            ord = yShift + ord.float()

        abs,ord = abs.unsqueeze(-1).float(),ord.unsqueeze(-1).float()
        points = torch.cat((abs,ord,depth),dim=-1).float()
        retDict['points'] = torch.cat((abs,ord,depth,pointFeat),dim=-1).float()
        retDict["batch"] = torch.arange(points.size(0)).unsqueeze(1).expand(points.size(0),points.size(1)).reshape(-1).to(points.device)
        retDict["pos"] = points.reshape(points.size(0)*points.size(1),points.size(2))
        retDict["pointfeatures"] = pointFeat.reshape(pointFeat.size(0)*pointFeat.size(1),pointFeat.size(2))
        return retDict

    def fastSoftCoordRefiner(self,x,abs,ord,kerSize=5,secondOrderSpatWeight=False):

        self.updateDict(x.device)

        ordTens = torch.arange(x.size(-2)).unsqueeze(1).expand(x.size(-2),x.size(-1)).float().to(x.device)
        absTens = torch.arange(x.size(-1)).unsqueeze(0).expand(x.size(-2),x.size(-1)).float().to(x.device)

        unorm_ordShiftMean = F.conv2d(x, self.ordKerDict[x.device]*self.spatialWeightKerDict[x.device], stride=1, padding=kerSize//2)
        unorm_absShiftMean = F.conv2d(x, self.absKerDict[x.device]*self.spatialWeightKerDict[x.device], stride=1, padding=kerSize//2)

        weightSum = F.conv2d(x, self.spatialWeightKerDict[x.device], stride=1, padding=kerSize//2)

        startOrd,endOrd = ord[0,0].int().item()-kerSize//2,ord[0,0].int().item()+kerSize//2+1
        startAbs,endAbs = abs[0,0].int().item()-kerSize//2,abs[0,0].int().item()+kerSize//2+1

        ordMean = ordTens.unsqueeze(0).unsqueeze(0)+unorm_ordShiftMean/(weightSum+0.001)
        absMean = absTens.unsqueeze(0).unsqueeze(0)+unorm_absShiftMean/(weightSum+0.001)

        ordList = ordMean[:,0][torch.arange(x.size(0), dtype=torch.long).unsqueeze(1),ord.long(),abs.long()]
        absList = absMean[:,0][torch.arange(x.size(0), dtype=torch.long).unsqueeze(1),ord.long(),abs.long()]

        return absList,ordList

    def updateDict(self,device):

        if not device in self.ordKerDict.keys():
            self.ordKerDict[device] = self.ordKer.to(device)
            self.absKerDict[device] = self.absKer.to(device)
            self.spatialWeightKerDict[device] = self.spatialWeightKer.to(device)

class PointNet2(FirstModel):

    def __init__(self,cuda,videoMode,classNb,nbFeat,featModelName='resnet18',pretrainedFeatMod=True,topk=False,\
                topk_softcoord=False,topk_softCoord_kerSize=2,topk_softCoord_secOrder=False,point_nb=256,reconst=False,\
                encoderChan=1,encoderHidChan=64,predictDepth=False,topk_softcoord_shiftpred=False,topk_fps=False,topk_fps_nb_pts=64,\
                topk_dropout=0,**kwargs):

        super(PointNet2,self).__init__(videoMode,featModelName,pretrainedFeatMod,True,chan=encoderHidChan,**kwargs)

        if topk:
            self.pointExtr = TopkPointExtractor(cuda,nbFeat,self.featMod,topk_softcoord,topk_softCoord_kerSize,\
                                                topk_softCoord_secOrder,point_nb,reconst,encoderChan,predictDepth,\
                                                topk_softcoord_shiftpred,topk_fps,topk_fps_nb_pts,topk_dropout)
        else:
            self.pointExtr = DirectPointExtractor(point_nb)

        self.pn2 = pointnet2.Net(num_classes=classNb,input_channels=encoderChan if topk else 0)

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)) if self.videoMode else x
        retDict = self.pointExtr(x)
        x = self.pn2(retDict['pointfeatures'],retDict['pos'],retDict['batch'])

        retDict['x'] = x

        return retDict

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self,videoMode,nbFeat,nbClass,useTime):
        super(SecondModel,self).__init__()
        self.nbFeat,self.nbClass = nbFeat,nbClass
        self.useTime = useTime

        if useTime:
            self.nbFeat += 1

        self.videoMode = videoMode

    def forward(self,x):
        raise NotImplementedError

    def catWithTimeFeat(self,x,timeTensor):
        if self.useTime:
            timeTensor = timeTensor.view(-1).unsqueeze(1)
            x = torch.cat((x,timeTensor),dim=-1)
        return x

class LinearSecondModel(SecondModel):

    def __init__(self,videoMode,nbFeat,nbClass,useTime,dropout):
        super(LinearSecondModel,self).__init__(videoMode,nbFeat,nbClass,useTime)
        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat,self.nbClass)

    def forward(self,x,batchSize,timeTensor=None):
        x = self.catWithTimeFeat(x,timeTensor)

        # NT x D
        x = self.dropout(x)
        x = self.linLay(x)
        # NT x classNb

        x = x.view(batchSize,-1,self.nbClass) if self.videoMode else x

        #N x T x classNb
        return {"pred":x}

class LSTMSecondModel(SecondModel):

    def __init__(self,videoMode,nbFeat,nbClass,useTime,dropout,nbLayers,nbHidden):
        super(LSTMSecondModel,self).__init__(videoMode,nbFeat,nbClass,useTime)

        self.lstmTempMod = nn.LSTM(input_size=self.nbFeat,hidden_size=nbHidden,num_layers=nbLayers,batch_first=True,dropout=dropout,bidirectional=True)
        self.linTempMod = LinearSecondModel(videoMode=videoMode,nbFeat=nbHidden*2,nbClass=self.nbClass,useTime=False,dropout=dropout)

        if not self.videoMode:
            raise ValueError("Can't use LSTM as a second model when working on non video mode.")

    def forward(self,x,batchSize,timeTensor):

        self.lstmTempMod.flatten_parameters()

        x = self.catWithTimeFeat(x,timeTensor)

        # NT x D
        x = x.view(batchSize,-1,x.size(-1))
        # N x T x D
        x,_ = self.lstmTempMod(x)
        # N x T x H
        x = x.contiguous().view(-1,x.size(-1))
        # NT x H
        x = self.linTempMod(x,batchSize)["pred"]
        # N x T x classNb
        return {"pred":x}

class Identity(SecondModel):

    def __init__(self,videoMode,nbFeat,nbClass,useTime):

        super(Identity,self).__init__(videoMode,nbFeat,nbClass,useTime)

    def forward(self,x,batchSize,timeTensor):

        x = x.view(batchSize,-1,self.nbClass) if self.videoMode else x

        return {"pred":x}

def getResnetFeat(backbone_name,backbone_inplanes):

    if backbone_name=="resnet50" or backbone_name=="resnet101" or backbone_name=="resnet151":
        nbFeat = backbone_inplanes*4*2**(4-1)
    elif backbone_name.find("deeplab") != -1:
        nbFeat = 256
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes*2**(4-1)
    elif backbone_name.find("resnet14") != -1:
        nbFeat = backbone_inplanes*2**(3-1)
    elif backbone_name.find("resnet9") != -1:
        nbFeat = backbone_inplanes*2**(2-1)
    elif backbone_name.find("resnet4") != -1:
        nbFeat = backbone_inplanes*2**(1-1)
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args):

    ############### Visual Model #######################
    if args.feat.find("resnet") != -1:
        nbFeat = getResnetFeat(args.feat,args.resnet_chan)

        if not args.resnet_simple_att:
            CNNconst = CNN2D
            kwargs={}
        else:
            CNNconst = CNN2D_simpleAttention
            kwargs={"featMap":True,"topk":args.resnet_simple_att_topk,"topk_pxls_nb":args.resnet_simple_att_topk_pxls_nb}

        firstModel = CNNconst(args.video_mode,args.feat,args.pretrained_visual,chan=args.resnet_chan,stride=args.resnet_stride,dilation=args.resnet_dilation,\
                            attChan=args.resnet_att_chan,attBlockNb=args.resnet_att_blocks_nb,attActFunc=args.resnet_att_act_func,\
                            multiModel=args.resnet_multi_model,\
                            multiModSparseConst=args.resnet_multi_model_sparse_const,num_classes=args.class_nb,**kwargs)

    elif args.feat.find("vgg") != -1:
        nbFeat = 4096
        firstModel = CNN2D(args.video_mode,args.feat,args.pretrained_visual)
    elif args.feat == "r2plus1d_18":
        nbFeat = 512
        firstModel = CNN3D(args.video_mode,args.feat,args.pretrained_visual)
    else:
        raise ValueError("Unknown visual model type : ",args.feat)

    if args.freeze_visual:
        for param in firstModel.parameters():
            param.requires_grad = False

    ############### Temporal Model #######################
    if args.temp_mod == "lstm":
        secondModel = LSTMSecondModel(args.video_mode,nbFeat,args.class_nb,args.use_time,args.dropout,args.lstm_lay,args.lstm_hid_size)
    elif args.temp_mod == "linear":
        secondModel = LinearSecondModel(args.video_mode,nbFeat,args.class_nb,args.use_time,args.dropout)
    elif args.temp_mod == "feat_attention" or args.temp_mod == "feat_attention_full":

        classBiasMod = ClassBias(nbFeat,args.class_nb) if args.class_bias_model else None

        if args.temp_mod == "feat_attention":
            firstModel = AttentionModel(args.video_mode,args.feat,args.pretrained_visual,nbFeat,args.class_nb,classBiasMod,args.feat_attention_att_type,args.feat_attention_grouped_att,\
                                        chan=args.resnet_chan,multiModel=args.resnet_multi_model,dilation=args.resnet_dilation,\
                                        multiModSparseConst=args.resnet_multi_model_sparse_const,layerSizeReduce=args.resnet_layer_size_reduce)
            secondModel = Identity(args.video_mode,nbFeat,args.class_nb,False)

        elif args.temp_mod == "feat_attention_full":
            firstModel = AttentionFullModel(args.video_mode,args.feat,args.pretrained_visual,nbFeat,args.class_nb,classBiasMod,args.feat_attention_att_type,args.feat_attention_grouped_att,\
                                            chan=args.resnet_chan,multiModel=args.resnet_multi_model,dilation=args.resnet_dilation,\
                                            multiModSparseConst=args.resnet_multi_model_sparse_const,layerSizeReduce=args.resnet_layer_size_reduce)
            secondModel = Identity(args.video_mode,nbFeat,args.class_nb,False)
    elif args.temp_mod == "pointnet2":
        firstModel = PointNet2(args.cuda,args.video_mode,args.class_nb,nbFeat=nbFeat,featModelName=args.feat,pretrainedFeatMod=args.pretrained_visual,encoderHidChan=args.pn_topk_hid_chan,\
                                topk=args.pn_topk,topk_softcoord=args.pn_topk_softcoord,topk_softCoord_kerSize=args.pn_topk_softcoord_kersize,topk_softCoord_secOrder=args.pn_topk_softcoord_secorder,\
                                point_nb=args.pn_point_nb,reconst=args.pn_topk_reconst,topk_softcoord_shiftpred=args.pn_topk_softcoord_shiftpred,\
                                encoderChan=args.pn_topk_enc_chan,multiModel=args.resnet_multi_model,multiModSparseConst=args.resnet_multi_model_sparse_const,predictDepth=args.pn_topk_pred_depth,\
                                layerSizeReduce=args.resnet_layer_size_reduce,preLayerSizeReduce=args.resnet_prelay_size_reduce,dilation=args.resnet_dilation,\
                                topk_fps=args.pn_topk_farthest_pts_sampling,topk_fps_nb_pts=args.pn_topk_fps_nb_points,topk_dropout=args.pn_topk_dropout)
        secondModel = Identity(args.video_mode,nbFeat,args.class_nb,False)
    elif args.temp_mod == "identity":
        secondModel = Identity(args.video_mode,nbFeat,args.class_nb,False)
    else:
        raise ValueError("Unknown temporal model type : ",args.temp_mod)

    ############### Whole Model ##########################

    if args.spat_transf:
        spatTransf = SpatialTransformer(args.img_size,args.spat_transf_img_size)
    else:
        spatTransf = None

    net = Model(firstModel,secondModel,spatTransf=spatTransf)


    if args.temp_mod == "pointnet2":
        net = net.float()

    if args.cuda:
        net.cuda()

    if args.multi_gpu:
        net = DataParallelModel(net)

    #net.to("cuda" if args.cuda else "cpu")

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='MOD',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--temp_mod', type=str,metavar='MOD',
                        help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--lstm_lay', type=int,metavar='N',
                        help='Number of layers for the lstm temporal model')

    argreader.parser.add_argument('--lstm_hid_size', type=int,metavar='N',
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

    argreader.parser.add_argument('--pn_point_nb', type=int, metavar='NB',
                        help='For the topk point net model. This is the number of point extracted for each image.')
    argreader.parser.add_argument('--pn_topk_reconst', type=args.str2bool, metavar='BOOL',
                        help='For the topk point net model. An input image reconstruction term will added to the loss function if True.')
    argreader.parser.add_argument('--pn_topk_enc_chan', type=int, metavar='NB',
                        help='For the topk point net model. This is the number of output channel of the encoder')
    argreader.parser.add_argument('--pn_topk_hid_chan', type=int, metavar='NB',
                        help='For the topk point net model. This is the number of hidden channel of the encoder')
    argreader.parser.add_argument('--pn_topk_pred_depth', type=args.str2bool, metavar='INT',
                        help='For the pointnet2 model. To predict the depth of chosen points.')
    argreader.parser.add_argument('--pn_use_xyz', type=args.str2bool, metavar='INT',
                        help='For the pointnet2 model. To use the point coordinates as feature.')
    argreader.parser.add_argument('--pn_topk_dropout', type=float, metavar='FLOAT',
                        help='The proportion of point to randomly drop to decrease overfitting.')

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

    argreader.parser.add_argument('--resnet_att_chan', type=int, metavar='INT',
                        help='For the \'resnetX_att\' feat models. The number of channels in the attention module.')
    argreader.parser.add_argument('--resnet_att_blocks_nb', type=int, metavar='INT',
                        help='For the \'resnetX_att\' feat models. The number of blocks in the attention module.')
    argreader.parser.add_argument('--resnet_att_act_func', type=str, metavar='INT',
                        help='For the \'resnetX_att\' feat models. The activation function for the attention weights. Can be "sigmoid", "relu" or "tanh+relu".')

    return argreader

if __name__ == "__main__":

    import pims
    import cv2
    import numpy as np
    from skimage.transform import resize
    from scipy import ndimage

    def sobelFunc(x):
        img = np.array(x).astype('int32')
        dx = ndimage.sobel(img, 0)  # horizontal derivative
        dy = ndimage.sobel(img, 1)  # vertical derivative
        mag = np.hypot(dx, dy)  # magnitude
        mag *= 255.0 / np.max(mag)  # normalize (Q&D)
        mag= mag.astype("uint8")
        return mag

    abs = torch.tensor([[56]])
    ord = torch.tensor([[77]])
    x = sobelFunc(resize(pims.Video("../data/big/AA83-7.avi")[0],(125,125))*255)
    cv2.imwrite("in.png",x)

    x[ord[0],abs[0],0] = 255
    x[ord[0],abs[0],1:] = 0
    cv2.imwrite("in_masked.png",x)
    print(abs,ord,x[ord,abs,0])
    x = torch.tensor(x).float().permute(2,0,1).unsqueeze(0)

    softAbs,softOrd = fastSoftCoordRefiner(x,abs.float(),ord.float(),kerSize=5)
    print(softAbs,softOrd)
