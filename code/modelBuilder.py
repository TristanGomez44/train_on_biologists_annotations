import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet
import resnet3D
import vgg
import args
import sys

def buildFeatModel(featModelName):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("resnet") != -1:
        featModel = getattr(resnet,featModelName)(pretrained=True)
    elif featModelName == "r2plus1d_18":
        featModel = getattr(resnet3D,featModelName)(pretrained=True)
    elif featModelName.find("vgg") != -1:
        featModel = getattr(vgg,featModelName)(pretrained=True)
    else:
        raise ValueError("Unknown model type : ",featModelName)

    return featModel

'''
class CNN(nn.Module):

    def __init__(self,featModelName="resnet50",dropout=0.5,classNb=16):

        super(CNN,self).__init__()

        self.featMod = buildFeatModel(featModelName)
        self.classNb = classNb

        if featModelName=="resnet50" or featModelName=="resnet101" or featModelName=="resnet151":
            self.nbFeat = 256*2**(4-1)
        elif featModelName.find("vgg") != -1:
            self.nbFeat = 4096
        else:
            self.nbFeat = 64*2**(4-1)

        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat,classNb)

        self.transMat = torch.zeros((classNb,classNb))

    def forward(self,x):
        # N x T x C x H x L
        x = self.computeFeat(x)
        # NT x D
        x = self.computeScore(x)
        # N x T x classNb
        return x

    def computeFeat(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        # NT x C x H x L
        x = self.featMod(x)
        # NT x D
        return x

    def computeScore(self,x):
        # NT x D
        x = self.dropout(x)
        x = self.linLay(x)
        # NT x classNb
        x = x.view(self.batchSize,-1,self.classNb)
        # N x T x classNb
        return x

class CNN3D(CNN):

    def __init__(self,featModelName,dropout=0.5,classNb=16):

        super(CNN3D,self).__init__(featModelName,dropout,classNb)

        #The r2plus1d_18 architecture has 512 features
        self.nbFeat = 512

        self.linLay = nn.Linear(self.nbFeat,classNb)

    def computeFeat(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.permute(0,2,1,3,4)
        # N x C x T x H x L
        x = self.featMod(x)
        # N x D x T
        x = x.permute(0,2,1)
        # N x T x D
        x = x.contiguous().view(x.size(0)*x.size(1),-1)
        # NT x D
        return x
'''
class Model(nn.Module):

    def __init__(self,visualModel,tempModel):
        super(Model,self).__init__()
        self.visualModel = visualModel
        self.tempModel = tempModel

        self.transMat = torch.zeros((self.tempModel.nbClass,self.tempModel.nbClass))

    def forward(self,x):
        x = self.visualModel(x)
        x = self.tempModel(x,self.visualModel.batchSize)
        return x

class VisualModel(nn.Module):

    def __init__(self,featModelName):
        super(VisualModel,self).__init__()

        self.featMod = buildFeatModel(featModelName)

    def forward(self,x):
        raise NotImplementedError

class CNN2D(VisualModel):

    def __init__(self,featModelName):
        super(CNN2D,self).__init__(featModelName)

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        # NT x C x H x L
        x = self.featMod(x)
        # NT x D
        return x

class CNN3D(VisualModel):

    def __init__(self,featModelName):
        super(CNN3D,self).__init__(featModelName)

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.permute(0,2,1,3,4)
        # N x C x T x H x L
        x = self.featMod(x)
        # N x D x T
        x = x.permute(0,2,1)
        # N x T x D
        x = x.contiguous().view(x.size(0)*x.size(1),-1)
        # NT x D
        return x

class TempModel(nn.Module):

    def __init__(self,nbFeat,nbClass):
        super(TempModel,self).__init__()
        self.nbFeat,self.nbClass = nbFeat,nbClass

    def forward(self,x):
        raise NotImplementedError

class LinearTempModel(TempModel):

    def __init__(self,nbFeat,nbClass,dropout):
        super(LinearTempModel,self).__init__(nbFeat,nbClass)

        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat,self.nbClass)

    def forward(self,x,batchSize):
        # NT x D
        x = self.dropout(x)
        x = self.linLay(x)
        # NT x classNb
        x = x.view(batchSize,-1,self.nbClass)
        # N x T x classNb
        return x

def netBuilder(args):

    if args.feat.find("resnet") != -1:
        if featModelName=="resnet50" or featModelName=="resnet101" or featModelName=="resnet151":
            nbFeat = 256*2**(4-1)
        else:
            nbFeat = 64*2**(4-1)
        visualModelConstruct = CNN2D
    elif args.feat.find("vgg") != -1:
        nbFeat = 4096
        visualModelConstruct = CNN2D
    elif args.feat == "r2plus1d_18":
        nbFeat = 512
        visualModelConstruct = CNN3D
    else:
        raise ValueError("Unknown model type : ",args.feat)

    visualModel = visualModelConstruct(args.feat)
    tempModel = LinearTempModel(nbFeat,args.class_nb,args.dropout)

    net = Model(visualModel,tempModel)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='N',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    return argreader
