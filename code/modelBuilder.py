import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet

import args

def buildFeatModel(featModelName):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''

    featModel = getattr(resnet,featModelName)(pretrained=True)

    return featModel

class CNN(nn.Module):

    def __init__(self,featModelName="resnet50",dropout=0.5,classNb=16):

        super(CNN,self).__init__()

        self.featMod = buildFeatModel(featModelName)
        self.classNb = classNb

        if featModelName=="resnet50" or featModelName=="resnet101" or featModelName=="resnet151":
            self.nbFeat = 256*2**(4-1)
        else:
            self.nbFeat = 64*2**(4-1)

        self.dropout = nn.Dropout(p=dropout)
        self.linLay = nn.Linear(self.nbFeat,classNb)

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

def netBuilder(args):
    net = CNN(args.feat,args.dropout,args.class_nb)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='N',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    return argreader
