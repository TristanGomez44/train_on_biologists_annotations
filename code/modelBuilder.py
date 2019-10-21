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

class LSTMTempModel(TempModel):

    def __init__(self,nbFeat,nbClass,dropout,nbLayers,nbHidden):
        super(LSTMTempModel,self).__init__(nbFeat,nbClass)

        self.lstmTempMod = nn.LSTM(input_size=self.nbFeat,hidden_size=nbHidden,num_layers=nbLayers,batch_first=True,dropout=dropout,bidirectional=True)
        self.linTempMod = LinearTempModel(nbFeat=nbHidden*2,nbClass=self.nbClass,dropout=dropout)

    def forward(self,x,batchSize):
        # NT x D
        x = x.view(batchSize,-1,x.size(-1))
        # N x T x D
        x,_ = self.lstmTempMod(x)
        # N x T x H
        x = x.view(-1,x.size(-1))
        # NT x H
        x = self.linTempMod(x,batchSize)
        # N x T x classNb
        return x

def netBuilder(args):

    ############### Visual Model #######################
    if args.feat.find("resnet") != -1:
        if args.feat=="resnet50" or args.feat=="resnet101" or args.feat=="resnet151":
            nbFeat = 256*2**(4-1)
        else:
            nbFeat = 64*2**(4-1)
        visualModel = CNN2D(args.feat)
    elif args.feat.find("vgg") != -1:
        nbFeat = 4096
        visualModel = CNN2D(args.feat)
    elif args.feat == "r2plus1d_18":
        nbFeat = 512
        visualModel = CNN3D(args.feat)
    else:
        raise ValueError("Unknown visual model type : ",args.feat)

    ############### Temporal Model #######################
    if args.temp_mod == "lstm":
        tempModel = LSTMTempModel(nbFeat,args.class_nb,args.dropout,args.lstm_lay,args.lstm_hid_size)
    elif args.temp_mod == "linear":
        tempModel = LinearTempModel(nbFeat,args.class_nb,args.dropout)
    else:
        raise ValueError("Unknown temporal model type : ",args.temp_mod)

    ############### Whole Model ##########################
    net = Model(visualModel,tempModel)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='MOD',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--temp_mod', type=str,metavar='MOD',
                        help='The temporal model. Can be "linear" or "lstm".')

    argreader.parser.add_argument('--lstm_lay', type=int,metavar='N',
                        help='Number of layers for the lstm temporal model')

    argreader.parser.add_argument('--lstm_hid_size', type=int,metavar='N',
                        help='Size of hidden layers for the lstm temporal model')

    return argreader
