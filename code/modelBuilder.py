import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet
import resnet3D
import vgg
import args
import sys

def buildFeatModel(featModelName,pretrainedFeatMod,featMap=False,bigMaps=False):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("resnet") != -1:
        featModel = getattr(resnet,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    elif featModelName == "r2plus1d_18":
        featModel = getattr(resnet3D,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    elif featModelName.find("vgg") != -1:
        featModel = getattr(vgg,featModelName)(pretrained=pretrainedFeatMod,featMap=featMap,bigMaps=bigMaps)
    else:
        raise ValueError("Unknown model type : ",featModelName)

    return featModel

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

    def __init__(self,visualModel,tempModel):
        super(Model,self).__init__()
        self.visualModel = visualModel
        self.tempModel = tempModel

        self.transMat = torch.zeros((self.tempModel.nbClass,self.tempModel.nbClass))
        self.priors = torch.zeros((self.tempModel.nbClass))

    def forward(self,x,timeElapsed=None):
        x = self.visualModel(x)["x"]
        x = self.tempModel(x,self.visualModel.batchSize,timeElapsed)
        return x

    def setTransMat(self,transMat):
        self.transMat = transMat
    def setPriors(self,priors):
        self.priors = priors

################################# Visual Model ##########################""

class VisualModel(nn.Module):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False):
        super(VisualModel,self).__init__()

        self.featMod = buildFeatModel(featModelName,pretrainedFeatMod,featMap,bigMaps)
        self.featMap = featMap
        self.bigMaps = bigMaps
    def forward(self,x):
        raise NotImplementedError

class CNN2D(VisualModel):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False):
        super(CNN2D,self).__init__(featModelName,pretrainedFeatMod,featMap,bigMaps)

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        # NT x C x H x L
        x = self.featMod(x)
        # NT x D
        return {'x':x}

class CNN3D(VisualModel):

    def __init__(self,featModelName,pretrainedFeatMod=True,featMap=False,bigMaps=False):
        super(CNN3D,self).__init__(featModelName,pretrainedFeatMod,featMap,bigMaps)

    def forward(self,x):
        # N x T x C x H x L
        self.batchSize = x.size(0)
        x = x.permute(0,2,1,3,4)
        # N x C x T x H x LpretrainedFeatModpretrainedFeatMod

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

class Attention(VisualModel):

    def __init__(self,featModelName,pretrainedFeatMod,bigMaps,attKerSize,nbFeat,nbClass):
        super(Attention,self).__init__(featModelName,pretrainedFeatMod,True,bigMaps)

        self.classConv = nn.Conv2d(nbFeat,nbClass,1)
        self.attention = nn.Conv2d(nbClass,nbClass,attKerSize,padding=attKerSize//2,groups=nbClass)
        self.nbClass = nbClass

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        x = self.featMod(x)
        x = self.classConv(x)
        attWeight = torch.sigmoid(self.attention(x))
        x = x*attWeight
        x = x.sum(dim=-1).sum(dim=-1)

        #x = x.view(batchSize,-1,self.nbClass)
        #attWeight = attWeight.view(batchSize,-1,self.nbClass,attWeight.size(-2),attWeight.size(-1))

        return {"x":x,"attention":attWeight}


################################ Temporal Model ########################""

class TempModel(nn.Module):

    def __init__(self,nbFeat,nbClass,regression,useTime):
        super(TempModel,self).__init__()
        self.nbFeat,self.nbClass,self.regression = nbFeat,nbClass,regression
        self.useTime = useTime

        if useTime:
            self.nbFeat += 1

    def forward(self,x):
        raise NotImplementedError

    def catWithTimeFeat(self,x,timeTensor):
        if self.useTime:
            timeTensor = timeTensor.view(-1).unsqueeze(1)
            x = torch.cat((x,timeTensor),dim=-1)
        return x

class LinearTempModel(TempModel):

    def __init__(self,nbFeat,nbClass,regression,useTime,dropout):
        super(LinearTempModel,self).__init__(nbFeat,nbClass,regression,useTime)

        self.dropout = nn.Dropout(p=dropout)
        if regression:
            self.linLay = nn.Linear(self.nbFeat,1)
        else:
            self.linLay = nn.Linear(self.nbFeat,self.nbClass)

    def forward(self,x,batchSize,timeTensor=None):
        x = self.catWithTimeFeat(x,timeTensor)

        # NT x D
        x = self.dropout(x)
        x = self.linLay(x)
        # NT x classNb
        if self.regression:
            # N x T
            x = x.view(batchSize,-1)
        else:
            #N x T x classNb
            x = x.view(batchSize,-1,self.nbClass)

        return {"pred":x}

class LSTMTempModel(TempModel):

    def __init__(self,nbFeat,nbClass,regression,useTime,dropout,nbLayers,nbHidden):
        super(LSTMTempModel,self).__init__(nbFeat,nbClass,regression,useTime)

        self.lstmTempMod = nn.LSTM(input_size=self.nbFeat,hidden_size=nbHidden,num_layers=nbLayers,batch_first=True,dropout=dropout,bidirectional=True)
        self.linTempMod = LinearTempModel(nbFeat=nbHidden*2,nbClass=self.nbClass,regression=regression,useTime=False,dropout=dropout)

    def forward(self,x,batchSize,timeTensor):
        x = self.catWithTimeFeat(x,timeTensor)

        # NT x D
        x = x.view(batchSize,-1,x.size(-1))
        # N x T x D
        x,_ = self.lstmTempMod(x)
        # N x T x H
        x = x.contiguous().view(-1,x.size(-1))
        # NT x H
        x = self.linTempMod(x,batchSize)
        # N x T x classNb (or N x T in case of regression)
        return {"pred":x}

class ScoreConvTempModel(TempModel):

    def __init__(self,nbFeat,nbClass,regression,useTime,dropout,kerSize,chan,biLay,attention):

        super(ScoreConvTempModel,self).__init__(nbFeat,nbClass,regression,useTime)

        self.linTempMod = LinearTempModel(nbFeat=self.nbFeat,nbClass=self.nbClass,regression=regression,useTime=False,dropout=dropout)
        self.scoreConv = ScoreConv(kerSize,chan,biLay,attention,regression)

    def forward(self,x,batchSize,timeTensor):
        x = self.catWithTimeFeat(x,timeTensor)

        # NT x D
        x = self.linTempMod(x,batchSize)
        # N x T x classNb (or N x T in case of regression)
        x = x.unsqueeze(1)
        # N x 1 x T x classNb (or N x 1 x T in case of regression)
        if self.regression:
            x = x.unsqueeze(3)
            #N x 1 x T x 1

        x = self.scoreConv(x)
        # N x 1 x T x classNb (or N x 1 x T x 1 in case of regression)
        x = x.squeeze(1)
        # N x T x classNb (or N x T x 1 in case of regression)
        if self.regression:
            x = x.squeeze(2)
            #N x T

        return {"pred":x}

class ScoreConv(nn.Module):
    ''' This is a module that reads the classes scores just before they are passed to the softmax by
    the temporal model. It apply one or two convolution layers to the signal and uses 1x1 convolution to
    outputs a transformed signal of the same shape as the input signal.

    It can return this transformed signal and can also returns this transformed signal multiplied
    by the input, like an attention layer.

    Args:
    - kerSize (int): the kernel size of the convolution(s)
    - chan (int): the number of channel when using two convolutions
    - biLay (bool): whether or not to apply two convolutional layers instead of one
    - attention (bool): whether or not to multiply the transformed signal by the input before returning it
    - regression (bool): True if the problem is treated as regression (i.e. the model has only one output)

    '''

    def __init__(self,kerSize,chan,biLay,attention,regression):

        super(ScoreConv,self).__init__()

        self.attention = attention

        if regression:
            kerSize = (kerSize,1)
        else:
            kerSize = (kerSize,kerSize)

        if biLay:
            self.conv1 = torch.nn.Conv2d(1,chan,kerSize,padding=(kerSize[0]//2,kerSize[1]//2))
            self.conv2 = torch.nn.Conv2d(chan,1,1)
            self.layers = nn.Sequential(self.conv1,nn.ReLU(),self.conv2)
        else:
            self.layers = torch.nn.Conv2d(1,1,kerSize,padding=(kerSize[0]//2,kerSize[1]//2))

    def forward(self,x):

        if not self.attention:
            return self.layers(x)
        else:
            weights = self.layers(x)
            return weights*x

class Identity(TempModel):

    def __init__(self,nbFeat,nbClass,regression,useTime):

        super(Identity,self).__init__(nbFeat,nbClass,regression,useTime)

    def forward(self,x,batchSize,timeTensor):

        x = x.view(batchSize,-1,self.nbClass)

        return {"pred":x}

def netBuilder(args):

    ############### Visual Model #######################
    if args.feat.find("resnet") != -1:
        if args.feat=="resnet50" or args.feat=="resnet101" or args.feat=="resnet151":
            nbFeat = 256*2**(4-1)
        else:
            nbFeat = 64*2**(4-1)
        visualModel = CNN2D(args.feat,args.pretrained_visual)
    elif args.feat.find("vgg") != -1:
        nbFeat = 4096
        visualModel = CNN2D(args.feat,args.pretrained_visual)
    elif args.feat == "r2plus1d_18":
        nbFeat = 512
        visualModel = CNN3D(args.feat,args.pretrained_visual)
    else:
        raise ValueError("Unknown visual model type : ",args.feat)

    if args.freeze_visual:
        for param in visualModel.parameters():
            param.requires_grad = False

    ############### Temporal Model #######################
    if args.temp_mod == "lstm":
        tempModel = LSTMTempModel(nbFeat,args.class_nb,args.regression,args.use_time,args.dropout,args.lstm_lay,args.lstm_hid_size)
    elif args.temp_mod == "linear":
        tempModel = LinearTempModel(nbFeat,args.class_nb,args.regression,args.use_time,args.dropout)
    elif args.temp_mod == "score_conv":
        tempModel = ScoreConvTempModel(nbFeat,args.class_nb,args.regression,args.use_time,args.dropout,args.score_conv_ker_size,args.score_conv_chan,args.score_conv_bilay,args.score_conv_attention)
    elif args.temp_mod == "feat_attention":
        visualModel = Attention(args.feat,args.pretrained_visual,args.feat_attention_big_maps,args.feat_attention_ker_size,nbFeat,args.class_nb)
        tempModel = Identity(nbFeat,args.class_nb,False,False)
    else:
        raise ValueError("Unknown temporal model type : ",args.temp_mod)

    ############### Whole Model ##########################

    net = Model(visualModel,tempModel)

    if args.multi_gpu:
        net = DataParallelModel(net)

    return net

def addArgs(argreader):

    argreader.parser.add_argument('--feat', type=str, metavar='MOD',
                        help='the net to use to produce feature for each frame')

    argreader.parser.add_argument('--dropout', type=float,metavar='D',
                        help='The dropout amount on each layer of the RNN except the last one')

    argreader.parser.add_argument('--regression', type=args.str2bool,metavar='D',
                        help='Set to True to train a regression model instead of a discriminator')

    argreader.parser.add_argument('--uncertainty', type=args.str2bool,metavar='D',
                        help='Set to True to also model prediction uncertainty and data uncertainty')

    argreader.parser.add_argument('--temp_mod', type=str,metavar='MOD',
                        help='The temporal model. Can be "linear", "lstm" or "score_conv".')

    argreader.parser.add_argument('--lstm_lay', type=int,metavar='N',
                        help='Number of layers for the lstm temporal model')

    argreader.parser.add_argument('--lstm_hid_size', type=int,metavar='N',
                        help='Size of hidden layers for the lstm temporal model')

    argreader.parser.add_argument('--score_conv_ker_size', type=int, metavar='N',
                        help='The size of the 2d convolution kernel to apply on scores if temp model is a ScoreConvTempModel.')

    argreader.parser.add_argument('--score_conv_bilay', type=args.str2bool, metavar='N',
                        help='To apply two convolution (the second is a 1x1 conv) on the scores instead of just one layer')

    argreader.parser.add_argument('--score_conv_attention', type=args.str2bool, metavar='BOOL',
                        help='To apply the score convolution(s) as an attention layer.')

    argreader.parser.add_argument('--score_conv_chan', type=int, metavar='N',
                        help='The number of channel of the score convolution layer (used only if --score_conv_bilay')

    argreader.parser.add_argument('--freeze_visual', type=args.str2bool, metavar='BOOL',
                        help='To freeze the weights of the visual model during training.')

    argreader.parser.add_argument('--use_time', type=args.str2bool, metavar='BOOL',
                        help='To use the time elapsed of each image as a feature')

    argreader.parser.add_argument('--feat_attention_ker_size', type=int, metavar='BOOL',
                        help='The kernel size of the feature attention.')

    argreader.parser.add_argument('--feat_attention_big_maps', type=args.str2bool, metavar='BOOL',
                        help='To make the feature and the attention maps bigger.')

    argreader.parser.add_argument('--pretrained_visual', type=args.str2bool, metavar='BOOL',
                        help='To have a visual feature extractor pretrained on ImageNet.')

    return argreader
