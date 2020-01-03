import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import DataParallel
import resnet
import resnet3D
import vgg
import args
import sys
import pointnet2
import cv2

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

    def __init__(self,visualModel,tempModel,spatTransf=None):
        super(Model,self).__init__()
        self.visualModel = visualModel
        self.tempModel = tempModel
        self.spatTransf = spatTransf

        self.transMat = torch.zeros((self.tempModel.nbClass,self.tempModel.nbClass))
        self.priors = torch.zeros((self.tempModel.nbClass))

    def forward(self,x,timeElapsed=None):
        if self.spatTransf:
            x = self.spatTransf(x)["x"]

        x = self.visualModel(x)["x"]
        resDict = self.tempModel(x,self.visualModel.batchSize,timeElapsed)

        return resDict

    #This method can not be used when running on several gpu
    def computeVisual(self,x):
        if self.spatTransf:
            resDict = self.spatTransf(x)
            x = resDict["x"]
            theta = resDict["theta"]

        resDict = self.visualModel(x)

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

        self.visFeat = buildFeatModel("resnet4",False,featMap=True,bigMaps=False)
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
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4)).contiguous()
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

class AttentionModel(VisualModel):

    def __init__(self,featModelName,pretrainedFeatMod,bigMaps,nbFeat,nbClass,classBiasMod=None,attType="shallow",groupedAtt=True):
        super(AttentionModel,self).__init__(featModelName,pretrainedFeatMod,True,bigMaps)

        self.classConv = nn.Conv2d(nbFeat,nbClass,1)
        #self.attention = nn.Conv2d(nbClass,nbClass,attKerSize,padding=attKerSize//2,groups=nbClass)
        self.attention = Attention(attType,nbClass,nbClass,groupedAtt)
        self.nbClass = nbClass
        self.classBiasMod = classBiasMod

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
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

class AttentionFullModel(VisualModel):

    def __init__(self,featModelName,pretrainedFeatMod,bigMaps,nbFeat,nbClass,classBiasMod=None,attType="shallow",groupedAtt=True):
        super(AttentionFullModel,self).__init__(featModelName,pretrainedFeatMod,True,bigMaps)

        #self.attention = nn.Conv2d(nbFeat,nbClass,attKerSize,padding=attKerSize//2)
        self.attention = Attention(attType,nbFeat,nbClass,groupedAtt)

        self.lin = nn.Linear(nbFeat*nbClass,nbClass)
        self.nbClass = nbClass
        self.classBiasMod = classBiasMod

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
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

    def __init__(self):
        super(DirectPointExtractor,self).__init__()
        self.feat = resnet.resnet4(pretrained=True,chan=64,featMap=True)
        self.conv1x1 = nn.Conv2d(64, 32, kernel_size=1, stride=1)
        self.size_red = nn.AdaptiveAvgPool2d((8, 8))
        self.dense = nn.Linear(8*8*32,256*2)

    def forward(self,x):
        x = self.feat(x)
        x = self.conv1x1(x)
        x = self.size_red(x)
        x = x.view(x.size(0),-1)
        x = self.dense(x)
        points = x.view(x.size(0),x.size(1)//2,2)
        points = torch.cat((points,torch.zeros(points.size(0),points.size(1),1).to(x.device)),dim=-1)

        return points

class TopkPointExtractor(nn.Module):

    def __init__(self,attention,softCoord):
        super(TopkPointExtractor,self).__init__()

        self.feat = resnet.resnet4(pretrained=True,chan=64,featMap=True)
        self.conv1x1 = nn.Conv2d(64, 1, kernel_size=1, stride=1)
        self.point_extracted = 256
        self.attention = attention
        self.softCoord = softCoord
        self.softCoord_kerSize = 2

    def forward(self,img):

        x = self.feat(img)
        x = self.conv1x1(x)
        x = torch.sigmoid(x)

        if self.attention:
            #Normalizing the attention weights
            x = (x-x.min(dim=-1,keepdim=True)[0].min(dim=-1,keepdim=True)[0])/(x.max(dim=-1,keepdim=True)[0].max(dim=-1,keepdim=True)[0]-x.min(dim=-1,keepdim=True)[0].min(dim=-1,keepdim=True)[0])

            img = torch.nn.functional.adaptive_max_pool2d(img[:,0:1],(x.size(-2),x.size(-1)))
            #Shifting the image minimum to 0 so the attention weights only reduce the values in the image
            img = img-img.min(dim=-1,keepdim=True)[0].min(dim=-1,keepdim=True)[0]

            x = x*img

        flatX = x.view(x.size(0),-1)
        flatVals,flatInds = torch.topk(flatX, self.point_extracted, dim=-1, largest=True)
        abs,ord = (flatInds%x.shape[-1],flatInds//x.shape[-1])

        if self.softCoord:
            abs,ord,flatVals = softCoordRefiner(x,abs,ord,kerSize=self.softCoord_kerSize)

        abs,ord,flatVals = abs.unsqueeze(-1),ord.unsqueeze(-1),flatVals.unsqueeze(-1)
        points = torch.cat((abs,ord),dim=-1).float()
        points = torch.cat((points,torch.zeros(points.size(0),points.size(1),1).to(x.device)),dim=-1)
        points = torch.cat((points,flatVals),dim=-1)
        return points


def softCoordRefiner(x,abs,ord,kerSize=6):

    refAbsTens = torch.arange(x.size(-1)).unsqueeze(0).expand((x.size(-2),x.size(-1))).float().to(x.device)
    refOrdTens = torch.arange(x.size(-2)).unsqueeze(1).expand((x.size(-2),x.size(-1))).float().to(x.device)

    allValues,allSoftAbs,allSoftOrd = [],[],[]

    for batch_ind in range(x.size(0)):

        values,softAbs,softOrd = [],[],[]

        for i in range(len(abs[0])):

            absTens = abs[batch_ind,i].unsqueeze(0).unsqueeze(1).expand((x.size(-2),x.size(-1))).float()
            ordTens = ord[batch_ind,i].unsqueeze(0).unsqueeze(1).expand((x.size(-2),x.size(-1))).float()

            distTens = torch.abs(refAbsTens-absTens)+torch.abs(refOrdTens-ordTens)

            weightTens = torch.relu(kerSize - distTens).float()*x[:,0:1]

            softAbs.append(((weightTens*refAbsTens).sum()/weightTens.sum()).unsqueeze(0))
            softOrd.append(((weightTens*refOrdTens).sum()/weightTens.sum()).unsqueeze(0))
            values.append(((weightTens*x[batch_ind]).sum()/weightTens.sum()).unsqueeze(0))

        softAbs,softOrd,values = torch.cat(softAbs,dim=0),torch.cat(softOrd,dim=0),torch.cat(values,dim=0)

        allSoftAbs.append(softAbs.unsqueeze(0))
        allSoftOrd.append(softOrd.unsqueeze(0))
        allValues.append(values.unsqueeze(0))

    allSoftAbs,allSoftOrd,allValues = torch.cat(allSoftAbs,dim=0),torch.cat(allSoftOrd,dim=0),torch.cat(allValues,dim=0)

    return allSoftAbs,allSoftOrd,allValues

class PointNet2(VisualModel):

    def __init__(self,classNb,featModelName='resnet18',pretrainedFeatMod=True,featMap=False,bigMaps=False,topk=False,topk_attention=False,topk_softcoord=False):

        super(PointNet2,self).__init__(featModelName,pretrainedFeatMod,True,bigMaps)

        if topk:
            self.pointExtr = TopkPointExtractor(topk_attention,topk_softcoord)
        else:
            self.pointExtr = DirectPointExtractor()

        self.pn2 = pointnet2.models.pointnet2_ssg_cls.Pointnet2SSG(num_classes=classNb,input_channels=topk,use_xyz=True)

    def forward(self,x):

        self.batchSize = x.size(0)
        x = x.view(x.size(0)*x.size(1),x.size(2),x.size(3),x.size(4))
        points = self.pointExtr(x)

        x = self.pn2(points.float())
        return {"x":x,"points":points}

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
        x = self.linTempMod(x,batchSize)["pred"]
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
        x = self.linTempMod(x,batchSize)["pred"]
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
        elif args.feat == "resnet9":
            nbFeat = 8*2**(2-1)
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
    elif args.temp_mod == "feat_attention" or args.temp_mod == "feat_attention_full":

        classBiasMod = ClassBias(nbFeat,args.class_nb) if args.class_bias_model else None

        if args.temp_mod == "feat_attention":
            visualModel = AttentionModel(args.feat,args.pretrained_visual,args.feat_attention_big_maps,nbFeat,args.class_nb,classBiasMod,args.feat_attention_att_type,args.feat_attention_grouped_att)
            tempModel = Identity(nbFeat,args.class_nb,False,False)

        elif args.temp_mod == "feat_attention_full":
            visualModel = AttentionFullModel(args.feat,args.pretrained_visual,args.feat_attention_big_maps,nbFeat,args.class_nb,classBiasMod,args.feat_attention_att_type,args.feat_attention_grouped_att)
            tempModel = Identity(nbFeat,args.class_nb,False,False)
    elif args.temp_mod == "pointnet2":
        visualModel = PointNet2(args.class_nb,topk=args.pn_topk,topk_attention=args.pn_topk_attention,topk_softcoord=args.pn_topk_softcoord)
        tempModel = Identity(nbFeat,args.class_nb,False,False)
    else:
        raise ValueError("Unknown temporal model type : ",args.temp_mod)

    ############### Whole Model ##########################

    if args.spat_transf:
        spatTransf = SpatialTransformer(args.img_size,args.spat_transf_img_size)
    else:
        spatTransf = None

    net = Model(visualModel,tempModel,spatTransf=spatTransf)

    if args.multi_gpu:
        net = DataParallelModel(net)

    if args.temp_mod == "pointnet2":
        net = net.float()

    if args.cuda:
        net = net.cuda()

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

    argreader.parser.add_argument('--feat_attention_att_type', type=str, metavar='TYPE',
                        help="The attention type. Can be 'shallow' or 'deep'.")

    argreader.parser.add_argument('--feat_attention_grouped_att', type=args.str2bool, metavar='BOOl',
                        help="To use grouped convolution in the attention module.")

    argreader.parser.add_argument('--feat_attention_big_maps', type=args.str2bool, metavar='BOOL',
                        help='To make the feature and the attention maps bigger.')

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

    argreader.parser.add_argument('--pn_topk_attention', type=args.str2bool, metavar='BOOL',
                        help='For the topk point net model. The point extractor will be used as an attention module if True.')
    argreader.parser.add_argument('--pn_topk_softcoord', type=args.str2bool, metavar='BOOL',
                        help='For the topk point net model. The point coordinate will be computed using soft argmax.')

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

    abs = torch.tensor([56,47])
    ord = torch.tensor([77,10])
    x = sobelFunc(resize(pims.Video("../data/big/AA83-7.avi")[0],(125,125))*255)
    cv2.imwrite("in.png",x)

    x[ord[0],abs[0],0] = 255
    x[ord[0],abs[0],1:] = 0
    cv2.imwrite("in_masked.png",x)

    x = torch.tensor(x).float().permute(2,0,1).unsqueeze(0)
    abs,ord,values = softCoordRefiner(x,abs.float(),ord.float(),kerSize=10)

    print(abs,ord)
