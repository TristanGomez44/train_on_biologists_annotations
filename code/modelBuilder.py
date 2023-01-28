import math

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Identity
plt.switch_backend('agg')

from models import resnet
from models import inter_by_parts
from models import abn
import args
EPS = 0.000001

import torch.nn.functional as F

def buildFeatModel(featModelName, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if featModelName.find("resnet") != -1:
        featModel = getattr(resnet, featModelName)(**kwargs)
    else:
        raise ValueError("Unknown model type : ", featModelName)

    return featModel

class GradCamMod(torch.nn.Module):
    def __init__(self,net):
        super().__init__()
        self.net = net
        self.layer4 = net.firstModel.featMod.layer4
        self.features = net.firstModel.featMod

    def forward(self,x):
        feat = self.net.firstModel.featMod(x)["x"]

        x = torch.nn.functional.adaptive_avg_pool2d(feat,(1,1))
        x = x.view(x.size(0),-1)
        x = self.net.secondModel.linLay(x)

        return x

# This class is just the class nn.DataParallel that allow running computation on multiple gpus
# but it adds the possibility to access the attribute of the model
class DataParallelModel(nn.DataParallel):
    def __init__(self, model):
        super().__init__(model)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

class Model(nn.Module):

    def __init__(self, firstModel, secondModel):
        super().__init__()
        self.firstModel = firstModel
        self.secondModel = secondModel

    def forward(self, origImgBatch):

        if not self.firstModel is None:

            visResDict = self.firstModel(origImgBatch)

            resDict = self.secondModel(visResDict)

            if visResDict != resDict:
                resDict = merge(visResDict,resDict)

        else:
            resDict = self.secondModel(origImgBatch)

        return resDict

def merge(dictA,dictB,suffix=""):
    for key in dictA.keys():
        if key in dictB:
            dictB[key+"_"+suffix] = dictA[key]
        else:
            dictB[key] = dictA[key]
    return dictB

################################# Visual Model ##########################

class FirstModel(nn.Module):

    def __init__(self, featModelName,**kwargs):
        super().__init__()

        self.featMod = buildFeatModel(featModelName,**kwargs)

    def forward(self, x):
        raise NotImplementedError

class CNN2D(FirstModel):

    def __init__(self, featModelName,**kwargs):
        super().__init__(featModelName,**kwargs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        retDict = self.featMod(x)

        retDict["feat_pooled"] = self.avgpool(retDict["feat"]).squeeze(-1).squeeze(-1)

        return retDict

def buildImageAttention(inFeat,outChan=1):
    attention = []
    attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

def representativeVectors(x,nbVec,no_refine=False,randVec=False):

    xOrigShape = x.size()

    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    if randVec:
        raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
    else:
        raw_reprVec_score = norm.clone()

    repreVecList = []
    simList = []
    for _ in range(nbVec):
        _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
        raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)

        simNorm = sim/sim.sum(dim=1,keepdim=True)

        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)

        if not no_refine:
            repreVecList.append(reprVec)
        else:
            repreVecList.append(raw_reprVec[:,0])

        if randVec:
            raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
        else:
            raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    return repreVecList,simList

class CNN2D_bilinearAttPool(FirstModel):

    def __init__(self, featModelName,inFeat=512,nb_parts=3,cluster=False,no_refine=False,rand_vec=False,**kwargs):

        super().__init__(featModelName,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.nb_parts = nb_parts
        self.cluster = cluster

        if not cluster:
            self.attention = buildImageAttention(inFeat,nb_parts+1)
            self.attention_activation = torch.relu
        else:
            self.attention = None
            self.attention_activation = None
            self.no_refine = no_refine
            self.rand_vec = rand_vec

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        retDict = self.featMod(x)

        features = retDict["feat"]
  
        if not self.cluster:
            spatialWeights = self.attention_activation(self.attention(features))
            features_weig = (spatialWeights[:,:self.nb_parts].unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),features.size(1)*(spatialWeights.size(1)-1),features.size(2),features.size(3))
            features_agr = self.avgpool(features_weig)
            features_agr = features_agr.view(features.size(0), -1)
        else:

            vecList,simList = representativeVectors(features,self.nb_parts,self.no_refine,self.rand_vec)

            features_agr = torch.cat(vecList,dim=-1)
            spatialWeights = torch.cat(simList,dim=1)

        retDict["feat_pooled"] = features_agr
        retDict["attMaps"] = spatialWeights

        return retDict

class CNN2D_interbyparts(FirstModel):

    def __init__(self,featModelName,classNb,nb_parts,**kwargs):

        super().__init__(featModelName,**kwargs)

        self.featMod = inter_by_parts.ResNet50(classNb,nb_parts)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self,x):

        pred,att,features = self.featMod(x)
        
        return {"pred":pred,"attMaps":att,"feat":features,"feat_pooled":self.avgpool(features)}

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super().__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbClass, dropout,bias=True):

        super().__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)

        self.linLay = nn.Linear(self.nbFeat, self.nbClass,bias=bias)

    def forward(self, retDict):
        x = retDict["feat_pooled"]
        x = self.dropout(x)
        pred = self.linLay(x)
        retDict["pred"]=pred
        return retDict

def getResnetFeat(backbone_name, backbone_inplanes):
    if backbone_name in ["resnet50","resnet101","resnet152"]:
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("resnet34") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name == "hrnet44":
        nbFeat = 44
    elif backbone_name == "hrnet64":
        nbFeat = 64
    elif backbone_name == "hrnet18":
        nbFeat = 16
    elif backbone_name.find("efficientnet") != -1:
        nbFeat = 1792
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def getResnetDownSampleRatio(args):
    backbone_name = args.first_mod
    if backbone_name.find("resnet") != -1:
        ratio = 32
        for stride in [args.stride_lay2,args.stride_lay3,args.stride_lay4]:
            if stride == 1:
                ratio /= 2

        return int(ratio)

    raise ValueError("Unkown backbone",backbone_name)

def netBuilder(args,gpu=None):
    ############### Visual Model #######################

    nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)

    if not args.prototree and not args.protonet and not args.abn:
        if args.resnet_bilinear:
            CNNconst = CNN2D_bilinearAttPool
            kwargs = {"inFeat":nbFeat,"nb_parts":args.resnet_bil_nb_parts,\
                        "cluster":args.bil_cluster,
                        "no_refine":args.bil_cluster_norefine,\
                        "rand_vec":args.bil_cluster_randvec}

            nbFeat *= args.resnet_bil_nb_parts
 
        elif args.inter_by_parts:
            CNNconst = CNN2D_interbyparts
            kwargs = {"classNb":args.class_nb,"nb_parts":args.resnet_bil_nb_parts}
        else:
            CNNconst = CNN2D
            kwargs = {}

        if args.first_mod.find("bagnet") == -1 and args.first_mod.find("hrnet") == -1:
            firstModel = CNNconst(args.first_mod,chan=args.resnet_chan, stride=args.resnet_stride,\
                                    strideLay2=args.stride_lay2,strideLay3=args.stride_lay3,\
                                    strideLay4=args.stride_lay4,\
                                    endRelu=args.end_relu,\
                                    **kwargs)
        else:
            firstModel = CNNconst(args.first_mod,**kwargs)


        ############### Second Model #######################
        if not args.inter_by_parts:
            if args.second_mod == "linear":
                secondModel = LinearSecondModel(nbFeat, args.class_nb, args.dropout,args.lin_lay_bias)
            else:
                raise ValueError("Unknown second model type : ", args.second_mod)
        else:
            secondModel = Identity()

        ############### Whole Model ##########################

        net = Model(firstModel, secondModel)
    else:
        net = abn.resnet50(fullpretrained=args.abn_pretrained)

    if args.distributed:
        torch.cuda.set_device(gpu)
        net.cuda(gpu)
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[gpu],find_unused_parameters=True)
    else:
        if args.cuda and torch.cuda.is_available():
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

    argreader.parser.add_argument('--resnet_chan', type=int, metavar='INT',
                                  help='The channel number for the visual model when resnet is used')
    argreader.parser.add_argument('--resnet_stride', type=int, metavar='INT',
                                  help='The stride for the visual model when resnet is used')

    argreader.parser.add_argument('--stride_lay2', type=int, metavar='NB',
                                  help='Stride for layer 2.')
    argreader.parser.add_argument('--stride_lay3', type=int, metavar='NB',
                                  help='Stride for layer 3.')
    argreader.parser.add_argument('--stride_lay4', type=int, metavar='NB',
                                  help='Stride for layer 4.')

    argreader.parser.add_argument('--resnet_bil_nb_parts', type=int, metavar='INT',
                                  help="The number of parts for the bilinear model.")
    argreader.parser.add_argument('--resnet_bilinear', type=args.str2bool, metavar='BOOL',
                                  help="To use bilinear attention")

    argreader.parser.add_argument('--bil_cluster', type=args.str2bool, metavar='BOOL',
                                  help="To have a cluster bilinear")
    argreader.parser.add_argument('--bil_cluster_ensemble', type=args.str2bool, metavar='BOOL',
                                  help="To classify each of the feature vector obtained and then aggregates those decision.")
    argreader.parser.add_argument('--apply_softmax_on_sim', type=args.str2bool, metavar='BOOL',
                                  help="Apply softmax on similarity computed during clustering.")
    argreader.parser.add_argument('--softm_coeff', type=float, metavar='BOOL',
                                  help="The softmax temperature. The higher it is, the sharper weights will be.")
    argreader.parser.add_argument('--bil_clust_unnorm', type=args.str2bool, metavar='BOOL',
                                  help="To mulitply similarity by norm to make weights superior to 1.")
    argreader.parser.add_argument('--bil_clus_vect_gate', type=args.str2bool, metavar='BOOL',
                                  help="To add a gate that reorder the vectors.")

    argreader.parser.add_argument('--bil_clus_vect_ind_to_use',type=str, metavar='BOOL',
                                  help="Specify this list to only use some of the vectors collected. Eg : --bil_clus_vect_ind_to_use 1,2")

    argreader.parser.add_argument('--bil_clust_update_sco_by_norm_sim', type=args.str2bool, metavar='BOOL',
                                  help="To update score using normalised similarity.")

    argreader.parser.add_argument('--bil_cluster_ensemble_gate', type=args.str2bool, metavar='BOOL',
                                  help="To add a gate network at the end of the cluster ensemble network.")
    argreader.parser.add_argument('--bil_cluster_ensemble_gate_drop', type=args.str2bool, metavar='BOOL',
                                  help="To drop the feature vector with the most important weight.")
    argreader.parser.add_argument('--bil_cluster_ensemble_gate_randdrop', type=args.str2bool, metavar='BOOL',
                                  help="To randomly drop one feature vector.")

    argreader.parser.add_argument('--bil_cluster_norefine', type=args.str2bool, metavar='BOOL',
                                  help="To not refine feature vectors by using similar vectors.")
    argreader.parser.add_argument('--bil_cluster_randvec', type=args.str2bool, metavar='BOOL',
                                  help="To select random vectors as initial estimation instead of vectors with high norms.")

    argreader.parser.add_argument('--bil_cluster_lay_ind', type=int, metavar='BOOL',
                                  help="The layer at which to group pixels.")

    argreader.parser.add_argument('--protonet', type=args.str2bool, metavar='BOOL',
                                  help="To train a protonet model")
    argreader.parser.add_argument('--proto_nb', type=int, metavar='BOOL',
                                  help="The nb of prototypes per class.")
    argreader.parser.add_argument('--protonet_warm', type=int, metavar='BOOL',
                                  help="Warmup epoch number")
    argreader.parser.add_argument('--prototree', type=args.str2bool, metavar='BOOL',
                                  help="To train a prototree model")

    argreader.parser.add_argument('--inter_by_parts', type=args.str2bool, metavar='BOOL',
                                  help="To train the model from https://github.com/zxhuang1698/interpretability-by-parts/tree/650f1af573075a41f04f2f715f2b2d4bc0363d31")

    argreader.parser.add_argument('--lin_lay_bias', type=args.str2bool, metavar='BOOL',
                                  help="To add a bias to the final layer.")

    argreader.parser.add_argument('--aux_on_masked', type=args.str2bool, metavar='BOOL',
                                  help="To train dense layers on masked version of the feature matrix.")

    argreader.parser.add_argument('--master_net', type=args.str2bool, help='To distill a master network into the trained network.')
    argreader.parser.add_argument('--m_model_id', type=str, help='The model id of the master network')
    argreader.parser.add_argument('--kl_interp', type=float, help='If set to 0, will use regular target, if set to 1, will only use master net target')
    argreader.parser.add_argument('--kl_temp', type=float, help='KL temperature.')

    argreader.parser.add_argument('--transfer_att_maps', type=args.str2bool, help='To also transfer attention maps during distillation.')
    argreader.parser.add_argument('--att_weights', type=float, help='Attention map transfer weight.')
    argreader.parser.add_argument('--att_pow', type=int, help='The power at which to compute the difference between the maps.')
    argreader.parser.add_argument('--att_term_included', type=args.str2bool, help='To force the studen att maps to be included in the teach att maps.')
    argreader.parser.add_argument('--att_term_reg', type=args.str2bool, help='To force the student att maps to be centered where the teach maps are centered.')

    argreader.parser.add_argument('--end_relu', type=args.str2bool, help='To add a relu at the end of the first block of each layer.')
    argreader.parser.add_argument('--abn', type=args.str2bool, help='To train an attention branch network')
    argreader.parser.add_argument('--abn_pretrained', type=args.str2bool, help='To load imagenet pretrained weights for ABN.')

    return argreader
