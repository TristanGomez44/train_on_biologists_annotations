import math

import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.nn.modules.linear import Identity
plt.switch_backend('agg')

from models import resnet
from models import hrnet
from models import inception
from models import efficientnet
from models import inter_by_parts
from models import prototree
from models import protopnet
import args
import time 
import sys 
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
    elif featModelName == "hrnet44":
        featModel = hrnet.get_cls_net(w=44)
    elif featModelName == "hrnet64":
        featModel = hrnet.get_cls_net(w=64)
    elif featModelName == "hrnet18":
        featModel = hrnet.get_cls_net(w=18)
    elif featModelName == "inception":
        featModel = inception.inception_v3(pretrained=True)
    elif featModelName.find("efficientnet") != -1:
        featModel = getattr(efficientnet,featModelName)()
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

    def forward(self, x):

        # N x C x H x L
        self.batchSize = x.size(0)

        # N x C x H x L
        featModRetDict = self.featMod(x)

        features = featModRetDict["x"]

        spatialWeights = torch.pow(features, 2).sum(dim=1, keepdim=True)
        retDict = {}

        if not "attMaps" in featModRetDict.keys():
            retDict["attMaps"] = spatialWeights
            retDict["features"] = features
        else:
            retDict["attMaps"] = featModRetDict["attMaps"]
            retDict["features"] = featModRetDict["features"]

        retDict["x"] = features.mean(dim=-1).mean(dim=-1)

        return retDict

def buildImageAttention(inFeat,outChan=1):
    attention = []
    attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

def representativeVectors(x,nbVec,applySoftMax=False,softmCoeff=1,no_refine=False,randVec=False,update_sco_by_norm_sim=False,vectIndToUse="all"):

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

        if applySoftMax:
            simNorm = torch.softmax(softmCoeff*sim,dim=1)
        else:
            simNorm = sim/sim.sum(dim=1,keepdim=True)

        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)

        if not no_refine:
            repreVecList.append(reprVec)
        else:
            repreVecList.append(raw_reprVec[:,0])

        if randVec:
            raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
        else:
            if update_sco_by_norm_sim:
                raw_reprVec_score = (1-simNorm)*raw_reprVec_score
            else:
                raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = simNorm.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    if vectIndToUse == "all":
        return repreVecList,simList
    else:
        vectIndToUse = [int(ind) for ind in vectIndToUse.split(",")]
        return [repreVecList[ind] for ind in vectIndToUse],[simList[ind] for ind in vectIndToUse]

class CNN2D_bilinearAttPool(FirstModel):

    def __init__(self, featModelName,
                 inFeat=512,nb_parts=3,\
                 cluster=False,cluster_ensemble=False,applySoftmaxOnSim=False,\
                 softmCoeff=1,no_refine=False,rand_vec=False,update_sco_by_norm_sim=False,\
                 vect_gate=False,vect_ind_to_use="all",cluster_lay_ind=4,\
                 **kwargs):

        super().__init__(featModelName,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if not cluster:
            self.attention = buildImageAttention(inFeat,nb_parts+1)
        else:
            self.attention = None

        self.nb_parts = nb_parts
        self.cluster = cluster
        if not cluster:
            self.attention_activation = torch.relu
        else:
            self.attention_activation = None
            self.cluster_ensemble = cluster_ensemble
            self.applySoftmaxOnSim = applySoftmaxOnSim
            self.no_refine = no_refine
            self.rand_vec = rand_vec
            self.update_sco_by_norm_sim = update_sco_by_norm_sim

        self.softmSched_interpCoeff = 0
        self.softmCoeff = softmCoeff

        self.vect_gate = vect_gate
        if self.vect_gate:
            self.vect_gate_proto = torch.nn.Parameter(torch.zeros(nb_parts,inFeat),requires_grad=True)
            stdv = 1. / math.sqrt(self.vect_gate_proto.size(1))
            self.vect_gate_proto.data.uniform_(0, 2*stdv)

        self.vect_ind_to_use = vect_ind_to_use
        self.cluster_lay_ind = cluster_lay_ind

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        output = self.featMod(x)

        if (not self.cluster) or (self.cluster_lay_ind == 4):
            features = output["x"]
        else:
            features = output["layerFeat"][self.cluster_lay_ind]

        retDict = {}

        #features[:,:,:,:3] = 0
        #features[:,:,:,-3:] = 0

        if not self.cluster:
            spatialWeights = self.attention_activation(self.attention(features))
            features_weig = (spatialWeights[:,:self.nb_parts].unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),features.size(1)*(spatialWeights.size(1)-1),features.size(2),features.size(3))
            features_agr = self.avgpool(features_weig)
            features_agr = features_agr.view(features.size(0), -1)
        else:

            vecList,simList = representativeVectors(features,self.nb_parts,self.applySoftmaxOnSim,self.softmCoeff,\
                                                    self.no_refine,self.rand_vec,self.update_sco_by_norm_sim,self.vect_ind_to_use)

            if not self.cluster_ensemble:
                if self.vect_gate:
                    features_agr = torch.cat(vecList,dim=0)

                    if self.vect_ind_to_use == "all":
                        features_agr = features_agr.unsqueeze(1).reshape(features_agr.size(0)//self.nb_parts,self.nb_parts,features_agr.size(1))
                    else:
                        effectivePartNb = len(self.vect_ind_to_use.split(","))
                        features_agr = features_agr.unsqueeze(1).reshape(features_agr.size(0)//effectivePartNb,effectivePartNb,features_agr.size(1))

                    # (N 1 3 512) x (1 3 1 512) -> (N 3 3 1)
                    sim = (features_agr.unsqueeze(1) * self.vect_gate_proto.unsqueeze(0).unsqueeze(2)).sum(dim=-1,keepdim=True)

                    featNorm = torch.sqrt(torch.pow(features_agr,2).sum(dim=-1,keepdim=True))
                    vect_gate_proto_norm = torch.sqrt(torch.pow(self.vect_gate_proto,2).sum(dim=-1,keepdim=True))

                    sim = sim/(featNorm.unsqueeze(2) * vect_gate_proto_norm.unsqueeze(0).unsqueeze(1))

                    # (N 1 3 512) x (N 3 3 1) -> (N 3 3 512) -> (N 3 512)
                    features_agr = (features_agr.unsqueeze(1) * torch.softmax(sim,dim=-2)).sum(dim=-2)
                    features_agr = features_agr.reshape(features_agr.size(0),-1)
                else:
                    features_agr = torch.cat(vecList,dim=-1)

            else:
                features_agr = vecList

            spatialWeights = torch.cat(simList,dim=1)

        retDict["x"] = features_agr
        retDict["attMaps"] = spatialWeights
        retDict["features"] = features

        return retDict


class CNN2D_protoNet(FirstModel):

    def __init__(self, featModelName,
                 inFeat=512,nb_parts=3,protoPerClass=10,classNb=200,**kwargs):

        super().__init__(featModelName,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.featMod = protopnet.resnet50_features(pretrained=True)
        self.protopnet = protopnet.construct_PPNet(self.featMod,num_parts=nb_parts, num_classes=classNb,prototype_activation_function="linear")

        self.nb_parts = nb_parts
        self.linLay_aux = nn.Linear(2048,200)

    def forward(self, x):

        logits, min_distances,distances,features = self.protopnet(x)

        retDict = {"pred":logits,"dist":min_distances,"attMaps":distances,"features":features}

        retDict["prototype_shape"] = self.protopnet.prototype_shape
        retDict["prototype_class_identity"] = self.protopnet.prototype_class_identity
        retDict["pred_aux"] = self.linLay_aux(features.mean(dim=-1).mean(dim=-1))

        return retDict

class CNN2D_interbyparts(FirstModel):

    def __init__(self,featModelName,classNb,nb_parts,**kwargs):

        super().__init__(featModelName,**kwargs)

        self.featMod = inter_by_parts.ResNet50(classNb,nb_parts)

    def forward(self,x):

        pred,att,features = self.featMod(x)
        
        return {"pred":pred,"attMaps":att,"features":features}

class CNN2D_prototree(FirstModel):

    def __init__(self,featModelName,classNb,**kwargs):

        super().__init__(featModelName,**kwargs)

        self.mod = prototree.prototree(classNb)
        
        self.featMod = self.mod._net

        self.linLay_aux = nn.Linear(2048,200)

    def forward(self,x):

        pred,info,att,features = self.mod(x)
 
        #return {"pred":pred,"attMaps":att,"features":features,"pred_aux":self.linLay_aux(features.mean(dim=-1).mean(dim=-1)),"info":info}
        return {"pred":pred,"attMaps":att,"features":features}

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super().__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nbClass, dropout,bil_cluster_ensemble=False,\
                        bias=True,aux_on_masked=False,protonet=False,num_parts=None):

        super().__init__(nbFeat, nbClass)
        self.dropout = nn.Dropout(p=dropout)

        self.linLay = nn.Linear(self.nbFeat, self.nbClass,bias=bias and not protonet)

        if protonet:
            self.linLay.requires_grad = False
            self.linLay.weight.data[:,:] = 0.5
            for classInd in range(nbFeat//num_parts):
                self.linLay.weight.data[classInd,classInd*num_parts:(classInd+1)*num_parts] = 1

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.bil_cluster_ensemble = bil_cluster_ensemble

        self.aux_on_masked = aux_on_masked
        if self.aux_on_masked:
            self.lin01 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin12 = nn.Linear(int(nbFeat*2/3),nbClass)
            self.lin0 = nn.Linear(nbFeat//3,nbClass)
            self.lin1 = nn.Linear(nbFeat//3,nbClass)
            self.lin2 = nn.Linear(nbFeat//3,nbClass)

    def forward(self, visResDict):

        if not self.bil_cluster_ensemble:
            x = visResDict["x"]

            if len(x.size()) == 4:
                x = self.avgpool(x).squeeze(-1).squeeze(-1)

            x = self.dropout(x)

            pred = self.linLay(x)

            retDict = {"pred": pred}

            if self.aux_on_masked:
                retDict["pred_01"] = self.lin01(x[:,:int(self.nbFeat*2/3)].detach())
                retDict["pred_12"] = self.lin12(x[:,int(self.nbFeat*1/3):].detach())
                retDict["pred_0"] = self.lin0(x[:,:int(self.nbFeat*1/3)].detach())
                retDict["pred_1"] = self.lin1(x[:,int(self.nbFeat*1/3):int(self.nbFeat*2/3)].detach())
                retDict["pred_2"] = self.lin2(x[:,int(self.nbFeat*2/3):].detach())

        else:
            predList = []

            for featVec in visResDict["x"]:
                predList.append(self.linLay(featVec).unsqueeze(0))

            x = torch.cat(predList,dim=0).mean(dim=0)

            retDict = {"pred": x}
            retDict.update({"predBilClusEns{}".format(i):predList[i][0] for i in range(len(predList))})

        return retDict


class Identity(SecondModel):

    def __init__(self,nbFeat,nbClass):
        super().__init__(nbFeat, nbClass)

    def forward(self, x):
        return x

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
    elif backbone_name == "inception":
        nbFeat = 2048
    elif backbone_name.find("efficientnet") != -1:
        nbFeat = 1792
    else:
        raise ValueError("Unkown backbone : {}".format(backbone_name))
    return nbFeat

def netBuilder(args,gpu=None):
    ############### Visual Model #######################

    nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)

    if args.resnet_bilinear:
        CNNconst = CNN2D_bilinearAttPool
        kwargs = {"inFeat":nbFeat,"nb_parts":args.resnet_bil_nb_parts,\
                    "cluster":args.bil_cluster,"cluster_ensemble":args.bil_cluster_ensemble,\
                    "applySoftmaxOnSim":args.apply_softmax_on_sim,\
                    "softmCoeff":args.softm_coeff,\
                    "no_refine":args.bil_cluster_norefine,\
                    "rand_vec":args.bil_cluster_randvec,\
                    "update_sco_by_norm_sim":args.bil_clust_update_sco_by_norm_sim,\
                    "vect_gate":args.bil_clus_vect_gate,\
                    "vect_ind_to_use":args.bil_clus_vect_ind_to_use,\
                    "cluster_lay_ind":args.bil_cluster_lay_ind}

        if not args.bil_cluster_ensemble:

            if args.bil_cluster_lay_ind != 4:
                if nbFeat == 2048:
                    nbFeat = nbFeat//2**(4-args.bil_cluster_lay_ind)
                elif nbFeat == 512:
                    nbFeat = nbFeat//2**(4-args.bil_cluster_lay_ind)
                else:
                    raise ValueError("Unknown feature nb.")

            if args.bil_clus_vect_ind_to_use == "all":
                nbFeat *= args.resnet_bil_nb_parts
            else:
                nbFeat *= len(args.bil_clus_vect_ind_to_use.split(","))

    elif args.protonet:
        CNNconst = CNN2D_protoNet
        kwargs = {"inFeat":nbFeat,"nb_parts":args.resnet_bil_nb_parts,"protoPerClass":args.proto_nb,"classNb":args.class_nb}
    elif args.inter_by_parts:
        CNNconst = CNN2D_interbyparts
        kwargs = {"classNb":args.class_nb,"nb_parts":args.resnet_bil_nb_parts}
    elif args.prototree:
        CNNconst = CNN2D_prototree
        kwargs = {"classNb":args.class_nb}
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

    if args.second_mod == "linear":
        if args.inter_by_parts or args.prototree or args.protonet:
            secondModel = Identity(nbFeat,args.class_nb)
        else:
            #if args.protonet:
            #    nbFeat = args.class_nb*args.proto_nb

            secondModel = LinearSecondModel(nbFeat, args.class_nb, args.dropout,bil_cluster_ensemble=args.bil_cluster_ensemble,\
                                            bias=args.lin_lay_bias,aux_on_masked=args.aux_on_masked,protonet=args.protonet,num_parts=args.resnet_bil_nb_parts)

    else:
        raise ValueError("Unknown second model type : ", args.second_mod)

    ############### Whole Model ##########################

    net = Model(firstModel, secondModel)

    if args.distributed:
        torch.cuda.set_device(gpu)
        net.cuda(gpu)
        net = torch.nn.parallel.DistributedDataParallel(net,device_ids=[gpu],find_unused_parameters=True)
    else:
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

    argreader.parser.add_argument('--inter_by_parts', type=args.str2bool, metavar='BOOL',
                                  help="To train the model from https://github.com/zxhuang1698/interpretability-by-parts/tree/650f1af573075a41f04f2f715f2b2d4bc0363d31")
    argreader.parser.add_argument('--prototree', type=args.str2bool, metavar='BOOL',
                                  help="To train the model from https://github.com/M-Nauta/ProtoTree/blob/86b9bfb38a009576c8e073100b92dd2f639c01e3")

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


    return argreader
