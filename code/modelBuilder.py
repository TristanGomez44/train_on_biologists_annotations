import torch
from torch import nn
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from models import resnet,transformer,transformer_dino,transformer_swin
import load_data
import args
EPS = 0.000001

import torch.nn.functional as F

from torch.autograd import Function

from load_data import get_img_size,get_class_nb
import utils 

from enums import Tasks

def buildFeatModel(featModelName, **kwargs):
    ''' Build a visual feature model

    Args:
    - featModelName (str): the name of the architecture. Can be resnet50, resnet101
    Returns:
    - featModel (nn.Module): the visual feature extractor

    '''
    if "resnet"in featModelName:
        featModel = getattr(resnet, featModelName)(**kwargs)
    elif "vit" in featModelName:
        featModel = getattr(transformer, featModelName)(weights="IMAGENET1K_V1",**kwargs)
    elif "dit" in featModelName:
        featModel = getattr(transformer_dino, featModelName)()
    elif "swin" in featModelName:
        featModel = getattr(transformer_swin, featModelName)()    
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
        feat = self.net.firstModel.featMod(x)["feat"]

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

        if "feat" in retDict:
            retDict["feat_pooled"] = self.avgpool(retDict["feat"]).squeeze(-1).squeeze(-1)

        return retDict

def buildImageAttention(inFeat,outChan=1):
    attention = []
    attention.append(resnet.BasicBlock(inFeat, inFeat))
    attention.append(resnet.conv1x1(inFeat, outChan))
    return nn.Sequential(*attention)

class CNN2D_bilinearAttPool(FirstModel):

    def __init__(self, featModelName,inFeat=512,nb_parts=3,cluster=False,no_refine=False,rand_vec=False,**kwargs):

        super().__init__(featModelName,**kwargs)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.nb_parts = nb_parts
        self.cluster = cluster

        self.attention = buildImageAttention(inFeat,nb_parts)
        self.attention_activation = torch.relu

    def forward(self, x):
        # N x C x H x L
        self.batchSize = x.size(0)
        # N x C x H x L
        retDict = self.featMod(x)

        features = retDict["feat"]

        spatialWeights = self.attention_activation(self.attention(features))
        features_weig = (spatialWeights.unsqueeze(2)*features.unsqueeze(1)).reshape(features.size(0),spatialWeights.size(1),features.size(1),features.size(2),features.size(3))
        features_agr = self.avgpool(features_weig).squeeze(-1).squeeze(-1)
        #features_agr = features_agr.view(features.size(0), -1)

        retDict["feat_pooled"] = features_agr
        retDict["attMaps"] = spatialWeights

        return retDict

################################ Temporal Model ########################""

class SecondModel(nn.Module):

    def __init__(self, nbFeat, nbClass):
        super().__init__()
        self.nbFeat, self.nbClass = nbFeat, nbClass

    def forward(self, x):
        raise NotImplementedError

class DINOHead(torch.nn.Module):

    def __init__(self,dimension,norm_last_layer=False,bottleneck_dim=256):
        super().__init__()
        layers = []

        layers.append(nn.Linear(dimension, dimension))
        layers.append(nn.GELU())
        layers.append(nn.Linear(dimension, dimension))
        layers.append(nn.GELU())
        layers.append(nn.Linear(dimension, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, dimension, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x
    
class LinearSecondModel(SecondModel):

    def __init__(self, nbFeat, nb_class_dic, dropout,bias=True,tasks=None,ssl=False,regression=False):

        super().__init__(nbFeat, 1)
        self.dropout = nn.Dropout(p=dropout)
        
        if tasks is None:
            tasks = [task.value for task in Tasks]

        self.tasks = np.array(tasks)
        self.regression = regression

        for task in self.tasks:
            if regression:
                output_dim=1
            else:
                output_dim = nb_class_dic[task]
            layer = nn.Linear(self.nbFeat, output_dim,bias=bias)
            setattr(self,"lin_lay_"+task,layer)
 
        self.ssl = ssl
        if self.ssl:
            self.ssl_head = DINOHead(self.nbFeat)

    def get_feat(self,x,key_ind):
        if len(x.size()) == 3:
            return x[:,key_ind]
        elif len(x.size()) == 2:
            return x
        else:
            raise ValueError("Unkown size of x",x.size(),". x should have 2 or 3 dimensions.")

    def get_output(self,x):
        output_dic = {}
        for i,key in enumerate(self.tasks):
            x_ = self.get_feat(x,i)
            output = getattr(self,"lin_lay_"+key)(x_)
            
            output_dic["output_"+key] = output

        return output_dic
    
    def forward(self, retDict):

        if self.ssl:
            x = retDict["feat_pooled"]
            x = self.ssl_head(x)
            retDict["output"] = x
        else:
            x = retDict["feat_pooled"]
            x = self.dropout(x)
            output_dic = self.get_output(x)
            retDict.update(output_dic)

        return retDict

def get_mlp(nbFeat):
    return nn.Sequential(nn.Linear(nbFeat,nbFeat*2),nn.ReLU(),nn.Linear(nbFeat*2,2))

def getResnetFeat(backbone_name, backbone_inplanes):
    if backbone_name in ["resnet50","resnet101","resnet152"]:
        nbFeat = backbone_inplanes * 4 * 2 ** (4 - 1)
    elif backbone_name.find("resnet34") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name.find("resnet18") != -1:
        nbFeat = backbone_inplanes * 2 ** (4 - 1)
    elif backbone_name == "convnext_small":
        nbFeat = 768
    elif backbone_name == "convnext_base":
        nbFeat = 1024
    elif "vit" in backbone_name or "dit" in backbone_name:
        nbFeat = 768
    elif "swin" in backbone_name:
        nbFeat = 1024        
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

def advNetBuilder(args):
    nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)
    adv_mlp = nn.Sequential(nn.Linear(nbFeat,nbFeat*2),nn.ReLU(),nn.Linear(nbFeat*2,2))

    if args.cuda and torch.cuda.is_available():
        adv_mlp.cuda()
        if args.multi_gpu:
            adv_mlp = DataParallelModel(adv_mlp)

    return adv_mlp


def netBuilder(args):

    nbFeat = getResnetFeat(args.first_mod, args.resnet_chan)

    if args.resnet_bilinear:
        CNNconst = CNN2D_bilinearAttPool
        att_map_nb = len(Tasks) if args.task_to_train=="all" else 1

        kwargs = {"inFeat":nbFeat,"nb_parts":att_map_nb,"strideLay2":args.stride_lay2,"strideLay3":args.stride_lay3,"strideLay4":args.stride_lay4}
    else:
        CNNconst = CNN2D
        if "vit" in args.first_mod:
            kwargs = {"image_size":get_img_size(args)}
        else:
            kwargs = {"chan":args.resnet_chan, "stride":args.resnet_stride,\
                        "strideLay2":args.stride_lay2,"strideLay3":args.stride_lay3,
                        "strideLay4":args.stride_lay4} 
        
    firstModel = CNNconst(args.first_mod,**kwargs)

    if args.second_mod == "linear":
        nb_class_dic = utils.make_class_nb_dic(args)
        tasks = [task.value for task in Tasks]
        secondModel = LinearSecondModel(nbFeat, nb_class_dic, args.dropout,args.lin_lay_bias,tasks,args.ssl,args.regression)
    else:
        raise ValueError("Unknown second model type : ", args.second_mod)

    ############## Whole Model ##########################

    net = Model(firstModel, secondModel)

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

    argreader.parser.add_argument('--bil_cluster_norefine', type=args.str2bool, metavar='BOOL',
                                  help="To not refine feature vectors by using similar vectors.")
    argreader.parser.add_argument('--bil_cluster_randvec', type=args.str2bool, metavar='BOOL',
                                  help="To select random vectors as initial estimation instead of vectors with high norms.")

    argreader.parser.add_argument('--protonet', type=args.str2bool, metavar='BOOL',
                                  help="To train a protonet model")
    argreader.parser.add_argument('--proto_nb', type=int, metavar='BOOL',
                                  help="The nb of prototypes per class.")
    argreader.parser.add_argument('--protonet_warm', type=int, metavar='BOOL',
                                  help="Warmup epoch number")
    argreader.parser.add_argument('--prototree', type=args.str2bool, metavar='BOOL',
                                  help="To train a prototree model")

    argreader.parser.add_argument('--lin_lay_bias', type=args.str2bool, metavar='BOOL',
                                  help="To add a bias to the final layer.")

    argreader.parser.add_argument('--aux_on_masked', type=args.str2bool, metavar='BOOL',
                                  help="To train dense layers on masked version of the feature matrix.")

    argreader.parser.add_argument('--master_net', type=args.str2bool, help='To distill a master network into the trained network.')
    argreader.parser.add_argument('--m_model_id', type=str, help='The model id of the master network')
    argreader.parser.add_argument('--kl_interp', type=float, help='If set to 0, will use regular target, if set to 1, will only use master net target')
    argreader.parser.add_argument('--kl_temp', type=float, help='KL temperature.')

    argreader.parser.add_argument('--end_relu', type=args.str2bool, help='To add a relu at the end of the first block of each layer.')
    argreader.parser.add_argument('--nce_proj_layer', type=args.str2bool, help='To add a projection layer when running NCE training.')

    argreader.parser.add_argument('--temperature', type=float, help='Temperature applied to output')
    
    argreader.parser.add_argument('--one_feat_per_head', type=args.str2bool, metavar='M',
                                  help='To compute one feature per prediction head. Is useful for example-based explanations.')       
         
    argreader.parser.add_argument('--init_range_for_reg_to_class_centroid', type=float, metavar='M',
                                  help='The range to use to init the reg to class centroids.')       

    return argreader
