import torch.nn as nn
import torch.utils.model_zoo as model_zoo

import numpy as np
from  torch.nn.modules.upsampling import Upsample

from torch.nn.functional import interpolate as interpo
import sys
import glob
import torch

from skimage.transform import resize
import torch.nn.functional as F
'''

Just a modification of the torchvision resnet model to get the before-to-last activation


'''

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1,dilation=1,groups=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False,dilation=dilation,groups=groups)


def conv1x1(in_planes, out_planes, stride=1,groups=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,groups=groups)

def conv3x3Transp(in_planes, out_planes, stride=1,dilation=1,groups=1):
    """3x3 convolution with padding"""
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride,padding=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,endRelu=True,dilation=1,groups=1,applyStrideOnAll=False,\
                replaceBy1x1=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride,dilation,groups=groups)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        if replaceBy1x1:
            self.conv2 = conv1x1(planes, planes,groups=groups,stride=stride if applyStrideOnAll else 1)
        else:
            self.conv2 = conv3x3(planes, planes,groups=groups,stride=stride if applyStrideOnAll else 1)

        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        self.endRelu = endRelu

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity

        if self.endRelu:
            out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm_layer=None,endRelu=False,dilation=1,applyStrideOnAll=True,replaceBy1x1=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = norm_layer(planes)
        self.conv2 = conv3x3(planes, planes, stride,dilation)
        self.bn2 = norm_layer(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.endRelu = endRelu

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.endRelu:
            out = self.relu(out)

        return out

class TanHPlusRelu(nn.Module):

    def __init__(self):
        super(TanHPlusRelu,self).__init__()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
    def forward(self,x):
        return self.relu(self.tanh(x))

class MultiLevelFeat(nn.Module):

    def __init__(self,chan,outChan,cat):
        super(MultiLevelFeat, self).__init__()

        self.cat = cat
        if cat:
            self.multLev_conv1x1_1 = conv1x1(chan,outChan//4)
            self.multLev_conv1x1_2 = conv1x1(chan*2,outChan//4)
            self.multLev_conv1x1_3 = conv1x1(chan*4,outChan//4)
            self.multLev_conv1x1_4 = conv1x1(chan*8,outChan//4)
        else:
            self.multLev_conv1x1_1 = conv1x1(chan,outChan)
            self.multLev_conv1x1_2 = conv1x1(chan*2,outChan)
            self.multLev_conv1x1_3 = conv1x1(chan*4,outChan)
            self.multLev_conv1x1_4 = conv1x1(chan*8,outChan)

    def norm(self,maps):
        return maps/torch.abs(maps).max(dim=-1,keepdim=True)[0].max(dim=-1,keepdim=True)[0].max(dim=-1,keepdim=True)[0]

    def forward(self,featMaps):

        maps1 = self.multLev_conv1x1_1(featMaps[1])
        maps2 = interpo(self.multLev_conv1x1_2(featMaps[2]),(maps1.size(-2),maps1.size(-1)))
        maps3 = interpo(self.multLev_conv1x1_3(featMaps[3]),(maps1.size(-2),maps1.size(-1)))
        maps4 = interpo(self.multLev_conv1x1_4(featMaps[4]),(maps1.size(-2),maps1.size(-1)))

        if self.cat:
            maps1 = self.norm(maps1)
            maps2 = self.norm(maps2)
            maps3 = self.norm(maps3)
            maps4 = self.norm(maps4)
            maps = torch.cat((maps1,maps2,maps3,maps4),dim=1)
        else:
            maps = (maps1+maps2+maps3+maps4)/4

        return maps

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False, norm_layer=None,maxPoolKer=(3,3),maxPoolPad=(1,1),stride=(2,2),\
                    featMap=False,chan=64,inChan=3,dilation=1,layerSizeReduce=True,preLayerSizeReduce=True,layersNb=4,attention=False,attChan=16,attBlockNb=1,\
                    attActFunc="sigmoid",applyStrideOnAll=False,replaceBy1x1=False,reluOnLast=False,multiLevelFeat=False,multiLev_outChan=64,multiLev_cat=False):

        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.inplanes = chan
        self.conv1 = nn.Conv2d(inChan, chan, kernel_size=7, stride=1 if not preLayerSizeReduce else stride,bias=False,padding=3)
        self.bn1 = norm_layer(chan)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=maxPoolKer, stride=1 if not preLayerSizeReduce else stride, padding=maxPoolPad)

        if type(dilation) is int:
            dilation = [dilation,dilation,dilation]
        elif len(dilation) != 3:
            raise ValueError("dilation must be a list of 3 int or an int.")

        self.nbLayers = len(layers)

        self.replaceBy1x1 = replaceBy1x1
        #All layers are built but they will not necessarily be used
        self.layer1 = self._make_layer(block, chan*1, layers[0], stride=1,norm_layer=norm_layer,reluOnLast=reluOnLast if self.nbLayers==1 else True,\
                                        dilation=1,applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1)
        self.layer2 = self._make_layer(block, chan*2, layers[1], stride=1 if not layerSizeReduce else stride, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==2 else True,dilation=dilation[0],applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1)
        self.layer3 = self._make_layer(block, chan*4, layers[2], stride=1 if not layerSizeReduce else stride, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==3 else True,dilation=dilation[1],applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1)
        self.layer4 = self._make_layer(block, chan*8, layers[3], stride=1 if not layerSizeReduce else stride, norm_layer=norm_layer,\
                                        reluOnLast=reluOnLast if self.nbLayers==4 else True,dilation=dilation[2],applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1)

        if layersNb<1 or layersNb>4:
            raise ValueError("Wrong number of layer : ",layersNb)

        self.layersNb = layersNb

        self.fc = nn.Linear(chan*(2**(4-1)) * block.expansion, 1000)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        self.featMap = featMap

        self.multiLevelFeat = multiLevelFeat
        if self.multiLevelFeat:
            self.multiLevMod = MultiLevelFeat(chan,multiLev_outChan,multiLev_cat)


        self.attention = attention
        if attention:
            self.inplanes = attChan
            self.att1 = self._make_layer(block, attChan, attBlockNb, stride=1, norm_layer=norm_layer,feat=True)
            self.att2 = self._make_layer(block, attChan, attBlockNb, stride=1, norm_layer=norm_layer,feat=True)
            self.att3 = self._make_layer(block, attChan, attBlockNb, stride=1, norm_layer=norm_layer,feat=True)
            self.att4 = self._make_layer(block, attChan, attBlockNb, stride=1, norm_layer=norm_layer,feat=True)
            self.att_conv1x1_1 = conv1x1(chan*1, attChan, stride=1)
            self.att_conv1x1_2 = conv1x1(chan*2, attChan, stride=1)
            self.att_conv1x1_3 = conv1x1(chan*4, attChan, stride=1)
            self.att_conv1x1_4 = conv1x1(chan*8, attChan, stride=1)
            self.att_final_conv1x1 = conv1x1(attChan, 1, stride=1)

            if attActFunc == "sigmoid":
                actFuncConstructor = nn.Sigmoid
            elif attActFunc == "relu":
                actFuncConstructor = nn.ReLU
            elif attActFunc == "tanh+relu":
                actFuncConstructor = TanHPlusRelu

            self.att_1 = nn.Sequential(self.att_conv1x1_1,self.att1,self.att_final_conv1x1,actFuncConstructor())
            self.att_2 = nn.Sequential(self.att_conv1x1_2,self.att2,self.att_final_conv1x1,actFuncConstructor())
            self.att_3 = nn.Sequential(self.att_conv1x1_3,self.att3,self.att_final_conv1x1,actFuncConstructor())
            self.att_4 = nn.Sequential(self.att_conv1x1_4,self.att4,self.att_final_conv1x1,actFuncConstructor())

    def _make_layer(self, block, planes, blocks, stride=1, norm_layer=None,reluOnLast=False,dilation=1,applyStrideOnAll=False,replaceBy1x1=False):

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, [stride[0]*2,stride[1]*2] if (block is BasicBlock) and applyStrideOnAll else stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, norm_layer,applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):

            if i == blocks-1:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,endRelu=reluOnLast,dilation=dilation,applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1))
            else:
                layers.append(block(self.inplanes, planes, norm_layer=norm_layer,endRelu=True,dilation=dilation,applyStrideOnAll=applyStrideOnAll,replaceBy1x1=replaceBy1x1))

        return nn.Sequential(*layers)

    def forward(self,x,returnLayer="last"):

        retDict = {}
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if self.attention:
            attWeightsDict = {}

        layerFeat = {}
        multiLevelFeat = {}
        lastLayer = self.layersNb if returnLayer=="last" else int(returnLayer)

        for i in range(1,lastLayer+1):
            x = getattr(self,"layer{}".format(i))(x)
            if self.attention:
                attWeights = getattr(self,"att_{}".format(i))(x)
                attWeightsDict[i] = attWeights
                x = x*attWeights

            layerFeat[i] = x

        if self.multiLevelFeat:
            x = self.multiLevMod(layerFeat)

        if not self.featMap:
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)

        retDict["layerFeat"] = layerFeat
        retDict["x"] = x
        if self.attention:
            retDict["attMaps"] = attWeightsDict

        if self.multiLevelFeat:
            retDict["multiLevelFeat"] = multiLevelFeat

        return retDict

def removeTopLayer(params):
    params.pop("fc.weight")
    params.pop("fc.bias")
    return params

def resnet4(pretrained=False,chan=8, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=1,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet4 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet9_att(pretrained=False,chan=8,attChan=16,attBlockNb=1, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=2,attention=True,attChan=attChan,attBlockNb=attBlockNb,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet9 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)

        paramsToLoad = {}
        for key in params:
            if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
                paramsToLoad[key] = params[key]
        params = paramsToLoad

        model.load_state_dict(params,strict=False)
    return model


def resnet9(pretrained=False,chan=8, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=2,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet9 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet14_att(pretrained=False,chan=8,attChan=16,attBlockNb=1, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], chan=chan,layersNb=3,attention=True,attChan=attChan,attBlockNb=attBlockNb,**kwargs)

    if pretrained and chan != 64:
        raise ValueError("ResNet14 with {} channel does not have pretrained weights on ImageNet.".format(chan))
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)

        paramsToLoad = {}
        for key in params:
            if key in model.state_dict() and model.state_dict()[key].size() == params[key].size():
                paramsToLoad[key] = params[key]
        params = paramsToLoad

        model.load_state_dict(params,strict=False)
    return model

def resnet18_att(pretrained=False, strict=True,attChan=16,attBlockNb=1,**kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],attention=True,attChan=attChan,attBlockNb=attBlockNb, **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet18'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model

def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet34'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model


def resnet50(pretrained=False, strict=True,**kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet50'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet101'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        params = model_zoo.load_url(model_urls['resnet152'])
        params = removeTopLayer(params)
        model.load_state_dict(params,strict=False)
    return model
