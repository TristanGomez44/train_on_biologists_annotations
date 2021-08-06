import os
import argparse
import pickle
import numpy as np

import torch
import torch.nn as nn

#This comes from https://github.com/M-Nauta/ProtoTree/blob/86b9bfb38a009576c8e073100b92dd2f639c01e3/prototree/prototree.py

import argparse

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import os
import copy
import sys

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

model_dir = './pretrained_models'

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    # class attribute
    expansion = 1
    num_layers = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # only conv with possibly not 1 stride
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # the residual connection
        out += identity
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [3, 3]
        block_strides = [self.stride, 1]
        block_paddings = [1, 1]

        return block_kernel_sizes, block_strides, block_paddings


class Bottleneck(nn.Module):
    # class attribute
    expansion = 4
    num_layers = 3

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        # only conv with possibly not 1 stride
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        # if stride is not 1 then self.downsample cannot be None
        self.downsample = downsample
        self.stride = stride

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
        out = self.relu(out)

        return out

    def block_conv_info(self):
        block_kernel_sizes = [1, 3, 1]
        block_strides = [1, self.stride, 1]
        block_paddings = [0, 1, 0]

        return block_kernel_sizes, block_strides, block_paddings


class ResNet_features(nn.Module):
    '''
    the convolutional layers of ResNet
    the average pooling and final fully convolutional layer is removed
    '''

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet_features, self).__init__()

        self.inplanes = 64

        # the first convolutional layer before the structured sequence of blocks
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # comes from the first conv and the following max pool
        self.kernel_sizes = [7, 3]
        self.strides = [2, 2]
        self.paddings = [3, 1]

        # the following layers, each layer is a sequence of blocks
        self.block = block
        self.layers = layers
        self.layer1 = self._make_layer(block=block, planes=64, num_blocks=self.layers[0])
        self.layer2 = self._make_layer(block=block, planes=128, num_blocks=self.layers[1], stride=2)
        self.layer3 = self._make_layer(block=block, planes=256, num_blocks=self.layers[2], stride=2)
        self.layer4 = self._make_layer(block=block, planes=512, num_blocks=self.layers[3], stride=2)

        # initialize the parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
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

    def _make_layer(self, block, planes, num_blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        # only the first block has downsample that is possibly not None
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes))

        # keep track of every block's conv size, stride size, and padding size
        for each_block in layers:
            block_kernel_sizes, block_strides, block_paddings = each_block.block_conv_info()
            self.kernel_sizes.extend(block_kernel_sizes)
            self.strides.extend(block_strides)
            self.paddings.extend(block_paddings)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings

    def num_layers(self):
        '''
        the number of conv layers in the network, not counting the number
        of bypass layers
        '''

        return (self.block.num_layers * self.layers[0]
              + self.block.num_layers * self.layers[1]
              + self.block.num_layers * self.layers[2]
              + self.block.num_layers * self.layers[3]
              + 1)


    def __repr__(self):
        template = 'resnet{}_features'
        return template.format(self.num_layers() + 1)

def resnet18_features(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet18'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet34_features(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet34'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

def resnet50_features(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet50'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model

def resnet50_features_inat(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on Inaturalist2017
    """
    model = ResNet_features(Bottleneck, [3, 4, 6, 3], **kwargs)
    
    if pretrained:
        #use BBN pretrained weights of the conventional learning branch (from BBN.iNaturalist2017.res50.180epoch.best_model.pth)
        #https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhou_BBN_Bilateral-Branch_Network_With_Cumulative_Learning_for_Long-Tailed_Visual_Recognition_CVPR_2020_paper.pdf
        model_dict = torch.load(os.path.join(os.path.join('features', 'state_dicts'), 'BBN.iNaturalist2017.res50.180epoch.best_model.pth'))
        # rename last residual block from cb_block to layer4.2
        new_model = copy.deepcopy(model_dict)
        for k in model_dict.keys():
            if k.startswith('module.backbone.cb_block'):
                splitted = k.split('cb_block')
                new_model['layer4.2'+splitted[-1]]=model_dict[k]
                del new_model[k]
            elif k.startswith('module.backbone.rb_block'):
                del new_model[k]
            elif k.startswith('module.backbone.'):
                splitted = k.split('backbone.')
                new_model[splitted[-1]]=model_dict[k]
                del new_model[k]
            elif k.startswith('module.classifier'):
                del new_model[k]
        # print(new_model.keys())
        model.load_state_dict(new_model, strict=True)
    return model


def resnet101_features(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet101'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


def resnet152_features(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet_features(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        my_dict = model_zoo.load_url(model_urls['resnet152'], model_dir=model_dir)
        my_dict.pop('fc.weight')
        my_dict.pop('fc.bias')
        model.load_state_dict(my_dict, strict=False)
    return model


#########" PROTOTREE ################"

def min_pool2d(xs, **kwargs):
    return -F.max_pool2d(-xs, **kwargs)

class L2Conv2D(nn.Module):

    """
    Convolutional layer that computes the squared L2 distance instead of the conventional inner product. 
    """

    def __init__(self, num_prototypes, num_features, w_1, h_1):
        """
        Create a new L2Conv2D layer
        :param num_prototypes: The number of prototypes in the layer
        :param num_features: The number of channels in the input features
        :param w_1: Width of the prototypes
        :param h_1: Height of the prototypes
        """
        super().__init__()
        # Each prototype is a latent representation of shape (num_features, w_1, h_1)
        prototype_shape = (num_prototypes, num_features, w_1, h_1)

        self.prototype_vectors = torch.relu(torch.randn(prototype_shape))
        self.prototype_vectors = nn.Parameter(self.prototype_vectors, requires_grad=True)

    def forward(self, xs):
        """
        Perform convolution over the input using the squared L2 distance for all prototypes in the layer
        :param xs: A batch of input images obtained as output from some convolutional neural network F. Following the
                   notation from the paper, let the shape of xs be (batch_size, D, W, H), where
                     - D is the number of output channels of the conv net F
                     - W is the width of the convolutional output of F
                     - H is the height of the convolutional output of F
        :return: a tensor of shape (batch_size, num_prototypes, W, H) obtained from computing the squared L2 distances
                 for patches of the input using all prototypes
        """
        # Adapted from ProtoPNet
        # Computing ||xs - ps ||^2 is equivalent to ||xs||^2 + ||ps||^2 - 2 * xs * ps
        # where ps is some prototype image

        # So first we compute ||xs||^2  (for all patches in the input image that is. We can do this by using convolution
        # with weights set to 1 so each patch just has its values summed)
        ones = torch.ones_like(self.prototype_vectors,
                               device=xs.device)  # Shape: (num_prototypes, num_features, w_1, h_1)
        xs_squared_l2 = F.conv2d(xs ** 2, weight=ones)  # Shape: (bs, num_prototypes, w_in, h_in)

        # Now compute ||ps||^2
        # We can just use a sum here since ||ps||^2 is the same for each patch in the input image when computing the
        # squared L2 distance
        ps_squared_l2 = torch.sum(self.prototype_vectors ** 2,
                                  dim=(1, 2, 3))  # Shape: (num_prototypes,)
        # Reshape the tensor so the dimensions match when computing ||xs||^2 + ||ps||^2
        ps_squared_l2 = ps_squared_l2.view(-1, 1, 1)

        # Compute xs * ps (for all patches in the input image)
        xs_conv = F.conv2d(xs, weight=self.prototype_vectors)  # Shape: (bs, num_prototypes, w_in, h_in)

        # Use the values to compute the squared L2 distance
        distance = xs_squared_l2 + ps_squared_l2 - 2 * xs_conv
        #print("terms",xs_squared_l2.mean().item(),ps_squared_l2.mean().item(),xs_conv.mean().item())
        cos = xs_conv/(torch.sqrt(xs_squared_l2)*torch.sqrt(ps_squared_l2))
        #print("cos",cos.min(),cos.mean(),cos.max())
       
        distance = torch.sqrt(torch.abs(distance)+1e-14) #L2 distance (not squared). Small epsilon added for numerical stability
        
        #print("dist",distance.mean(dim=-1).mean(dim=-1).abs().mean().item())
        #print(round(xs.mean(dim=-1).mean(dim=-1).abs().mean().item(),3))
        #print(self.prototype_vectors.size())

        if torch.isnan(distance).any():
            raise Exception('Error: NaN values! Using the --log_probabilities flag might fix this issue')
        return distance/48  # Shape: (bs, num_prototypes, w_in, h_in)
        #return -6*cos 

class Node(nn.Module):

    def __init__(self, index: int):
        super().__init__()
        self._index = index

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @property
    def index(self) -> int:
        return self._index

    @property
    def size(self) -> int:
        raise NotImplementedError

    @property
    def nodes(self) -> set:
        return self.branches.union(self.leaves)

    @property
    def leaves(self) -> set:
        raise NotImplementedError

    @property
    def branches(self) -> set:
        raise NotImplementedError

    @property
    def nodes_by_index(self) -> dict:
        raise NotImplementedError

    @property
    def num_branches(self) -> int:
        return len(self.branches)

    @property
    def num_leaves(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        raise NotImplementedError

class Leaf(Node):

    def __init__(self,
                 index: int,
                 num_classes: int,
                 args: argparse.Namespace
                 ):
        super().__init__(index)

        # Initialize the distribution parameters
        if args.disable_derivative_free_leaf_optim:
            self._dist_params = nn.Parameter(torch.randn(num_classes), requires_grad=True)

            #torch.nn. init.uniform_(self._dist_params, -4, 4)

        elif args.kontschieder_normalization:
            self._dist_params = nn.Parameter(torch.ones(num_classes), requires_grad=False)
        else:
            self._dist_params = nn.Parameter(torch.zeros(num_classes), requires_grad=False)

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

        self._kontschieder_normalization = args.kontschieder_normalization

    def forward(self, xs: torch.Tensor, **kwargs):

        # Get the batch size
        batch_size = xs.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        if not self._log_probabilities:
            node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            node_attr.setdefault((self, 'pa'), torch.zeros(batch_size, device=xs.device))

        # Obtain the leaf distribution
        dist = self.distribution()  # shape: (k,)
        # Reshape the distribution to a matrix with one single row
        dist = dist.view(1, -1)  # shape: (1, k)
        # Duplicate the row for all x in xs
        dists = torch.cat((dist,) * batch_size, dim=0)  # shape: (bs, k)

        # Store leaf distributions as node property
        node_attr[self, 'ds'] = dists

        #print("leaf",self.index,xs.mean().item(),dists.mean().item())

        # Return both the result of the forward pass as well as the node properties
        return dists, node_attr

    def distribution(self) -> torch.Tensor:
        if not self._kontschieder_normalization:
            if self._log_probabilities:
                return F.log_softmax(self._dist_params, dim=0)
            else:
                # Return numerically stable softmax (see http://www.deeplearningbook.org/contents/numerical.html)
                #return F.softmax(self._dist_params - torch.max(self._dist_params), dim=0)
                return self._dist_params
        
        else:
            #kontschieder_normalization's version that uses a normalization factor instead of softmax:
            if self._log_probabilities:
                return torch.log((self._dist_params / torch.sum(self._dist_params))+1e-10) #add small epsilon for numerical stability
            else:
                return (self._dist_params / torch.sum(self._dist_params))
        
    @property
    def requires_grad(self) -> bool:
        return self._dist_params.requires_grad

    @requires_grad.setter
    def requires_grad(self, val: bool):
        self._dist_params.requires_grad = val

    @property
    def size(self) -> int:
        return 1

    @property
    def leaves(self) -> set:
        return {self}

    @property
    def branches(self) -> set:
        return set()

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self}

    @property
    def num_branches(self) -> int:
        return 0

    @property
    def num_leaves(self) -> int:
        return 1

    @property
    def depth(self) -> int:
        return 0

class Branch(Node):

    def __init__(self,
                 index: int,
                 l: Node,
                 r: Node,
                 args: argparse.Namespace
                 ):
        super().__init__(index)
        self.l = l
        self.r = r

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

    def forward(self, xs: torch.Tensor, **kwargs):

        # Get the batch size
        batch_size = xs.size(0)

        # Keep a dict to assign attributes to nodes. Create one if not already existent
        node_attr = kwargs.setdefault('attr', dict())
        # In this dict, store the probability of arriving at this node.
        # It is assumed that when a parent node calls forward on this node it passes its node_attr object with the call
        # and that it sets the path probability of arriving at its child
        # Therefore, if this attribute is not present this node is assumed to not have a parent.
        # The probability of arriving at this node should thus be set to 1 (as this would be the root in this case)
        # The path probability is tracked for all x in the batch
        if not self._log_probabilities:
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))
        else:
            pa = node_attr.setdefault((self, 'pa'), torch.ones(batch_size, device=xs.device))

        # Obtain the probabilities of taking the right subtree
        ps = self.g(xs, **kwargs)  # shape: (bs,)

        #print("PS",self.index,ps.size(),ps.mean().item())
        if not self._log_probabilities:
            # Store decision node probabilities as node attribute
            node_attr[self, 'ps'] = ps
            # Store path probabilities of arriving at child nodes as node attributes
            node_attr[self.l, 'pa'] = (1 - ps) * pa
            node_attr[self.r, 'pa'] = ps * pa
            # # Store alpha value for this batch for this decision node
            # node_attr[self, 'alpha'] = torch.sum(pa * ps) / torch.sum(pa)

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.l.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.r.forward(xs, **kwargs)  # shape: (bs, k)
            #print("branch",ps.mean().item(),self.index,l_dists.size(),l_dists.mean().item(),r_dists.mean().item())
            # Weight the probability distributions by the decision node's output
            ps = ps.view(batch_size, 1)

            #print(self.index,node_attr[self.r, 'pa'].mean().item(),node_attr[self.l, 'pa'].mean().item())

            return (1 - ps) * l_dists + ps * r_dists, node_attr  # shape: (bs, k)
        else:
            # Store decision node probabilities as node attribute
            node_attr[self, 'ps'] = ps

            # Store path probabilities of arriving at child nodes as node attributes
            # source: rewritten to pytorch from
            # https://github.com/tensorflow/probability/blob/v0.9.0/tensorflow_probability/python/math/generic.py#L447-L471
            x = torch.abs(ps) + 1e-7  # add small epsilon for numerical stability
            oneminusp = torch.where(x < np.log(2), torch.log(-torch.expm1(-x)), torch.log1p(-torch.exp(-x)))

            node_attr[self.l, 'pa'] = oneminusp + pa
            node_attr[self.r, 'pa'] = ps + pa

            # Obtain the unweighted probability distributions from the child nodes
            l_dists, _ = self.l.forward(xs, **kwargs)  # shape: (bs, k)
            r_dists, _ = self.r.forward(xs, **kwargs)  # shape: (bs, k)

            # Weight the probability distributions by the decision node's output
            ps = ps.view(batch_size, 1)
            oneminusp = oneminusp.view(batch_size, 1)
            logs_stacked = torch.stack((oneminusp + l_dists, ps + r_dists))
            return torch.logsumexp(logs_stacked, dim=0), node_attr  # shape: (bs,)

    def g(self, xs: torch.Tensor, **kwargs):
        out_map = kwargs['out_map']  # Obtain the mapping from decision nodes to conv net outputs
        conv_net_output = kwargs['conv_net_output']  # Obtain the conv net outputs
        #try:  
        out = conv_net_output[out_map[self]]  # Obtain the output corresponding to this decision node
        #except:
        #    print(out_map[self].keys)
        #    print(conv_net_output.keys())
        #    sys.exit(0)
        
        return out.squeeze(dim=1)

    @property
    def size(self) -> int:
        return 1 + self.l.size + self.r.size

    @property
    def leaves(self) -> set:
        return self.l.leaves.union(self.r.leaves)

    @property
    def branches(self) -> set:
        return {self} \
            .union(self.l.branches) \
            .union(self.r.branches)

    @property
    def nodes_by_index(self) -> dict:
        return {self.index: self,
                **self.l.nodes_by_index,
                **self.r.nodes_by_index}

    @property
    def num_branches(self) -> int:
        return 1 + self.l.num_branches + self.r.num_branches

    @property
    def num_leaves(self) -> int:
        return self.l.num_leaves + self.r.num_leaves

    @property
    def depth(self) -> int:
        return self.l.depth + 1

class ProtoTree(nn.Module):

    ARGUMENTS = ['depth', 'num_features', 'W1', 'H1', 'log_probabilities']

    SAMPLING_STRATEGIES = ['distributed', 'sample_max', 'greedy']

    def __init__(self,
                 num_classes: int,
                 feature_net: torch.nn.Module,
                 add_on_layers: nn.Module = nn.Identity(),
                 ):
        super().__init__()

        args = argparse.Namespace

        args.depth = 9
        args.W1 = 1
        args.H1 = 1
        args.num_features = 2048 
        args.upsample_threshold = 0.98 
        args.pruning_threshold_leaves = 0.01 
        args.nr_trees_ensemble = 5
        args.disable_derivative_free_leaf_optim = True
        args.kontschieder_normalization=False
        args.log_probabilities = False
        args.kontschieder_train = False
    
        assert args.depth > 0
        assert num_classes > 0

        self._num_classes = num_classes

        # Build the tree
        self._root = self._init_tree(num_classes, args)

        self.num_features = args.num_features
        self.num_prototypes = self.num_branches
        self.prototype_shape = (args.W1, args.H1, args.num_features)
        
        # Keep a dict that stores a reference to each node's parent
        # Key: node -> Value: the node's parent
        # The root of the tree is mapped to None
        self._parents = dict()
        self._set_parents()  # Traverse the tree to build the self._parents dict

        # Set the feature network
        self._net = feature_net
        self._add_on = add_on_layers

        # Flag that indicates whether probabilities or log probabilities are computed
        self._log_probabilities = args.log_probabilities

        # Flag that indicates whether a normalization factor should be used instead of softmax. 
        self._kontschieder_normalization = args.kontschieder_normalization
        self._kontschieder_train = args.kontschieder_train
        # Map each decision node to an output of the feature net
        self._out_map = {n: i for i, n in zip(range(2 ** (args.depth) - 1), self.branches)}

        self.prototype_layer = L2Conv2D(self.num_prototypes,
                                        self.num_features,
                                        args.W1,
                                        args.H1)

    @property
    def root(self) -> Node:
        return self._root

    @property
    def leaves_require_grad(self) -> bool:
        return any([leaf.requires_grad for leaf in self.leaves])

    @leaves_require_grad.setter
    def leaves_require_grad(self, val: bool):
        for leaf in self.leaves:
            leaf.requires_grad = val

    @property
    def prototypes_require_grad(self) -> bool:
        return self.prototype_layer.prototype_vectors.requires_grad

    @prototypes_require_grad.setter
    def prototypes_require_grad(self, val: bool):
        self.prototype_layer.prototype_vectors.requires_grad = val

    @property
    def features_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._net.parameters()])

    @features_require_grad.setter
    def features_require_grad(self, val: bool):
        for param in self._net.parameters():
            param.requires_grad = val

    @property
    def add_on_layers_require_grad(self) -> bool:
        return any([param.requires_grad for param in self._add_on.parameters()])

    @add_on_layers_require_grad.setter
    def add_on_layers_require_grad(self, val: bool):
        for param in self._add_on.parameters():
            param.requires_grad = val

    def forward(self,
                xs: torch.Tensor,
                sampling_strategy: str = SAMPLING_STRATEGIES[0],  # `distributed` by default
                **kwargs,
                ) -> tuple:
        assert sampling_strategy in ProtoTree.SAMPLING_STRATEGIES

        '''
            PERFORM A FORWARD PASS THROUGH THE FEATURE NET
        '''

        # Perform a forward pass with the conv net
        features = self._net(xs)
        features = self._add_on(features)
        bs, D, W, H = features.shape

        '''
            COMPUTE THE PROTOTYPE SIMILARITIES GIVEN THE COMPUTED FEATURES
        '''

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)  # Shape: (batch_size, num_prototypes, W, H)

        attMaps = distances

        #print("dists",distances.mean().item())

        # Perform global min pooling to see the minimal distance for each prototype to any patch of the input image
        min_distances = min_pool2d(distances, kernel_size=(W, H))
        min_distances = min_distances.view(bs, self.num_prototypes)

        #print(min_distances.mean().item())

        if not self._log_probabilities:
            similarities = torch.exp(-min_distances)
            #similarities = -min_distances
        else:
            # Omit the exp since we require log probabilities
            similarities = -min_distances

        #print("sim",similarities.min().item(),similarities.mean().item(),similarities.max().item())

        # Add the conv net output to the kwargs dict to be passed to the decision nodes in the tree
        # Split (or chunk) the conv net output tensor of shape (batch_size, num_decision_nodes) into individual tensors
        # of shape (batch_size, 1) containing the logits that are relevant to single decision nodes
        kwargs['conv_net_output'] = similarities.chunk(similarities.size(1), dim=1)
        # Add the mapping of decision nodes to conv net outputs to the kwargs dict to be passed to the decision nodes in
        # the tree
        kwargs['out_map'] = dict(self._out_map)  # Use a copy of self._out_map, as the original should not be modified

        '''
            PERFORM A FORWARD PASS THROUGH THE TREE GIVEN THE COMPUTED SIMILARITIES
        '''

        # Perform a forward pass through the tree
        out, attr = self._root.forward(xs, **kwargs)

        #print("xs",out.mean().item(),xs.mean().item())

        info = dict()
        # Store the probability of arriving at all nodes in the decision tree
        info['pa_tensor'] = {n.index: attr[n, 'pa'].unsqueeze(1) for n in self.nodes}
        # Store the output probabilities of all decision nodes in the tree
        info['ps'] = {n.index: attr[n, 'ps'].unsqueeze(1) for n in self.branches}

        #print("out",out.mean().item())

        # Generate the output based on the chosen sampling strategy
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[0]:  # Distributed
            return out, info, attMaps,features
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[1]:  # Sample max
            # Get the batch size
            batch_size = xs.size(0)
            # Get an ordering of all leaves in the tree
            leaves = list(self.leaves)
            # Obtain path probabilities of arriving at each leaf
            pas = [attr[l, 'pa'].view(batch_size, 1) for l in leaves]  # All shaped (bs, 1)
            # Obtain output distributions of each leaf
            dss = [attr[l, 'ds'].view(batch_size, 1, self._num_classes) for l in leaves]  # All shaped (bs, 1, k)
            # Prepare data for selection of most probable distributions
            # Let L denote the number of leaves in this tree
            pas = torch.cat(tuple(pas), dim=1)  # shape: (bs, L)
            dss = torch.cat(tuple(dss), dim=1)  # shape: (bs, L, k)
            # Select indices (in the 'leaves' variable) of leaves with highest path probability
            ix = torch.argmax(pas, dim=1).long()  # shape: (bs,)
            # Select distributions of leafs with highest path probability
            dists = []
            for j, i in zip(range(dss.shape[0]), ix):
                dists += [dss[j][i].view(1, -1)]  # All shaped (1, k)
            dists = torch.cat(tuple(dists), dim=0)  # shape: (bs, k)

            # Store the indices of the leaves with the highest path probability
            info['out_leaf_ix'] = [leaves[i.item()].index for i in ix]

            return dists, info, attMaps,features
        if sampling_strategy == ProtoTree.SAMPLING_STRATEGIES[2]:  # Greedy
            # At every decision node, the child with highest probability will be chosen
            batch_size = xs.size(0)
            # Set the threshold for when either child is more likely
            threshold = 0.5 if not self._log_probabilities else np.log(0.5)
            # Keep track of the routes taken for each of the items in the batch
            routing = [[] for _ in range(batch_size)]
            # Traverse the tree for all items
            # Keep track of all nodes encountered
            for i in range(batch_size):
                node = self._root
                while node in self.branches:
                    routing[i] += [node]
                    if attr[node, 'ps'][i].item() > threshold:
                        node = node.r
                    else:
                        node = node.l
                routing[i] += [node]

            # Obtain output distributions of each leaf
            # Each selected leaf is at the end of a path stored in the `routing` variable
            dists = [attr[path[-1], 'ds'][0] for path in routing]
            # Concatenate the dists in a new batch dimension
            dists = torch.cat([dist.unsqueeze(0) for dist in dists], dim=0).to(device=xs.device)

            # Store info
            info['out_leaf_ix'] = [path[-1].index for path in routing]

            return dists, info, attMaps,features
        raise Exception('Sampling strategy not recognized!')

    def forward_partial(self, xs: torch.Tensor) -> tuple:

        # Perform a forward pass with the conv net
        features = self._net(xs)
        features = self._add_on(features)

        # Use the features to compute the distances from the prototypes
        distances = self.prototype_layer(features)  # Shape: (batch_size, num_prototypes, W, H)

        return features, distances, dict(self._out_map)

    @property
    def depth(self) -> int:
        d = lambda node: 1 if isinstance(node, Leaf) else 1 + max(d(node.l), d(node.r))
        return d(self._root)

    @property
    def size(self) -> int:
        return self._root.size

    @property
    def nodes(self) -> set:
        return self._root.nodes

    @property
    def nodes_by_index(self) -> dict:
        return self._root.nodes_by_index

    @property
    def node_depths(self) -> dict:

        def _assign_depths(node, d):
            if isinstance(node, Leaf):
                return {node: d}
            if isinstance(node, Branch):
                return {node: d, **_assign_depths(node.r, d + 1), **_assign_depths(node.l, d + 1)}

        return _assign_depths(self._root, 0)

    @property
    def branches(self) -> set:
        return self._root.branches

    @property
    def leaves(self) -> set:
        return self._root.leaves

    @property
    def num_branches(self) -> int:
        return self._root.num_branches

    @property
    def num_leaves(self) -> int:
        return self._root.num_leaves

    def save(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model.pth', 'wb') as f:
            torch.save(self, f)

    def save_state(self, directory_path: str):
        # Make sure the target directory exists
        if not os.path.isdir(directory_path):
            os.mkdir(directory_path)
        # Save the model to the target directory
        with open(directory_path + '/model_state.pth', 'wb') as f:
            torch.save(self.state_dict(), f)
        # Save the out_map of the model to the target directory
        with open(directory_path + '/tree.pkl', 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        
    @staticmethod
    def load(directory_path: str):
        return torch.load(directory_path + '/model.pth')      
       
    def _init_tree(self,
                   num_classes,
                   args: argparse.Namespace) -> Node:

        def _init_tree_recursive(i: int, d: int) -> Node:  # Recursively build the tree
            if d == args.depth:
                return Leaf(i,
                            num_classes,
                            args
                            )
            else:
                left = _init_tree_recursive(i + 1, d + 1)
                return Branch(i,
                              left,
                              _init_tree_recursive(i + left.size + 1, d + 1),
                              args,
                              )

        return _init_tree_recursive(0, 0)

    def _set_parents(self) -> None:
        self._parents.clear()
        self._parents[self._root] = None

        def _set_parents_recursively(node: Node):
            if isinstance(node, Branch):
                self._parents[node.r] = node
                self._parents[node.l] = node
                _set_parents_recursively(node.r)
                _set_parents_recursively(node.l)
                return
            if isinstance(node, Leaf):
                return  # Nothing to do here!
            raise Exception('Unrecognized node type!')

        # Set all parents by traversing the tree starting from the root
        _set_parents_recursively(self._root)

    def path_to(self, node: Node):
        assert node in self.leaves or node in self.branches
        path = [node]
        while isinstance(self._parents[node], Node):
            node = self._parents[node]
            path = [node] + path
        return path

def prototree(num_classes):

    backbone = resnet50_features(pretrained=True)

    tree = ProtoTree(num_classes=num_classes,
                feature_net = backbone)
    
    return tree
