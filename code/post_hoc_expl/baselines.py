
import sys
from torch.nn import Module
from torch import rand,arange,randint
from torch.nn.functional import adaptive_avg_pool2d 

class RandomMap(Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,targ):
        downsample_ratio = self.model.firstModel.featMod.downsample_ratio
        shape = (x.shape[0],1,x.shape[2]//downsample_ratio,x.shape[3]//downsample_ratio)
        random_map = rand(size=shape)
        return random_map

class TopFeatMap(Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,targ):
        retDict = self.model(x)
        feats = retDict["feat"]

        feat_pooled = adaptive_avg_pool2d(feats,1).squeeze(-1).squeeze(-1)
        
        top_featmap_inds = feat_pooled.argmax(dim=1)

        inds = top_featmap_inds.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        inds = inds.expand(-1,1,feats.shape[2],feats.shape[3])
        top_feat_maps = feats.gather(1,inds)

        return top_feat_maps

class RandomFeatMap(Module):
    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,targ):
        retDict = self.model(x)
        feats = retDict["feat"]

        rand_featmap_inds = randint(low=0,high=feats.shape[1],size=(x.shape[0],)).to(x.device)
 
        inds = rand_featmap_inds.unsqueeze(1).unsqueeze(1).unsqueeze(1)
        inds = inds.expand(-1,1,feats.shape[2],feats.shape[3])
        random_feat_maps = feats.gather(1,inds)
        
        return random_feat_maps

class AM(Module):

    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,targ):
        retDict = self.model(x)
        feats = retDict["feat"]
        am = feats.sum(dim=1,keepdim=True)
        return am   

class CAM(Module):

    def __init__(self,model) -> None:
        super().__init__()
        self.model = model

    def forward(self,x,targ):
        retDict = self.model(x)
        feats = retDict["feat"]
        
        weights = self.model.secondModel.linLay.weight.data[targ]
        weights /= weights.sum(dim=1,keepdim=True)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        cam = (feats*weights).sum(dim=1,keepdim=True)
        return cam