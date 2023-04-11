
import sys
from torch.nn import Module

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