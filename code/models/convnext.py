#import torchvision
import torch.nn
from torchvision.models.convnext import ConvNeXt as _ConvNeXt,CNBlockConfig
#from torchvision.models import ConvNeXt_Base_Weights,ConvNeXt_Small_Weights
#from torchvision.models._utils import _ovewrite_named_param
#from torchvision.transforms._presets import ImageClassification
#from torchvision.models._api import Weights, WeightsEnum
#from torchvision.models._meta import _IMAGENET_CATEGORIES

from torchvision._internally_replaced_utils import load_state_dict_from_url



from functools import partial
'''
class ConvNext(torch.nn.Module):
    def __init__(self,model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.convnext = getattr(torchvision.models,model_name)(weights="IMAGENET1K_V1")
        self.convnext.avgpool = torch.nn.Identity()
   
        self.layer_norm = self.convnext.classifier[0]
        self.convnext.classifier = torch.nn.Identity()

    def forward(self,x):
        feat = self.convnext(x)
        feat = self.layer_norm(feat)
        return {"feat":feat}

def convnext_small():
    return ConvNext("convnext_small")

def convnext_base():
    return ConvNext("convnext_base")
'''

class ConvNeXt(_ConvNeXt):
    def _forward_impl(self, x):
        retDict = {}
        x = self.features(x)
        retDict["feat"] = x
        x = self.avgpool(x)
        retDict["feat_pooled"] = x
        x = self.classifier(x)
        retDict["output"] = x
        return retDict

def _convnext(
    arch: str,
    block_setting,
    stochastic_depth_prob,
    pretrained,
    progress,
    **kwargs):
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, **kwargs)
    if pretrained:
        if arch not in _MODELS_URLS:
            raise ValueError(f"No checkpoint is available for model type {arch}")
        state_dict = load_state_dict_from_url(_MODELS_URLS[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model



def convnext_small(*, pretrained = False, progress = True, **kwargs):
    r"""ConvNeXt Small model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 27),
        CNBlockConfig(768, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.4)
    return _convnext("convnext_small", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)


def convnext_base(*, pretrained = False, progress = True, **kwargs):
    r"""ConvNeXt Base model architecture from the
    `"A ConvNet for the 2020s" <https://arxiv.org/abs/2201.03545>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block_setting = [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]
    stochastic_depth_prob = kwargs.pop("stochastic_depth_prob", 0.5)
    return _convnext("convnext_base", block_setting, stochastic_depth_prob, pretrained, progress, **kwargs)



_MODELS_URLS = {
    "convnext_tiny": "https://download.pytorch.org/models/convnext_tiny-983f1562.pth",
    "convnext_small": "https://download.pytorch.org/models/convnext_small-0c510722.pth",
    "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    "convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
}





if __name__ == "__main__":
    dummy_data = torch.ones(1,3,224,224)
    print(convnext_small(num_classes=7)(dummy_data)["feat"].shape)
    print(convnext_base(num_classes=7)(dummy_data)["feat"].shape)
