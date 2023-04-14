from pytorch_grad_cam import AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import from_numpy
from torch.nn.functional import interpolate

class AblationCAM2(AblationCAM):
    def __call__(self,input_tensor,targets):
        targets = [ClassifierOutputTarget(target) for target in targets]
        saliency_maps = from_numpy(super().__call__(input_tensor,targets))
        print(saliency_maps.shape)
        return saliency_maps

class AblationCAM2_NoUpScale(AblationCAM):
    def __call__(self,input_tensor,targets):
        targets = [ClassifierOutputTarget(target) for target in targets]
        saliency_maps = from_numpy(super().__call__(input_tensor,targets))
        saliency_maps = interpolate(saliency_maps.unsqueeze(0),scale_factor=1/self.model.net.firstModel.featMod.downsample_ratio)[0]
        print(saliency_maps.shape)
        return saliency_maps