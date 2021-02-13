from efficientnet_pytorch import EfficientNet
import torch

class EffNetB2(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b2')

    def forward(self,x):
        feat = self.model.extract_features(x)
        return {"x":feat}


def efficientnet_b2():
    model = EffNetB2()
    return model

if __name__ == "__main__":
    efficientnet_b2()
