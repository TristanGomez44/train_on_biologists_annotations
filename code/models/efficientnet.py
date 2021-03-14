from efficientnet_pytorch import EfficientNet
import torch

class EffNetB4(torch.nn.Module):

    def __init__(self):
        super().__init__()

        self.model = EfficientNet.from_pretrained('efficientnet-b4')

    def forward(self,x):
        feat = self.model.extract_features(x)
        return {"x":feat}


def efficientnet_b4():
    model = EffNetB4()
    return model

if __name__ == "__main__":
    model = efficientnet_b4()

    inp = torch.zeros(1,3,224,224)

    out = model(inp)
    print(out["x"].size())
