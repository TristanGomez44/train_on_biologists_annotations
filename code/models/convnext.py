import torchvision
import torch.nn

class ConvNext(torch.nn.Module):
    def __init__(self,model_name) -> None:
        super().__init__()
        self.model_name = model_name
        self.convnext = getattr(torchvision.models,model_name)(weights="IMAGENET1K_V1")
        self.convnext.avgpool = torch.nn.Identity()
        self.convnext.classifier = torch.nn.Identity()

    def forward(self,x):
        feat = self.convnext(x)
        return {"feat":feat}

def convnext_small():
    return ConvNext("convnext_small")

def convnext_base():
    return ConvNext("convnext_base")

if __name__ == "__main__":
    dummy_data = torch.ones(1,3,224,224)
    print(convnext_small()(dummy_data)["feat"].shape)
    print(convnext_base()(dummy_data)["feat"].shape)
