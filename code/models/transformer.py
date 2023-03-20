from torchvision.models import  VisionTransformer,ViT_B_16_Weights
from torchvision.models.vision_transformer import EncoderBlock
import torch
from torchvision.models._utils import _ovewrite_named_param
from math import sqrt 

class CustomEncoderBlock(EncoderBlock):

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=True)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

class CustomVisionTransformer(VisionTransformer):

    def __init__(self,**kwargs) -> None:
        super().__init__(**kwargs)

        att_block = self.encoder.layers[-1]
        state_dict = att_block.state_dict()
        self.encoder.layers[-1] = CustomEncoderBlock(att_block.num_heads,self.hidden_dim,self.mlp_dim,self.dropout,self.attention_dropout,self.norm_layer)
        self.encoder.layers[-1].load_state_dict(state_dict)

        self.buffer = []

        def fn(_, __, output):
            self.buffer = output[1]

        self.encoder.layers[-1].self_attention.register_forward_hook(fn)

    def forward(self,x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.encoder(x)
        attMaps = self.buffer[:,0,1:]

        batch_size = attMaps.shape[0]
        map_size = int(sqrt(attMaps.shape[1]))
        attMaps = attMaps.reshape(batch_size,map_size,map_size)
        attMaps = attMaps.unsqueeze(1)

        x = x[:, 0]

        return {"feat_pooled":x,"attMaps":attMaps}

def vit_b_16(*,weights=None,progress=True,**kwargs) -> CustomVisionTransformer:
    """
    Constructs a vit_b_16 architecture from
    `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale <https://arxiv.org/abs/2010.11929>`_.
    Args:
        weights (:class:`~torchvision.models.ViT_B_16_Weights`, optional): The pretrained
            weights to use. See :class:`~torchvision.models.ViT_B_16_Weights`
            below for more details and possible values. By default, no pre-trained weights are used.
        progress (bool, optional): If True, displays a progress bar of the download to stderr. Default is True.
        **kwargs: parameters passed to the ``torchvision.models.vision_transformer.VisionTransformer``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/vision_transformer.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.ViT_B_16_Weights
        :members:
    """
    weights = ViT_B_16_Weights.verify(weights)

    return _vision_transformer(
        patch_size=16,
        num_layers=12,
        num_heads=12,
        hidden_dim=768,
        mlp_dim=3072,
        weights=weights,
        progress=progress,
        **kwargs,
    )

def _vision_transformer(
    patch_size: int,
    num_layers: int,
    num_heads: int,
    hidden_dim: int,
    mlp_dim: int,
    weights,
    progress: bool,
    **kwargs,
) -> VisionTransformer:
    if weights is not None:
        _ovewrite_named_param(kwargs, "num_classes", len(weights.meta["categories"]))
        assert weights.meta["min_size"][0] == weights.meta["min_size"][1]
        _ovewrite_named_param(kwargs, "image_size", weights.meta["min_size"][0])
    image_size = kwargs.pop("image_size", 224)

    model = CustomVisionTransformer(
        image_size=image_size,
        patch_size=patch_size,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        **kwargs,
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=progress))

    return model

if __name__== "__main__":
    dummy = torch.zeros(1,3,224,224)
    transf = vit_b_16(weights="IMAGENET1K_V1")
    output = transf(dummy)["feat_pooled"]