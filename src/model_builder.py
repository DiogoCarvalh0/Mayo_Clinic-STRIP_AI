import torch
import torchvision
import torchvision.models as models
from torch import nn
from torch.nn import functional as F


def create_efficientnet_b0_model():
    weights = models.EfficientNet_B0_Weights.DEFAULT
    model = models.efficientnet_b0(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    # model.classifier = nn.Sequential(
    #    nn.Linear(in_features=1280, out_features=16, bias=True),
    #    nn.ReLU(),
    #    nn.Dropout(p=0.40),
    #    nn.Linear(in_features=16, out_features=16, bias=True),
    #    nn.ReLU(),
    #    nn.Dropout(p=0.40),
    #    nn.Linear(in_features=16, out_features=1, bias=True),
    # )

    model.classifier = nn.Sequential(nn.Dropout(p=0.50), nn.Linear(in_features=1280, out_features=1, bias=True))

    return model


def create_efficientnet_b4_model():
    weights = models.EfficientNet_B4_Weights.DEFAULT
    model = models.efficientnet_b4(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.50),
        nn.Linear(in_features=1792, out_features=1, bias=True),
    )

    return model


def create_efficientnet_v2_model(size):
    if size == "s" or size == "small":
        weights = models.EfficientNet_V2_S_Weights.DEFAULT
        model = models.efficientnet_v2_s(weights=weights)

    if size == "m" or size == "medium":
        weights = models.EfficientNet_V2_M_Weights.DEFAULT
        model = models.efficientnet_v2_m(weights=weights)

    if size == "l" or size == "large":
        weights = models.EfficientNet_V2_L_Weights.DEFAULT
        model = models.efficientnet_v2_l(weights=weights)

    for param in model.features.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.50),
        nn.Linear(in_features=1280, out_features=1, bias=True),
    )

    return model


def create_convnext_model(size):
    if size == "t" or size == "tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = models.convnext_tiny(weights=weights)
        in_features = 768

    if size == "s" or size == "small":
        weights = models.ConvNeXt_Small_Weights.DEFAULT
        model = models.convnext_small(weights=weights)
        in_features = 768

    if size == "b" or size == "base":
        weights = models.ConvNeXt_Base_Weights.DEFAULT
        model = models.convnext_base(weights=weights)
        in_features = 1024

    if size == "l" or size == "large":
        weights = models.ConvNeXt_Large_Weights.DEFAULT
        model = models.convnext_large(weights=weights)
        in_features = 1536

    for param in model.features.parameters():
        param.requires_grad = False

    class LayerNorm2d(nn.LayerNorm):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.permute(0, 2, 3, 1)
            x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
            x = x.permute(0, 3, 1, 2)
            return x

    model.classifier = nn.Sequential(
        LayerNorm2d((in_features,), eps=1e-06, elementwise_affine=True),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=in_features, out_features=1, bias=True),
    )

    return model
