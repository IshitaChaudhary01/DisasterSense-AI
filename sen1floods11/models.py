import os
import torch
import torch.nn as nn
import torchvision.models as models

def convert_bn_to_gn(module, num_groups=16):
    """
    Recursively replaces all BatchNorm2d layers with GroupNorm layers.
    Essential for small batch sizes in remote sensing / satellite segmentation.
    """
    if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
        gn = nn.GroupNorm(num_groups, module.num_features, eps=module.eps, affine=module.affine)
        if module.affine and module.weight is not None and module.bias is not None:
            gn.weight.data = module.weight.data.clone().detach()
            gn.bias.data = module.bias.data.clone().detach()
        return gn

    for name, child in module.named_children():
        module.add_module(name, convert_bn_to_gn(child, num_groups=num_groups))

    return module


def build_fcn_resnet50(in_channels=2, num_classes=2, pretrained_backbone=False):
    """
    Builds the Sen1Floods11 FCN-ResNet50 segmentation architecture.
    Accepts 2-channel Sentinel-1 SAR input (VV, VH) or custom in_channels.
    """
    try:
        # Modern torchvision syntax
        weights_backbone = models.ResNet50_Weights.DEFAULT if pretrained_backbone else None
        net = models.segmentation.fcn_resnet50(
            weights=None,
            weights_backbone=weights_backbone,
            num_classes=num_classes
        )
    except Exception:
        # Fallback to legacy syntax
        net = models.segmentation.fcn_resnet50(
            pretrained=False,
            pretrained_backbone=pretrained_backbone,
            num_classes=num_classes
        )

    # Adapt first convolution for 2-channel SAR input (VV, VH)
    if in_channels != 3:
        original_conv1 = net.backbone.conv1
        net.backbone.conv1 = nn.Conv2d(
            in_channels,
            original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=False
        )

    # Convert all BatchNorm2d to GroupNorm (16 groups) as in Sen1Floods11 paper
    net = convert_bn_to_gn(net, num_groups=16)

    return net


def load_model(checkpoint_path=None, in_channels=2, num_classes=2, device='cpu'):
    """
    Instantiates and optionally loads state_dict weights for FCN-ResNet50.
    """
    model = build_fcn_resnet50(in_channels=in_channels, num_classes=num_classes)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        # Check if wrapped in checkpoint dictionary
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint weights from {checkpoint_path}")
        
    model.to(device)
    model.eval()
    return model
