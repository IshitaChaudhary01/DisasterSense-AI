import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

def convert_bn_to_gn(module, num_groups=16):
    """
    Recursively replaces BatchNorm2d with GroupNorm.
    Essential for satellite remote sensing with small batch sizes.
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


class SAREncoder(nn.Module):
    """
    ResNet-based SAR feature extractor adapted for 2-channel Sentinel-1 input (VV, VH).
    """
    def __init__(self, in_channels=2, backbone="resnet18"):
        super().__init__()
        if backbone == "resnet18":
            resnet = models.resnet18(weights=None)
            self.feature_dim = 512
        elif backbone == "resnet34":
            resnet = models.resnet34(weights=None)
            self.feature_dim = 512
        else:
            resnet = models.resnet50(weights=None)
            self.feature_dim = 2048

        # Adapt first convolution for 2-channel SAR
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Convert to GroupNorm
        convert_bn_to_gn(self, num_groups=16)

    def forward(self, x):
        # x: (B, 2, H, W)
        x0 = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H/2, W/2)
        x_pool = self.maxpool(x0)                 # (B, 64, H/4, W/4)

        c1 = self.layer1(x_pool)                 # (B, 64, H/4, W/4)
        c2 = self.layer2(c1)                     # (B, 128, H/8, W/8)
        c3 = self.layer3(c2)                     # (B, 256, H/16, W/16)
        c4 = self.layer4(c3)                     # (B, 512, H/32, W/32)

        return {"x0": x0, "c1": c1, "c2": c2, "c3": c3, "c4": c4}


class SimCLRModel(nn.Module):
    """
    SimCLR Contrastive Self-Supervised Learning Framework for SAR imagery.
    Encoder f(x) -> Representation h -> 2-layer MLP Projector g(h) -> Projection z
    """
    def __init__(self, in_channels=2, backbone="resnet18", proj_dim=128):
        super().__init__()
        self.encoder = SAREncoder(in_channels=in_channels, backbone=backbone)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = self.encoder.feature_dim
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, proj_dim)
        )

    def forward(self, x):
        features = self.encoder(x)["c4"]
        h = self.pool(features).flatten(1)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


def nt_xent_loss(z1, z2, temperature=0.1):
    """
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss.
    Maximizes agreement between differently augmented views of the same image
    while pushing representations of distinct images apart.
    """
    batch_size = z1.size(0)
    device = z1.device

    # Concatenate views: 2 * batch_size
    z = torch.cat([z1, z2], dim=0)

    # Cosine similarity matrix: (2B, 2B)
    sim_matrix = torch.matmul(z, z.T) / temperature

    # Mask to eliminate self-similarity on the diagonal
    diag_mask = torch.eye(2 * batch_size, device=device, dtype=torch.bool)
    sim_matrix = sim_matrix.masked_fill(diag_mask, -1e9)

    # Positive targets: view 1 matches view 2 and vice-versa
    positives = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=device),
        torch.arange(0, batch_size, device=device)
    ])

    loss = F.cross_entropy(sim_matrix, positives)
    return loss


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(16, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(16, out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            # Handle slight padding/dimension differences if any
            if x.shape != skip.shape:
                x = F.interpolate(x, size=(skip.shape[2], skip.shape[3]), mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.relu(self.gn2(self.conv2(x)))
        return x


class SAR_UNet(nn.Module):
    """
    U-Net Semantic Segmentation Architecture for SAR Flood Water Detection.
    Connects to the pretrained SAREncoder with multi-scale skip connections.
    """
    def __init__(self, in_channels=2, num_classes=2, backbone="resnet18"):
        super().__init__()
        self.encoder = SAREncoder(in_channels=in_channels, backbone=backbone)

        # ResNet18 dimensions: c4: 512, c3: 256, c2: 128, c1: 64, x0: 64
        self.dec4 = DecoderBlock(512, 256, 256)
        self.dec3 = DecoderBlock(256, 128, 128)
        self.dec2 = DecoderBlock(128, 64, 64)
        self.dec1 = DecoderBlock(64, 64, 32)

        self.final_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.classifier = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, x):
        # Encoder forward pass
        feats = self.encoder(x)
        x0, c1, c2, c3, c4 = feats["x0"], feats["c1"], feats["c2"], feats["c3"], feats["c4"]

        # Decoder forward pass with skip connections
        d4 = self.dec4(c4, c3)
        d3 = self.dec3(d4, c2)
        d2 = self.dec2(d3, c1)
        d1 = self.dec1(d2, x0)

        out = self.final_upsample(d1)
        logits = self.classifier(out)

        # Ensure exact input spatial size matching
        if logits.shape[2:] != x.shape[2:]:
            logits = F.interpolate(logits, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        return logits

    def load_ssl_encoder(self, ssl_state_dict_path, device="cpu"):
        """
        Loads pretrained weights from self-supervised SimCLR pretraining.
        """
        checkpoint = torch.load(ssl_state_dict_path, map_location=device)
        encoder_state = {}
        for k, v in checkpoint.items():
            if k.startswith("encoder."):
                encoder_state[k[len("encoder."):]] = v
            elif not k.startswith("projector."):
                encoder_state[k] = v

        missing, unexpected = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"Loaded SSL pretrained encoder ({len(encoder_state)} weights).")
        print(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")


class DiceLoss(nn.Module):
    """
    Dice loss function for binary flood water segmentation.
    Directly maximizes spatial intersection over union (IoU).
    """
    def __init__(self, smooth=1.0, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # logits: (B, 2, H, W), targets: (B, H, W)
        probs = F.softmax(logits, dim=1)[:, 1]  # Water probability

        valid = (targets != self.ignore_index)
        probs = probs[valid]
        targets_bin = (targets[valid] == 1).float()

        intersection = (probs * targets_bin).sum()
        union = probs.sum() + targets_bin.sum()

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice


class CombinedFloodLoss(nn.Module):
    """
    Weighted Cross Entropy (1:8 to combat flood class imbalance) + Dice Loss.
    """
    def __init__(self, ce_weight=(1.0, 8.0), ignore_index=255, dice_weight=0.5):
        super().__init__()
        weight = torch.tensor(ce_weight).float()
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce_loss = self.ce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return ce_loss + self.dice_weight * dice_loss
