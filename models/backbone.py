# models/backbone.py
# ══════════════════════════════════════════════════════
# Image Backbone — supports ResNet50/101/152 + FPN
# UPDATED: configurable backbone for easy switching
# ══════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config.config import IMG_CHANNELS, BACKBONE
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class ImageBackbone(nn.Module):
    """
    Configurable ResNet + FPN backbone.
    Supports: resnet50, resnet101, resnet152

    ResNet101 gives ~+0.05-0.10 IoU over ResNet50
    due to deeper feature extraction.
    """

    def __init__(self,
                 out_channels: int  = IMG_CHANNELS,
                 pretrained:   bool = True,
                 backbone:     str  = BACKBONE):
        super().__init__()

        try:
            # ── Load backbone ───────────────────────────
            if backbone == 'resnet152':
                weights = (
                    models.ResNet152_Weights.DEFAULT
                    if pretrained else None
                )
                resnet = models.resnet152(weights=weights)

            elif backbone == 'resnet101':
                weights = (
                    models.ResNet101_Weights.DEFAULT
                    if pretrained else None
                )
                resnet = models.resnet101(weights=weights)

            else:  # resnet50 default
                weights = (
                    models.ResNet50_Weights.DEFAULT
                    if pretrained else None
                )
                resnet = models.resnet50(weights=weights)

            # ── Encoder stages ──────────────────────────
            # Same structure for all ResNet variants
            self.layer0 = nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
                resnet.maxpool
            )
            self.layer1 = resnet.layer1  # → 256
            self.layer2 = resnet.layer2  # → 512
            self.layer3 = resnet.layer3  # → 1024

            # ── FPN neck ────────────────────────────────
            self.fpn_layer3 = nn.Sequential(
                nn.Conv2d(1024, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.fpn_layer2 = nn.Sequential(
                nn.Conv2d(512, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.out_channels = out_channels

            # Count params
            total = sum(
                p.numel() for p in self.parameters()
            )
            logger.info(
                f"ImageBackbone | "
                f"backbone: {backbone} | "
                f"params: {total:,} | "
                f"pretrained: {pretrained}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to init ImageBackbone", e
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            x  = self.layer0(x)
            x  = self.layer1(x)
            c2 = self.layer2(x)
            c3 = self.layer3(c2)

            p3    = self.fpn_layer3(c3)
            p2    = self.fpn_layer2(c2)
            p3_up = F.interpolate(
                p3, size=p2.shape[-2:],
                mode='bilinear', align_corners=False
            )
            fused = p3_up + p2
            out   = self.output_conv(fused)

            return out

        except Exception as e:
            raise BEVException(
                "ImageBackbone forward failed", e
            ) from e