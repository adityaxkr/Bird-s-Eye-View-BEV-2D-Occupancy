import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from config.config import IMG_CHANNELS
from logger.custom_logger import CustomLogger
from exception.custom_exception import BEVException

logger = CustomLogger().get_logger(__name__)


class ImageBackbone(nn.Module):
    """
    ResNet50 + FPN backbone.

    Processes all 6 camera images with SHARED weights —
    meaning one backbone learns features for all cameras.

    Input:  (B*6, 3, IMG_H, IMG_W)
    Output: (B*6, IMG_CHANNELS, IMG_H/16, IMG_W/16)

    With default settings:
        Input  → (12, 3, 256, 704)
        Output → (12, 128, 16, 44)
    """

    def __init__(self,
                 out_channels: int = IMG_CHANNELS,
                 pretrained:   bool = True):
        super().__init__()

        try:
            # ── Load pretrained ResNet50 ────────────────
            weights = (
                models.ResNet101_Weights.DEFAULT
                if pretrained else None
            )
            resnet = models.resnet101(weights=weights)

            # ── Encoder: break ResNet into stages ───────
            # Each stage reduces spatial size
            self.layer0 = nn.Sequential(
                resnet.conv1,    # 3   → 64, stride 2
                resnet.bn1,
                resnet.relu,
                resnet.maxpool   # stride 2  (total: stride 4)
            )
            self.layer1 = resnet.layer1  # 64  → 256,  stride 4
            self.layer2 = resnet.layer2  # 256 → 512,  stride 8
            self.layer3 = resnet.layer3  # 512 → 1024, stride 16

            # ── FPN neck: reduce channels ────────────────
            # Both layers → same out_channels for easy fusion
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

            # ── Output conv: refine fused features ───────
            self.output_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

            self.out_channels = out_channels

            logger.info(
                f"ImageBackbone initialized | "
                f"out_channels: {out_channels} | "
                f"pretrained: {pretrained}"
            )

        except Exception as e:
            raise BEVException(
                "Failed to initialize ImageBackbone", e
            ) from e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ResNet50 + FPN.

        Args:
            x: (B*6, 3, H, W) — all camera images flattened

        Returns:
            features: (B*6, out_channels, H/16, W/16)
        """
        try:
            # ── Encode through ResNet stages ─────────────
            x  = self.layer0(x)   # (B*6, 64,   H/4,  W/4)
            x  = self.layer1(x)   # (B*6, 256,  H/4,  W/4)
            c2 = self.layer2(x)   # (B*6, 512,  H/8,  W/8)
            c3 = self.layer3(c2)  # (B*6, 1024, H/16, W/16)

            # ── FPN top-down pathway ──────────────────────
            # Reduce channels of deepest layer
            p3 = self.fpn_layer3(c3)   # (B*6, 128, H/16, W/16)

            # Reduce channels of middle layer
            p2 = self.fpn_layer2(c2)   # (B*6, 128, H/8,  W/8)

            # Upsample p3 to match p2 spatial size
            p3_up = F.interpolate(
                p3,
                size            = p2.shape[-2:],
                mode            = 'bilinear',
                align_corners   = False
            )                          # (B*6, 128, H/8, W/8)

            # Fuse: add deep + shallow features
            fused = p3_up + p2         # (B*6, 128, H/8, W/8)

            # Refine fused features
            out = self.output_conv(fused)  # (B*6, 128, H/8, W/8)

            return out

        except Exception as e:
            raise BEVException(
                "ImageBackbone forward pass failed", e
            ) from e