import cv2
from matplotlib import pyplot as plt
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from typing import Optional, List

from .mobilenetv3 import MobileNetV3LargeEncoder
from .resnet import ResNet50Encoder
from .lraspp import LRASPP
from .decoder import RecurrentDecoder, Projection
from .fast_guided_filter import FastGuidedFilterRefiner
from .deep_guided_filter import DeepGuidedFilterRefiner

class MattingNetwork(nn.Module):
    def __init__(self,
                 variant: str = 'mobilenetv3',
                 refiner: str = 'deep_guided_filter',
                 pretrained_backbone: bool = False):
        super().__init__()
        assert variant in ['mobilenetv3', 'resnet50']
        assert refiner in ['fast_guided_filter', 'deep_guided_filter']
        
        if variant == 'mobilenetv3':
            self.backbone = MobileNetV3LargeEncoder(pretrained_backbone)
            self.aspp = LRASPP(960, 128)
            self.decoder = RecurrentDecoder([16, 24, 40, 128], [80, 40, 32, 16])
        else:
            self.backbone = ResNet50Encoder(pretrained_backbone)
            self.aspp = LRASPP(2048, 256)
            self.decoder = RecurrentDecoder([64, 256, 512, 256], [128, 64, 32, 16])
            
        self.project_mat = Projection(16, 4)
        self.project_seg = Projection(16, 1)

        if refiner == 'deep_guided_filter':
            self.refiner = DeepGuidedFilterRefiner()
        else:
            self.refiner = FastGuidedFilterRefiner()
        
    def forward(self,
                src: Tensor,
                r1: Optional[Tensor] = None,
                r2: Optional[Tensor] = None,
                r3: Optional[Tensor] = None,
                r4: Optional[Tensor] = None,
                index: int = 0,
                downsample_ratio: float = 1,
                segmentation_pass: bool = False):
        if downsample_ratio != 1:
            src_sm = self._interpolate(src, scale_factor=downsample_ratio)
        else:
            src_sm = src
        rs=[r1,r2,r3,r4]
        f1, f2, f3, f4 = self.backbone(src_sm)
        encoded = [f1,f2,f3,f4]
        f4 = self.aspp(f4)
        hid, *rec = self.decoder(src_sm, f1, f2, f3, f4, r1, r2, r3, r4)
        if not segmentation_pass:
            fgr_residual, pha = self.project_mat(hid).split([3, 1], dim=-3)
            if downsample_ratio != 1:
                fgr_residual, pha = self.refiner(src, src_sm, fgr_residual, pha, hid) # display 
             
                # src: torch.Size([1, 12, 3, 1280, 720])
                # src_sm: torch.Size([1, 12, 3, 320, 180])
                # pha: torch.Size([1, 12, 1, 1280, 720])
                # fgr_residual: torch.Size([1, 12, 3, 1280, 720])
                
                if index == 48:
                    for i in range(len(src[0])):
                        src_np = src[0][i].permute(1, 2, 0).cpu().numpy()
                        pha_np = pha[0][i].permute(1, 2, 0).cpu().numpy()
                        src_sm_np = src_sm[0][i].permute(1, 2, 0).cpu().numpy()
                        comp = src_np * pha_np
                        
                        
                        for j in range(len(rs[0][0])):
                            _, ax = plt.subplots(3, len(rs), figsize=(20, 10))
                            ax[0, 0].imshow(src_np)
                            ax[0, 0].set_title("Frame")
                            ax[0, 1].imshow(src_sm_np)
                            ax[0, 1].set_title("Frame downsampled")
                            ax[0, 2].imshow(pha_np)
                            ax[0, 2].set_title("Pha")
                            ax[0, 3].imshow(comp)
                            ax[0, 3].set_title("Composite")
                            for k in range(len(rs)):
                                ax[1, k].imshow(rs[k][0][j].cpu().numpy(), cmap="gray")
                                ax[1, k].set_title("Decoder {} Channel {}".format(k + 1, j + 1))
                            for k in range(len(encoded)):
                                ax[2, k].imshow(encoded[k][0][0][j].cpu().numpy(), cmap="gray")
                                ax[2, k].set_title("Encoder {} Channel {}".format(k + 1, j + 1))
                            
                            plt.suptitle("Encoder-Decoder")
                            plt.subplots_adjust(
                                top=0.953,
                                bottom=0.036,
                                left=0.008,
                                right=0.992,
                                hspace=0.2,
                                wspace=0.2,
                            )
                            plt.show()

            fgr = fgr_residual + src
            fgr = fgr.clamp(0., 1.)
            pha = pha.clamp(0., 1.)
            return [fgr, pha, *rec]
        else:
            seg = self.project_seg(hid)
            return [seg, *rec]

    def _interpolate(self, x: Tensor, scale_factor: float):
        if x.ndim == 5:
            B, T = x.shape[:2]
            x = F.interpolate(x.flatten(0, 1), scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
            x = x.unflatten(0, (B, T))
        else:
            x = F.interpolate(x, scale_factor=scale_factor,
                mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return x
