"""HRNet-W48 + OCR with the validated 0.5/1.0 attention fusion."""

import torch
from torch import nn
import torch.nn.functional as F
from ._vendor.config import cfg
from ._vendor.network.ocrnet import HRNet_Mscale
from .taxonomy import LABELS


def build(classes=8):
    cfg.DATASET.NUM_CLASSES = classes
    cfg.MODEL.BNFUNC = nn.BatchNorm2d
    cfg.MODEL.HRNET_CHECKPOINT = ""
    cfg.OPTIONS.TORCH_VERSION = 2.1
    cfg.MODEL.N_SCALES = None
    return HRNet_Mscale(classes, None)


def load_checkpoint(path, device="cpu"):
    # Load only checkpoints from a trusted source: torch.load uses pickle.
    state = torch.load(path, map_location="cpu")
    if state.get("labels") != LABELS:
        raise ValueError("Checkpoint does not have the expected eight-class taxonomy")
    model = build()
    model.load_state_dict(state["state_dict"], strict=True)
    return model.to(device).eval(), state


def forward(model, images):
    """Original 0.5/1.0 attention fusion, exposing both supervised outputs.

    The existing inference source accepts a tensor but its old training branch
    expects a dict. Calling _fwd explicitly avoids that incompatible branch.
    Bilinear interpolation and attention weights follow two_scale_forward.
    """
    low = model._fwd(
        F.interpolate(
            images, scale_factor=0.5, mode="bilinear", align_corners=False, recompute_scale_factor=True
        )
    )
    high = model._fwd(images)
    size = high["cls_out"].shape[-2:]

    def resize(x):
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    attention = low["logit_attn"]
    complement = 1 - resize(attention)
    main = resize(attention * low["cls_out"]) + complement * high["cls_out"]
    auxiliary = resize(attention * low["aux_out"]) + complement * high["aux_out"]
    return main, auxiliary


def freeze_bn_statistics(model):
    # Batch size one cannot estimate reliable statistics. Affine parameters
    # still receive gradients; pretrained running means/variances stay fixed.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
