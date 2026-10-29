"""Shared, explicit preprocessing and reproducibility settings."""

from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def configure():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def autocast(device, precision):
    if precision == "fp32":
        return nullcontext()
    if precision != "bf16" or torch.device(device).type != "cuda":
        raise ValueError("BF16 requires a CUDA device; select fp32 for CPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This GPU does not support BF16")
    return torch.autocast("cuda", dtype=torch.bfloat16)


def image_tensor(image):
    array = np.array(image.convert("RGB")).transpose(2, 0, 1).copy()
    x = torch.from_numpy(array).float() / 255
    return ((x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None])[None]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)
