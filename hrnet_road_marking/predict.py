"""Native-resolution RGB inference; save direct class IDs and overlays."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from .model import load_checkpoint
from .runtime import configure, image_tensor, sha256, write_json
from .taxonomy import COLORS, NAMES


def overlay(image, mask):
    array = np.array(image.convert("RGB")).astype(np.float32)
    for c in range(2, 8):
        keep = mask == c
        array[keep] = 0.35 * array[keep] + 0.65 * np.array(COLORS[c])
    return Image.fromarray(array.astype(np.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True, help="One image or a directory, recursively")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--probabilities", action="store_true", help="Also save an 8xHxW float32 array")
    a = p.parse_args()
    configure()
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    paths = (
        [a.input]
        if a.input.is_file()
        else sorted(f for f in a.input.rglob("*") if f.suffix.lower() in extensions)
    )
    if a.limit < 0:
        p.error("--limit cannot be negative")
    if a.limit:
        paths = paths[: a.limit]
    if not paths:
        p.error("No input images")
    if a.input.is_dir() and a.output.resolve().is_relative_to(a.input.resolve()):
        p.error("Output must be outside the input tree")
    model, state = load_checkpoint(a.checkpoint, a.device)
    step = state["step"]
    del state
    records = []
    for path in paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
        with torch.inference_mode():
            logits = model(image_tensor(image).to(a.device))
            mask = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        relative = Path(path.name) if a.input.is_file() else path.relative_to(a.input)
        # Keep the input extension so foo.jpg and foo.png cannot collide.
        target = a.output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask).save(str(target) + ".labels.png")
        overlay(image, mask).save(str(target) + ".overlay.jpg", quality=92)
        if a.probabilities:
            np.save(str(target) + ".probabilities.npy", logits.softmax(1)[0].cpu().numpy())
        records.append(dict(image=str(relative), size=list(image.size)))
        print(relative, flush=True)
    write_json(
        a.output / "prediction.json",
        dict(
            checkpoint_sha256=sha256(a.checkpoint),
            step=step,
            classes=NAMES,
            precision="fp32",
            native_resolution=True,
            images=records,
        ),
    )


if __name__ == "__main__":
    main()
