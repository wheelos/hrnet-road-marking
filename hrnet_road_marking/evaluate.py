"""Evaluate direct eight-class output, retaining original-label audit counts."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .model import load_checkpoint
from .runtime import autocast, configure, sha256, write_json
from .sampling import validation
from .taxonomy import LOOKUP, metrics


@torch.inference_mode()
def evaluate(model, dataset, output, step, precision="bf16", workers=2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    lookup = torch.tensor(LOOKUP.astype(np.int64), device=device)
    confusion = torch.zeros((8, 8), dtype=torch.int64, device=device)
    fine = torch.zeros((124, 8), dtype=torch.int64, device=device)
    for index, (image, original) in enumerate(DataLoader(dataset, batch_size=1, num_workers=workers)):
        original = original.to(device)
        target = lookup[original]
        valid = original != 255
        with autocast(device, precision):
            logits = model(image.to(device))
        prediction = logits.argmax(1)
        confusion += torch.bincount(8 * target[valid] + prediction[valid], minlength=64).reshape(8, 8)
        fine += torch.bincount(8 * original[valid] + prediction[valid], minlength=992).reshape(124, 8)
        if (index + 1) % 100 == 0:
            logging.info("validation images=%d/%d", index + 1, len(dataset))
    matrix, fine_matrix = confusion.cpu().numpy(), fine.cpu().numpy()
    result = metrics(matrix, fine_matrix)
    result.update(step=step, images=len(dataset), decode="direct8", precision=precision, max_image_side=1536)
    write_json(output / f"step_{step:06}.json", result)
    np.savez_compressed(output / f"step_{step:06}.npz", confusion=matrix, fine_confusion=fine_matrix)
    logging.info(
        "step=%d lane=%.4f exclude=%.4f mean=%.4f",
        step,
        result["iou"][2],
        result["iou"][3],
        result["lane_filter_miou"],
    )
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--validation-ids", type=Path, help="Omit to evaluate the full validation split")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--limit", type=int, help="Smoke test only; written into the report")
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    configure()
    logging.basicConfig(level=logging.INFO)
    model, state = load_checkpoint(a.checkpoint, a.device)
    step = state["step"]
    del state
    data = validation(a.data, a.validation_ids)
    if a.limit is not None:
        if a.limit < 1:
            p.error("--limit must be positive")
        data.images = data.images[: a.limit]
    result = evaluate(model, data, a.output, step, a.precision, a.workers)
    write_json(
        a.output / "evaluation.json",
        dict(
            checkpoint_sha256=sha256(a.checkpoint),
            validation_ids_sha256=sha256(a.validation_ids) if a.validation_ids else None,
            limit=a.limit,
            result=result,
        ),
    )


if __name__ == "__main__":
    main()
