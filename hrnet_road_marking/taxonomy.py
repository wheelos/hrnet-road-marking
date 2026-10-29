"""Eight-class task definition, datasets and direct classifier initialization."""

import math

import numpy as np
import torch


NAMES = [
    "background",
    "flat_surface",
    "lane_marking",
    "exclude_marking",
    "crosswalk",
    "stop_line",
    "other_marking",
    "zigzag_marking",
]
COLORS = [
    (0, 0, 0),
    (100, 100, 100),
    (0, 180, 255),
    (255, 65, 65),
    (190, 70, 220),
    (30, 220, 120),
    (245, 220, 40),
    (255, 130, 0),
]
GROUPS = [
    [],
    list(range(13, 26)),
    [35, 36, 55],
    [39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 52, 53, 54, 58],
    [45, 56],
    [51],
    [50, 57],
    [37],
]
GROUPS[0] = sorted(set(range(124)) - {v for group in GROUPS[1:] for v in group})
LOOKUP = np.full(256, 255, dtype=np.uint8)
for group_id, members in enumerate(GROUPS):
    LOOKUP[members] = group_id
LABELS = [
    dict(name=n, readable=n, color=COLORS[i], evaluate=True, original_ids=GROUPS[i])
    for i, n in enumerate(NAMES)
]
CLASSIFIERS = ("ocr.cls_head.weight", "ocr.cls_head.bias", "ocr.aux_head.2.weight", "ocr.aux_head.2.bias")
LOSS_WEIGHTS = [1, 1, 4, 4, 2, 2, 1, 1]


def grouped_logits(logits):
    # log(sum(exp(logit))) gives the exact sum of original softmax probability
    # mass after a softmax across the eight groups, without taking an argmax first.
    return torch.cat([torch.logsumexp(logits[:, ids].float(), 1, keepdim=True) for ids in GROUPS], 1)


def initialize_coarse(model, checkpoint):
    parent = torch.load(checkpoint, map_location="cpu")
    if len(parent["labels"]) != 124:
        raise ValueError("Initialization requires a 124-class HRNet-OCR checkpoint")
    source = parent["state_dict"]
    target = model.state_dict()
    assert set(source) == set(target)
    for key, tensor in target.items():
        if key not in CLASSIFIERS:
            assert tensor.shape == source[key].shape, key
            target[key] = source[key]
        else:
            assert tensor.shape[0] == 8 and source[key].shape[0] == 124
            for c, ids in enumerate(GROUPS):
                if key.endswith("weight"):
                    tensor[c].copy_(source[key][ids].mean(0))
                else:
                    tensor[c].copy_(source[key][ids].mean(0) + math.log(len(ids)))
    model.load_state_dict(target, strict=True)
    for key, tensor in model.state_dict().items():
        if key not in CLASSIFIERS:
            assert torch.equal(tensor, source[key]), key
    return dict(
        parent=str(checkpoint),
        parent_step=parent["step"],
        output_classes=8,
        changed_tensors=list(CLASSIFIERS),
        initialization="mean classifier rows; bias mean + log(group size)",
        exact_parent_probability_equivalence=False,
        note="Direct eight-class OCR also changes object context; validate and train before use.",
    )


def metrics(matrix, fine_matrix):
    tp = matrix.diagonal().astype(float)
    truth, predicted = matrix.sum(1), matrix.sum(0)
    iou = np.divide(tp, truth + predicted - tp, out=np.zeros(8), where=(truth + predicted - tp) > 0)
    recall = np.divide(tp, truth, out=np.zeros(8), where=truth > 0)
    precision = np.divide(tp, predicted, out=np.zeros(8), where=predicted > 0)
    result = dict(
        iou=iou.tolist(),
        recall=recall.tolist(),
        precision=precision.tolist(),
        lane_filter_miou=float(iou[[2, 3]].mean()),
        lane_to_filter=float(matrix[2, 3] / max(1, truth[2])),
        filter_to_lane=float(matrix[3, 2] / max(1, truth[3])),
    )
    result.update(
        flat_to_lane=float(matrix[1, 2] / max(1, truth[1])),
        flat_to_filter=float(matrix[1, 3] / max(1, truth[1])),
        flat_to_marking=float((matrix[1, 2] + matrix[1, 3]) / max(1, truth[1])),
    )
    for name, ids in [
        ("arrows", list(range(39, 45))),
        ("hatching", [48, 49]),
        ("symbols_text", [52, 53, 54, 58]),
        ("give_way", [46, 47]),
    ]:
        total = int(fine_matrix[ids].sum())
        result[name + "_filter_recall"] = float(fine_matrix[ids, 3].sum() / max(1, total))
        result[name + "_to_lane"] = float(fine_matrix[ids, 2].sum() / max(1, total))
    return result
