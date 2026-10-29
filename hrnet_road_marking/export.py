"""Export normalized RGB NCHW input to eight-class NCHW logits (no softmax)."""

import argparse
import json
from pathlib import Path

import torch

from .model import load_checkpoint
from .runtime import configure, sha256
from .taxonomy import NAMES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-sha256")
    a = p.parse_args()
    import onnx

    configure()
    digest = sha256(a.checkpoint)
    if a.expected_sha256 and digest != a.expected_sha256:
        p.error("Checkpoint SHA256 mismatch")
    model, state = load_checkpoint(a.checkpoint)
    step = state["step"]
    del state
    a.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.output.with_suffix(".tmp.onnx")
    with torch.inference_mode():
        torch.onnx.export(
            model,
            torch.zeros(1, 3, 512, 768),
            str(temporary),
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
            dynamic_axes={"image": {2: "height", 3: "width"}, "logits": {2: "height", 3: "width"}},
        )
    exported = onnx.load(str(temporary))
    onnx.checker.check_model(exported)
    onnx.helper.set_model_props(
        exported,
        {
            "checkpoint_sha256": digest,
            "checkpoint_step": str(step),
            "semantic_contract": "ramp_native_coarse8_logits_v1",
            "labels": json.dumps(NAMES),
            "input": "RGB NCHW FP32; /255; mean=.485,.456,.406; std=.229,.224,.225; preprocessing outside graph",
            "output": "NCHW FP32 logits; class softmax and argmax outside graph",
        },
    )
    onnx.save(exported, str(temporary))
    temporary.replace(a.output)
    print(a.output, "sha256=" + sha256(a.output))


if __name__ == "__main__":
    main()
