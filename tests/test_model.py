import torch

from hrnet_road_marking.model import build
from hrnet_road_marking.runtime import configure


def test_native_eight_class_output():
    configure()
    model = build().eval()
    assert model.ocr.cls_head.out_channels == 8
    assert model.ocr.aux_head[2].out_channels == 8
    with torch.inference_mode():
        logits = model(torch.zeros(1, 3, 64, 96))
    assert logits.shape == (1, 8, 64, 96)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits.softmax(1).sum(1), torch.ones(1, 64, 96))
