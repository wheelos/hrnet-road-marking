import io
import json
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest
import torch

from hrnet_road_marking.arrow_dataset import complete_window
from hrnet_road_marking.dataset import Vistas, flip_labels
from hrnet_road_marking.prepare import inspect_instance, main as prepare_main
from hrnet_road_marking.runtime import image_tensor
from hrnet_road_marking.sampling import CoarseOrder
from hrnet_road_marking.taxonomy import GROUPS, LOOKUP, grouped_logits, metrics


def test_taxonomy_is_total_and_nonoverlapping():
    assert sorted(c for group in GROUPS for c in group) == list(range(124))
    assert LOOKUP[255] == 255
    assert LOOKUP[38] == 0
    assert list(LOOKUP[[35, 39, 45, 51, 50, 37]]) == [2, 3, 4, 5, 6, 7]


def test_grouped_probability_reference():
    logits = torch.randn(1, 124, 3, 4)
    expected = torch.cat([logits.softmax(1)[:, ids].sum(1, keepdim=True) for ids in GROUPS], 1)
    torch.testing.assert_close(grouped_logits(logits).softmax(1), expected)


def test_arrow_flip_roundtrip():
    mask = np.array([[39, 40, 41, 42, 43, 44, 255]], dtype=np.uint8)
    assert np.array_equal(mask, flip_labels(flip_labels(mask)))


class SamplingFixture:
    records = [None] * 6
    record_image_indices = [1, 2, 3, 4, 5, 6]

    def __len__(self):
        return 11


def test_resume_sampling_suffix_and_mix():
    data = SamplingFixture()
    sequence = list(CoarseOrder(data, 100))
    for start in [1, 3, 4, 17, 33, 99]:
        assert list(CoarseOrder(data, 100, start)) == sequence[start:]
    assert sum(len(item) == 3 for item in sequence) == 25


@pytest.mark.parametrize("size,box", [((1920, 1084), (900, 500, 1150, 900)), ((320, 240), (1, 1, 319, 239))])
def test_complete_arrow_crops_contain_annotation(size, box):
    for seed in range(20):
        left, top, right, bottom = complete_window(box, size, (512, 768), np.random.default_rng(seed))
        assert left <= box[0] < box[2] <= right
        assert top <= box[1] < box[3] <= bottom


def test_rgb_preprocessing():
    image = Image.fromarray(np.array([[[255, 0, 0]]], dtype=np.uint8))
    expected = torch.tensor([(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225])
    torch.testing.assert_close(image_tensor(image)[0, :, 0, 0], expected)


def test_palette_ids_and_ignore(tmp_path):
    for folder in ["validation/images", "validation/v2.0/labels"]:
        (tmp_path / folder).mkdir(parents=True)
    labels = [dict(evaluate=(i != 1)) for i in range(124)]
    (tmp_path / "config_v2.0.json").write_text(json.dumps(dict(labels=labels)))
    Image.new("RGB", (2, 2)).save(tmp_path / "validation/images/a.jpg")
    mask = Image.fromarray(np.array([[35, 39], [1, 255]], dtype=np.uint8)).convert("P")
    mask.putpalette([255] * 768)  # Deliberately identical RGB colors.
    mask.save(tmp_path / "validation/v2.0/labels/a.png")
    _, target = Vistas(tmp_path, "validation")[0]
    assert target.tolist() == [[35, 39], [255, 255]]


def test_adjacent_arrow_instances_stay_separate(tmp_path):
    instances = np.full((40, 60), 13 * 256, dtype=np.uint16)
    instances[3:23, 3:15] = 40 * 256 + 1
    instances[3:23, 15:27] = 40 * 256 + 2
    payload = io.BytesIO()
    Image.fromarray(instances).save(payload, format="PNG")
    labels = tmp_path / "labels.png"
    Image.fromarray((instances // 256).astype(np.uint8)).save(labels)
    rows = inspect_instance(payload.getvalue(), labels, "image")
    assert len(rows) == 2
    assert all(r["eligible"] for r in rows)
    assert rows[0]["bbox"] == [3, 3, 15, 23]
    assert rows[1]["bbox"] == [15, 3, 27, 23]


def test_error_metric_denominators():
    confusion = np.zeros((8, 8), dtype=np.int64)
    confusion[1, 1], confusion[1, 2], confusion[1, 3] = 90, 6, 4
    confusion[2, 2], confusion[2, 3] = 98, 2
    fine = np.zeros((124, 8), dtype=np.int64)
    fine[40, 3], fine[40, 2] = 9, 1
    values = metrics(confusion, fine)
    assert values["flat_to_marking"] == 0.1
    assert values["lane_to_filter"] == 0.02
    assert values["arrows_filter_recall"] == 0.9


@pytest.mark.parametrize("archive", [False, True])
def test_prepare_command_indexes_splits_by_image_id(tmp_path, monkeypatch, archive):
    data, instances, output = tmp_path / "data", tmp_path / "instances", tmp_path / "index"
    data.mkdir()
    instances.mkdir()
    (data / "config_v2.0.json").write_text(json.dumps({"labels": [{"evaluate": True}] * 124}))
    for split in ["training", "validation"]:
        (data / split / "images").mkdir(parents=True)
        (data / split / "v2.0/labels").mkdir(parents=True)
        Image.new("RGB", (60, 40)).save(data / split / "images" / (split + ".jpg"))
        ids = np.full((40, 60), 13 * 256, dtype=np.uint16)
        ids[5:25, 10:30] = 40 * 256 + 1
        Image.fromarray((ids // 256).astype(np.uint8)).save(data / split / "v2.0/labels" / (split + ".png"))
        Image.fromarray(ids).save(instances / (split + ".png"))
    source = instances
    if archive:
        source = tmp_path / "instances.zip"
        with zipfile.ZipFile(source, "w") as destination:
            for path in instances.glob("*.png"):
                destination.write(path, "prefix/training/v2.0/instances/instances/" + path.name)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--data",
            str(data),
            "--instances",
            str(source),
            "--output",
            str(output),
            "--allow-subset",
            "--workers",
            "1",
        ],
    )
    prepare_main()
    result = json.loads((output / "arrow_instances.json").read_text())
    assert len(result["training"]) == len(result["validation"]) == 1
    assert result["training"][0]["image"] == "training"
    assert result["training"][0]["eligible"]
    assert json.loads((output / "validation_ids.json").read_text()) == ["validation"]
