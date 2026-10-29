"""Single-GPU fine-tuning, deterministic sampling and complete resume state."""

import argparse
import csv
import json
import logging
import math
from pathlib import Path
import random
import signal

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .evaluate import evaluate
from .model import build, forward, freeze_bn_statistics
from .runtime import autocast, configure, sha256, write_json
from .sampling import CoarseOrder, CoarseTraining, validation
from .taxonomy import CLASSIFIERS, LABELS, initialize_coarse


def save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--arrow-index", type=Path, required=True)
    p.add_argument("--validation-ids", type=Path)
    p.add_argument("--output", type=Path, required=True)
    initial = p.add_mutually_exclusive_group(required=True)
    initial.add_argument("--initial-checkpoint", type=Path, help="Eight-class weights; fresh optimizer")
    initial.add_argument(
        "--initial-124", type=Path, help="Trained 124-class HRNet-OCR; initialize eight heads"
    )
    initial.add_argument("--resume", type=Path, help="Resume latest.pth in the same output directory")
    p.add_argument(
        "--stop-after", type=int, help="Stop safely at this update, without changing the LR schedule"
    )
    p.add_argument("--limit-validation", type=int, help="Smoke test only, recorded in the run identity")
    p.add_argument("--skip-final-validation", action="store_true")
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    for key in ("steps", "accumulation", "validate_every", "save_every"):
        if config[key] < 1:
            p.error(f"{key} must be positive")
    if config["warmup"] < 0 or config["workers"] < 0 or config["sample_offset"] < 0:
        p.error("warmup, workers and sample_offset cannot be negative")
    if len(config["base_lr"]) != 3 or len(config["loss_weights"]) != 8:
        p.error("Expected three learning rates and eight loss weights")
    if any(x <= 0 for x in config["base_lr"] + config["loss_weights"]):
        p.error("Learning rates and loss weights must be positive")
    if a.limit_validation is not None and a.limit_validation < 1:
        p.error("--limit-validation must be positive")
    if a.stop_after is not None and not 0 < a.stop_after <= config["steps"]:
        p.error("--stop-after must be within the training schedule")
    if not torch.cuda.is_available():
        p.error("Training requires CUDA")
    configure()
    # Validate precision before loading large weights or creating output files.
    autocast("cuda", config["precision"])
    autocast("cuda", config["validation_precision"])
    if a.resume:
        if a.resume.resolve() != (a.output / "latest.pth").resolve():
            p.error("Resume the original run directory using its latest.pth")
        if (a.output / "finished.json").exists():
            p.error("This run is already finished; use --initial-checkpoint for a new run")
    elif a.output.exists() and any(a.output.iterdir()):
        p.error("New training requires an empty output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(a.output / "train.log")],
    )
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    data = CoarseTraining(a.data, a.arrow_index, seed=config["seed"], crop=tuple(config["crop"]))
    val = validation(a.data, a.validation_ids)
    if {x.stem for x in data.images} & {x.stem for x in val.images}:
        raise ValueError("Training and validation image IDs overlap")
    if a.limit_validation:
        val.images = val.images[: a.limit_validation]
    identity = dict(
        config=config,
        taxonomy=LABELS,
        data=str(a.data.resolve()),
        arrow_index_sha256=sha256(a.arrow_index),
        validation_ids_sha256=sha256(a.validation_ids) if a.validation_ids else None,
        limit_validation=a.limit_validation,
    )
    model = build()
    if a.resume:
        previous = torch.load(a.resume, map_location="cpu")
        if previous["identity"] != identity:
            raise ValueError("Resume configuration/data identity changed")
        model.load_state_dict(previous["state_dict"], strict=True)
    else:
        path = a.initial_checkpoint or a.initial_124
        if a.initial_checkpoint:
            state = torch.load(path, map_location="cpu")
            if state["labels"] != LABELS:
                raise ValueError("Expected the original eight-class taxonomy")
            model.load_state_dict(state["state_dict"], strict=True)
            initialization = dict(kind="strict8", parent_step=state["step"])
            del state
        else:
            initialization = initialize_coarse(model, path)
        initialization.update(checkpoint_sha256=sha256(path), checkpoint=str(path.resolve()))
        write_json(a.output / "initialization.json", initialization)
        write_json(a.output / "config.json", identity)
    model.cuda()
    groups = [[], [], []]
    for name, parameter in model.named_parameters():
        index = 0 if name.startswith("backbone.") else 2 if name in CLASSIFIERS else 1
        groups[index].append(parameter)
    optimizer = torch.optim.SGD(
        [dict(params=ps, lr=lr) for ps, lr in zip(groups, config["base_lr"])],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    start, best = 0, -math.inf
    if a.resume:
        optimizer.load_state_dict(previous["optimizer"])
        start, best = previous["step"], previous["best"]
        torch.set_rng_state(previous["rng"])
        torch.cuda.set_rng_state_all(previous["cuda_rng"])
        random.setstate(previous["python_rng"])
        np.random.set_state(previous["numpy_rng"])
        del previous
    sampler = CoarseOrder(
        data,
        config["sample_offset"] + config["steps"] * config["accumulation"],
        config["sample_offset"] + start * config["accumulation"],
        config["seed"],
    )
    # A private generator prevents loader creation on resume from consuming the
    # model RNG. Augmentations already derive their seeds from sampler indices.
    loader = DataLoader(
        data,
        batch_size=1,
        sampler=sampler,
        num_workers=config["workers"],
        persistent_workers=config["workers"] > 0,
        pin_memory=True,
        generator=torch.Generator().manual_seed(config["seed"]),
    )
    iterator = iter(loader)
    weights = torch.tensor(config["loss_weights"], device="cuda", dtype=torch.float32)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))

    def checkpoint(step, path, full=True):
        payload = dict(
            state_dict=model.state_dict(),
            labels=LABELS,
            config=config,
            identity=identity,
            step=step,
            best=best,
        )
        if full:
            payload.update(
                optimizer=optimizer.state_dict(),
                rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),
                numpy_rng=np.random.get_state(),
            )
        save(payload, path)

    stats = a.output / "training.csv"
    if not stats.exists():
        stats.write_text("step,loss,gradient_norm,peak_gpu_gb\n")
    for step in range(start, config["steps"]):
        model.train()
        freeze_bn_statistics(model)
        schedule = (1 - step / config["steps"]) ** 0.9 * min(1.0, (step + 1) / max(1, config["warmup"]))
        for group, lr in zip(optimizer.param_groups, config["base_lr"]):
            group["lr"] = lr * schedule
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for _ in range(config["accumulation"]):
            image, target = next(iterator)
            image, target = image.cuda(non_blocking=True), target.cuda(non_blocking=True)
            if not (target != 255).any():
                raise ValueError("All training pixels are ignored")
            with autocast("cuda", config["precision"]):
                main, aux = forward(model, image)
                loss = (
                    F.cross_entropy(main.float(), target, weight=weights, ignore_index=255)
                    + config["aux_weight"]
                    * F.cross_entropy(aux.float(), target, weight=weights, ignore_index=255)
                ) / config["accumulation"]
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at update {step + 1}")
            loss.backward()
            loss_sum += loss.item()
            del image, target, main, aux, loss
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
        if not torch.isfinite(norm) or norm <= 0:
            raise RuntimeError(f"Invalid gradient at update {step + 1}")
        optimizer.step()
        completed = step + 1
        with stats.open("a") as stream:
            csv.writer(stream).writerow(
                [completed, loss_sum, norm.item(), torch.cuda.max_memory_allocated() / 1e9]
            )
        logging.info("step=%d/%d loss=%.5f grad=%.3f", completed, config["steps"], loss_sum, norm.item())
        if completed % config["validate_every"] == 0 or completed == config["steps"]:
            optimizer.zero_grad(set_to_none=True)
            result = evaluate(
                model,
                val,
                a.output / "validation",
                completed,
                config["validation_precision"],
                config["workers"],
            )
            if result["lane_filter_miou"] > best:
                best = result["lane_filter_miou"]
                checkpoint(completed, a.output / "best.pth", full=False)
            checkpoint(completed, a.output / f"checkpoint_{completed:06}.pth", full=False)
        stop = stopped[0] or (a.stop_after is not None and completed >= a.stop_after)
        if (
            completed % config["save_every"] == 0
            or completed % config["validate_every"] == 0
            or stop
            or completed == config["steps"]
        ):
            checkpoint(completed, a.output / "latest.pth")
        if stop:
            logging.info("Stopped safely at update %d; resume using latest.pth", completed)
            return
    if not a.skip_final_validation:
        optimizer.zero_grad(set_to_none=True)
        selected = torch.load(a.output / "best.pth", map_location="cpu")
        model.load_state_dict(selected["state_dict"], strict=True)
        selected_step = selected["step"]
        del selected
        result = evaluate(
            model,
            validation(a.data),
            a.output / "full_validation",
            selected_step,
            config["validation_precision"],
            config["workers"],
        )
        write_json(a.output / "selected_result.json", result)
    write_json(
        a.output / "finished.json",
        dict(
            updates=config["steps"],
            full_validation=not a.skip_final_validation,
            limit_validation=a.limit_validation,
        ),
    )


if __name__ == "__main__":
    main()
