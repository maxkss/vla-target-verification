#!/usr/bin/env python3
"""Object candidates of every scene (paper section 2.1): SAM 2.1 automatic masks on the first frame, minus
background and the robot. Runs in the SAM environment; never sees the instruction.

A mask is kept when its area lies within [min_area, max_area] of the frame, it touches at most `max_borders`
image borders (two or more: table, wall), it is not a near-duplicate (IoU > max_iou) of a larger kept mask, and
less than `robot_share` of it lies on the robot body. Larger masks come first (ties by box): this order is the
candidate numbering everywhere downstream.

  segment.py RUN [--worker i/n]   -> <task>/scenes/init<NN>/candidates.npz
"""

import argparse

import numpy as np

import common


def load_sam(cfg, video=False):
    import torch
    from huggingface_hub import hf_hub_download
    from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES, build_sam2, build_sam2_video_predictor

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config, filename = HF_MODEL_ID_TO_FILENAMES[cfg["model"]]
    checkpoint = hf_hub_download(cfg["model"], filename, revision=cfg["revision"])
    if video:
        return build_sam2_video_predictor(config, checkpoint, device="cuda")
    model = build_sam2(config, checkpoint, device="cuda")
    return model.eval().requires_grad_(False)


def box_of(mask):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def borders(mask):
    return int(mask[0].any()) + int(mask[-1].any()) + int(mask[:, 0].any()) + int(mask[:, -1].any())


def keep(masks, robot, cfg):
    kept = []
    for m in sorted(masks, key=lambda m: (-int(m.sum()), box_of(m))):
        area = m.sum() / m.size
        if area < cfg["min_area"] or area > cfg["max_area"] or borders(m) > cfg["max_borders"]:
            continue
        if any(np.logical_and(m, n).sum() / max(np.logical_or(m, n).sum(), 1) > cfg["max_iou"] for n in kept):
            continue
        kept.append(m)
    on_robot = [(m & robot).sum() > cfg["robot_share"] * max(m.sum(), 1) for m in kept]
    kept = [m for m, r in zip(kept, on_robot) if not r]
    return np.array(kept, bool).reshape(-1, *robot.shape)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    import torch
    from PIL import Image
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    run = common.Run(a.run)
    cfg = common.CONFIG["candidates"]
    started = common.now()
    i, n = map(int, a.worker.split("/"))
    todo = [s for s in run.scenes()[i::n] if not (run.scene_dir(*s) / "candidates.npz").exists()]
    generator = SAM2AutomaticMaskGenerator(load_sam(cfg), points_per_side=cfg["points_per_side"]) if todo else None
    for task, init in todo:
        d = run.scene_dir(task, init)
        image = np.asarray(Image.open(d / "frame.png").convert("RGB"))
        with np.load(d / "masks.npz") as z:
            robot = z["robot"]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            raw = [x["segmentation"].astype(bool) for x in generator.generate(image)]
        masks = keep(raw, robot, cfg)
        common.save_npz(d / "candidates.npz", masks=masks, raw=len(raw))
        print(f"{task}/init{init:02d}: {len(raw)} masks -> {len(masks)} candidates", flush=True)
    run.record_stage(f"segment.w{i}of{n}", started, scenes=len(todo))


if __name__ == "__main__":
    main()
