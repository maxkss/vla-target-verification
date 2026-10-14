#!/usr/bin/env python3
"""Candidates followed through each rollout (paper section 2.1): SAM 2.1 video predictor over the policy frames
of every call, starting from the first-frame candidates. Forward only: the masks at call t use calls 1..t, as
they would during the rollout. Runs in the SAM environment.

  track.py RUN [--worker i/n]   -> <rollout>/tracks.npz (masks: call, candidate, 224, 224)
"""

import argparse
import tempfile

import numpy as np

import common
from segment import load_sam


def track(predictor, frames, masks):
    import torch
    from PIL import Image

    tracked = np.zeros((len(frames), *masks.shape), bool)
    tracked[0] = masks
    if not len(masks) or len(frames) < 2:
        return tracked
    with tempfile.TemporaryDirectory() as d:
        for k, f in enumerate(frames):
            Image.fromarray(f).save(f"{d}/{k:05d}.jpg", quality=100)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            state = predictor.init_state(video_path=d)
            for k, m in enumerate(masks):
                predictor.add_new_mask(state, frame_idx=0, obj_id=k, mask=m)
            for frame, ids, logits in predictor.propagate_in_video(state):
                if frame > 0:
                    tracked[frame, list(ids)] = (logits[:, 0] > 0).cpu().numpy()
    return tracked


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    from PIL import Image

    run = common.Run(a.run)
    started = common.now()
    todo = [it for it in run.items(a.worker) if not (run.rollout_dir(it) / "tracks.npz").exists()]
    predictor = load_sam(common.CONFIG["candidates"], video=True) if todo else None
    for item in todo:
        d, scene = run.rollout_dir(item), run.scene_dir(item["task"], item["init"])
        with np.load(scene / "candidates.npz") as z:
            masks = z["masks"]
        with np.load(d / "calls.npz") as z:
            frames = z["image"]
        if not np.array_equal(frames[0], np.asarray(Image.open(scene / "frame.png").convert("RGB"))):
            raise SystemExit(f"{common.key(item)}: first call frame differs from the scene's (candidates' frame)")
        common.save_npz(d / "tracks.npz", masks=track(predictor, frames, masks))
        print(f"{common.key(item)}: {len(frames)} calls x {len(masks)} candidates", flush=True)
    i, n = a.worker.split("/")
    run.record_stage(f"track.w{i}of{n}", started, rollouts=len(todo))


if __name__ == "__main__":
    main()
