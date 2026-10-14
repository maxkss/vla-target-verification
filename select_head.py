#!/usr/bin/env python3
"""The readout head (paper section 2.2). On the original instruction's rollout of every task's head-selection
scene, take the first call's attention at the last denoising step: for each (layer, head), the share of its
image attention (all cameras) that falls on the object the hand then handled (simulator mask, first frame).
The head with the largest share averaged over the tasks is the readout head. A task whose original rollout
failed or handled nothing is left out. Labels are used here and in scoring only.

  select_head.py RUN   -> head.json, head.png
"""

import argparse

import numpy as np

import common


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    run = common.Run(a.run)
    started = common.now()
    init = common.CONFIG["head_selection"]["init"]
    shares, used, left_out = [], [], {}
    for task in common.TASKS:
        item = {"task": task, "sentence": common.ORIGINAL, "init": init}
        d, scene = run.rollout_dir(item), run.scene_dir(task, init)
        episode, label = common.read_json(d / "episode.json"), common.read_json(d / "label.json")
        if not episode["success"] or label["index"] is None:
            left_out[task] = "original failed" if not episode["success"] else "nothing handled"
            continue
        with np.load(d / "calls.npz") as z:
            attention, total = z["attention"][0].astype(np.float32), z["image_attention"][0]
        with np.load(scene / "masks.npz") as z:
            mask = z["objects"][label["index"]]
        layers, heads = total.shape
        on_object = attention.reshape(layers, heads, -1) @ common.patch_coverage(mask)
        shares.append(on_object / total)
        used.append(task)
    if not shares:
        raise SystemExit("no task usable for head selection")
    score = np.mean(shares, axis=0)
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    layer, head = map(int, order[0])
    per_task = {t: int(np.flatnonzero((np.dstack(np.unravel_index(np.argsort(-s, axis=None), s.shape))[0]
                                       == (layer, head)).all(1))[0]) + 1 for t, s in zip(used, shares)}
    common.write_json(run.path / "head.json", {
        "layer": layer, "head": head, "init": init, "tasks": used, "left_out": left_out,
        "top10": [[int(l), int(h), round(float(score[l, h]), 5)] for l, h in order[:10]],
        "rank_per_task": per_task, "score": score.round(6).tolist()})
    figure(run.path / "head.png", score, layer, head, len(used))
    run.record_stage("select_head", started, layer=layer, head=head, tasks=len(used))
    print(f"head: layer {layer}, head {head} (mean share {score[layer, head]:.4f} over {len(used)} tasks)", flush=True)


def figure(path, score, layer, head, n):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(score, cmap="viridis", aspect="auto")
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(score.shape[1]))
    ax.set_yticks(range(score.shape[0]))
    fig.colorbar(im, ax=ax, label="mean share of image attention on the handled object")
    ax.set_title(f"Head selection ({n} tasks): layer {layer}, head {head}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
