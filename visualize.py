#!/usr/bin/env python3
"""One rollout at a glance (material for figures 1 and 2): the policy frame at chosen calls with the readout
candidate boxed, the readout head's attention at those calls (plain heatmaps), and every call's verdict with
the alarm and the onset of contact.

  visualize.py RUN TASK/SENTENCE/initNN [--calls 0,4,8]   -> <run>/figures/<task>__<sentence>__initNN.png
"""

import argparse

import numpy as np

import common
import detector

COLOUR = {detector.ON: "#2a78d6", detector.OFF: "#eb6834", detector.ABSTAIN: "#c8c8c8"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("key", help="task/sentence/initNN")
    p.add_argument("--calls", help="comma-separated call indices (default: spread over the rollout)")
    a = p.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    run = common.Run(a.run)
    task, sentence, init = a.key.split("/")
    d = run.path / a.key
    episode, label = common.read_json(d / "episode.json"), common.read_json(d / "label.json")
    alarm, objects = common.read_json(d / "alarm.json"), common.read_json(d / "task_objects.json")
    with np.load(d / "calls.npz") as z:
        steps, frames, attention = z["env_step"], z["image"], z["attention"][:, alarm["head"][0], alarm["head"][1]]
    with np.load(d / "tracks.npz") as z:
        masks = z["masks"]
    n = len(steps)
    if a.calls:
        calls = [int(c) for c in a.calls.split(",")]
    else:
        calls = sorted(set(np.linspace(0, n - 1, min(n, 5)).astype(int)) | ({alarm["alarm_call"]} if alarm.get("alarm_call") is not None else set()))
    verdicts = alarm.get("verdicts") or [detector.ABSTAIN] * n
    shares = np.array(alarm.get("shares") or np.zeros((n, len(masks[0]))))

    plt.rcParams.update({"font.size": 11})
    fig = plt.figure(figsize=(3.1 * len(calls), 7.4))
    grid = fig.add_gridspec(3, len(calls), height_ratios=[1, 1, 0.42])
    for col, c in enumerate(calls):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(frames[c])
        k = int(np.argmax(shares[c])) if shares.size and shares[c].max() > 0 else None
        if k is not None and masks[c, k].any():
            ys, xs = np.nonzero(masks[c, k])
            ax.add_patch(Rectangle((xs.min(), ys.min()), xs.max() - xs.min(), ys.max() - ys.min(), fill=False,
                                   lw=2.5, ec=COLOUR[verdicts[c]]))
        ax.set_title(f"call {c + 1} · step {steps[c]}\n{verdicts[c]}", fontsize=10, color=COLOUR[verdicts[c]])
        ax.axis("off")
        ax = fig.add_subplot(grid[1, col])
        ax.imshow(attention[c].astype(np.float32), cmap="inferno")
        ax.axis("off")
    ax = fig.add_subplot(grid[2, :])
    for c, v in enumerate(verdicts):
        ax.add_patch(Rectangle((c, 0), 1, 1, color=COLOUR[v]))
    if alarm.get("alarm_call") is not None:
        ax.axvline(alarm["alarm_call"] + 1, color="black", lw=2.5)
        ax.text(alarm["alarm_call"] + 1, 1.08, f"alarm (step {alarm['alarm_step']})", ha="center", fontsize=10)
    if label["onset_step"] is not None:
        onset = label["onset_step"] / common.CONFIG["rollout"]["actions_per_call"]
        ax.axvline(onset, color="black", lw=1.5, ls="--")
        ax.text(onset, -0.5, f"contact: {label['object']} (step {label['onset_step']})", ha="center", fontsize=10)
    ax.set_xlim(0, n)
    ax.set_ylim(-0.8, 1.4)
    ax.set_yticks([])
    ax.set_xlabel("policy call")
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.suptitle(f"{common.instruction(task)} · init {init[4:]} · \"{episode['instruction']}\" · "
                 f"{'success' if episode['success'] else 'failure'} · task objects {objects.get('objects')}",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    out = run.path / "figures" / (a.key.replace("/", "__") + ".png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()
