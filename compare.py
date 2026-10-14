#!/usr/bin/env python3
"""Two runs side by side: repeat runs, a sample against the full run, two machines. Rollouts are matched by
task/sentence/init and compared stage by stage, from the first frame to the alarm; the runs' tables are put
next to each other. Any object exposing RunReader's methods (e.g. a reader for results kept outside this
repository) can be passed to compare() from Python.

  compare.py RUN_A RUN_B OUT   -> OUT/{report.md, rows.json, stages.png}
"""

import argparse
from pathlib import Path

import numpy as np

import common


class RunReader:
    """Read access to one run of this pipeline. Methods return None where a stage has no output."""

    def __init__(self, path):
        self.run = common.Run(path)
        self.name = self.run.path.name

    def keys(self):
        return [common.key(it) for it in self.run.plan]

    def _json(self, path):
        return common.read_json(path) if path.exists() else None

    def frame(self, task, init):
        from PIL import Image

        p = self.run.scene_dir(task, init) / "frame.png"
        return np.asarray(Image.open(p).convert("RGB")) if p.exists() else None

    def candidates(self, task, init):
        p = self.run.scene_dir(task, init) / "candidates.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return z["masks"]

    def head(self):
        h = self._json(self.run.path / "head.json")
        return (h["layer"], h["head"]) if h else None

    def episode(self, key):
        return self._json(self.run.path / key / "episode.json")

    def actions(self, key):
        with np.load(self.run.path / key / "trajectory.npz") as z:
            return z["actions"]

    def attention(self, key, head):
        with np.load(self.run.path / key / "calls.npz") as z:
            return z["attention"][:, head[0], head[1]].astype(np.float32)

    def label(self, key):
        return self._json(self.run.path / key / "label.json")

    def task_objects(self, key):
        return self._json(self.run.path / key / "task_objects.json")

    def tracks(self, key):
        p = self.run.path / key / "tracks.npz"
        if not p.exists():
            return None
        with np.load(p) as z:
            return z["masks"]

    def alarm(self, key):
        return self._json(self.run.path / key / "alarm.json")

    def summary(self):
        return self._json(self.run.path / "results" / "summary.json")


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 1.0


def compare(a, b, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k in a.keys() if k in set(b.keys())]
    head_a, head_b = a.head(), b.head()
    rows, scenes = [], {}
    for key in keys:
        task, sentence, init = key.split("/")
        init = int(init[4:])
        if (task, init) not in scenes:
            fa, fb, ca, cb = a.frame(task, init), b.frame(task, init), a.candidates(task, init), b.candidates(task, init)
            scenes[(task, init)] = {
                "frame_equal": None if fa is None or fb is None else bool(np.array_equal(fa, fb)),
                "candidates_equal": None if ca is None or cb is None else bool(ca.shape == cb.shape and np.array_equal(ca, cb)),
                "candidates": None if ca is None or cb is None else [len(ca), len(cb)],
                "candidate_iou": None if ca is None or cb is None or ca.shape != cb.shape
                else float(np.mean([iou(x, y) for x, y in zip(ca, cb)])) if len(ca) else 1.0}
        ea, eb = a.episode(key), b.episode(key)
        if ea is None or eb is None:
            continue
        r = {"key": key, "scene": f"{task}/init{init:02d}", "success": [ea["success"], eb["success"]],
             "steps": [ea["steps"], eb["steps"]]}
        xa, xb = a.actions(key), b.actions(key)
        m = min(len(xa), len(xb))
        diverge = np.flatnonzero(np.abs(xa[:m] - xb[:m]).max(-1) > 0)
        r["actions_identical"] = bool(len(xa) == len(xb) and not len(diverge))
        r["first_divergent_step"] = int(diverge[0]) if len(diverge) else None
        r["first_call_attention_max_abs"] = (float(np.abs(a.attention(key, head_a)[0] - b.attention(key, head_b)[0]).max())
                                             if head_a and head_a == head_b else None)
        la, lb = a.label(key), b.label(key)
        if la and lb:
            r["handled"] = [la["object"], lb["object"]]
            r["onset"] = [la["onset_step"], lb["onset_step"]]
        oa, ob = a.task_objects(key), b.task_objects(key)
        if oa and ob:
            r["task_objects"] = [oa["task_objects"], ob["task_objects"]]
            r["task_objects_equal"] = oa["task_objects"] == ob["task_objects"] and oa["status"] == ob["status"]
        ta, tb = a.tracks(key), b.tracks(key)
        if ta is not None and tb is not None and r["actions_identical"] and ta.shape == tb.shape:
            r["tracks_equal"] = bool(np.array_equal(ta, tb))
            r["tracks_iou"] = float(np.mean([iou(x, y) for x, y in zip(ta.reshape(-1, *ta.shape[-2:]),
                                                                       tb.reshape(-1, *tb.shape[-2:]))])) if ta.size else 1.0
        aa, ab = a.alarm(key), b.alarm(key)
        if aa and ab:
            r["alarm_step"] = [aa["alarm_step"], ab["alarm_step"]]
            va, vb = aa.get("verdicts") or [], ab.get("verdicts") or []
            n = min(len(va), len(vb))
            r["verdict_agreement"] = float(np.mean([x == y for x, y in zip(va[:n], vb[:n])])) if n else None
        rows.append(r)
    stages = stage_table(rows, scenes, head_a, head_b)
    common.write_json(out / "rows.json", {"scenes": {f"{t}/init{i:02d}": v for (t, i), v in scenes.items()}, "rollouts": rows})
    (out / "report.md").write_text(report(a, b, rows, stages))
    figure(out / "stages.png", stages, a.name, b.name)
    return stages


def stage_table(rows, scenes, head_a, head_b):
    def frac(values):
        values = [v for v in values if v is not None]
        return (sum(values), len(values))

    same = [r for r in rows if r["actions_identical"]]
    return [
        ("first frame identical", frac(s["frame_equal"] for s in scenes.values())),
        ("candidates identical", frac(s["candidates_equal"] for s in scenes.values())),
        ("readout head identical", (int(head_a == head_b), 1)),
        ("task objects identical", frac(r.get("task_objects_equal") for r in rows)),
        ("first-call attention identical", frac(None if r["first_call_attention_max_abs"] is None
                                                else r["first_call_attention_max_abs"] == 0 for r in rows)),
        ("actions identical (whole rollout)", frac(r["actions_identical"] for r in rows)),
        ("success agrees", frac(r["success"][0] == r["success"][1] for r in rows)),
        ("handled object agrees", frac(r["handled"][0] == r["handled"][1] if "handled" in r else None for r in rows)),
        ("tracks identical (identical actions)", frac(r.get("tracks_equal") for r in same)),
        ("alarm step agrees (identical actions)", frac(r["alarm_step"][0] == r["alarm_step"][1] if "alarm_step" in r else None
                                                       for r in same)),
        ("alarm fired agrees (all)", frac((r["alarm_step"][0] is None) == (r["alarm_step"][1] is None)
                                          if "alarm_step" in r else None for r in rows)),
    ]


def report(a, b, rows, stages):
    lines = [f"# {a.name} vs {b.name}", "", f"Rollouts compared: {len(rows)}", "",
             "| stage | agree | compared |", "|---|---|---|"]
    lines += [f"| {name} | {k} | {n} |" for name, (k, n) in stages]
    diverged = [r for r in rows if not r["actions_identical"]]
    lines += ["", f"Rollouts whose actions differ: {len(diverged)}"]
    for r in diverged[:30]:
        lines.append(f"- {r['key']}: first divergent step {r['first_divergent_step']}, success {r['success']}, "
                     f"handled {r.get('handled')}, alarm {r.get('alarm_step')}")
    sa, sb = a.summary(), b.summary()
    if sa and sb:
        lines += ["", "## Tables", "", f"| view · metric | {a.name} | {b.name} |", "|---|---|---|"]
        for view in sa["views"]:
            if view not in sb["views"]:
                continue
            ma, mb = sa["views"][view], sb["views"][view]
            for k in ("n", "failures", "TP", "FN", "FP", "TN", "precision", "recall", "F1", "false_alarm_rate",
                      "alarm_step_median"):
                lines.append(f"| {view} · {k} | {ma[k]} | {mb[k]} |")
            for t, v in ma["by_type"].items():
                w = mb["by_type"].get(t, {})
                lines.append(f"| {view} · {t} (n / caught / before contact) | {v['n']} / {v['caught']} / "
                             f"{v['before_contact']} | {w.get('n')} / {w.get('caught')} / {w.get('before_contact')} |")
    return "\n".join(lines) + "\n"


def figure(path, stages, name_a, name_b):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [s[0] for s in stages]
    share = [100 * k / n if n else np.nan for _, (k, n) in stages]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(stages) + 1.5))
    ax.barh(range(len(names)), share, color="#2a78d6")
    for i, (_, (k, n)) in enumerate(stages):
        ax.text(min(share[i] if n else 0, 100) + 1, i, f"{k}/{n}", va="center", fontsize=10)
    ax.set_yticks(range(len(names)), names)
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("agreement (%)")
    ax.set_title(f"{name_a} vs {name_b}", fontsize=11)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_a")
    p.add_argument("run_b")
    p.add_argument("out")
    a = p.parse_args()
    for name, (k, n) in compare(RunReader(a.run_a), RunReader(a.run_b), a.out):
        print(f"{name:42s} {k}/{n}")


if __name__ == "__main__":
    main()
