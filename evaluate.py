#!/usr/bin/env python3
"""The paper's tables for a run (section 4). Policy environment, CPU.

Positive = a failed sentence rollout (original-instruction rollouts are references, not scored). An alarm on a
failed rollout is a detection, on a successful one a false alarm.

Failure type of a failed sentence rollout, against the original instruction's rollout on the same scene:
  type undetermined             the original failed there or handled nothing: no reference object
  nothing handled               the hand handled no object
  right object, not completed   it handled the original's object
  wrong target → task object    it handled another object of the task (bddl obj_of_interest, e.g. the destination)
  wrong target → other object   it handled an object outside the task
Before contact: the alarm step precedes the onset of contact with the handled object; lead = onset - alarm step.

Two views: every sentence rollout, and only those whose scene has a reference object. 95 % intervals resample
scenes (task x init state) with all their rollouts, 2,000 times.

  evaluate.py RUN   -> results/{summary.json, summary.md, rows.json, summary.png}
"""

import argparse
from collections import Counter

import numpy as np

import common

TYPES = ["wrong target → other object", "wrong target → task object", "right object, not completed",
         "nothing handled", "type undetermined"]
FILES = ["episode.json", "calls.npz", "trajectory.npz", "label.json", "task_objects.json", "tracks.npz", "alarm.json"]
HZ = 20   # LIBERO control rate: one env step = 0.05 s


def load(run):
    missing, rows = [], []
    for item in run.plan:
        d = run.rollout_dir(item)
        lost = [f for f in FILES if not (d / f).exists()]
        if lost:
            missing.append(f"{common.key(item)}: {', '.join(lost)}")
            continue
        episode, label = common.read_json(d / "episode.json"), common.read_json(d / "label.json")
        objects, alarm = common.read_json(d / "task_objects.json"), common.read_json(d / "alarm.json")
        rows.append({"key": common.key(item), "task": item["task"], "sentence": item["sentence"], "init": item["init"],
                     "original": item["sentence"] == common.ORIGINAL, "success": episode["success"],
                     "steps": episode["steps"], "handled": label["index"], "handled_object": label["object"],
                     "onset": label["onset_step"], "label_reason": label["reason"], "names": label["object_names"],
                     "vlm": objects["status"], "alarm": alarm["alarm_step"]})
    if missing:
        raise SystemExit(f"{len(missing)} rollouts incomplete, e.g.\n  " + "\n  ".join(missing[:20]))
    return rows


def classify(rows):
    reference = {(r["task"], r["init"]): r["handled"] if r["success"] else None for r in rows if r["original"]}
    for r in rows:
        truth = reference.get((r["task"], r["init"]))
        r["has_reference"] = truth is not None
        if r["original"] or r["success"]:
            r["failure"] = None
        elif truth is None:
            r["failure"] = TYPES[4]
        elif r["handled"] is None:
            r["failure"] = TYPES[3]
        elif r["handled"] == truth:
            r["failure"] = TYPES[2]
        elif r["handled"] in common.task_objects(r["task"], r["names"]):
            r["failure"] = TYPES[1]
        else:
            r["failure"] = TYPES[0]
    return rows


def metrics(rows):
    y = np.array([not r["success"] for r in rows])
    a = np.array([r["alarm"] is not None for r in rows])
    tp, fp = int((a & y).sum()), int((a & ~y).sum())
    pos, neg = int(y.sum()), int((~y).sum())
    precision, recall = tp / max(tp + fp, 1), tp / max(pos, 1)
    caught = [r["alarm"] for r in rows if not r["success"] and r["alarm"] is not None]
    out = {"n": len(rows), "failures": pos, "successes": neg, "TP": tp, "FN": pos - tp, "FP": fp, "TN": neg - fp,
           "precision": precision, "recall": recall, "F1": 2 * precision * recall / max(precision + recall, 1e-12),
           "false_alarm_rate": fp / max(neg, 1), "success_rate": neg / max(len(rows), 1),
           "alarm_step_median": float(np.median(caught)) if caught else None, "by_type": {}}
    for t in TYPES:
        sub = [r for r in rows if r["failure"] == t]
        hit = [r for r in sub if r["alarm"] is not None]
        before = [r for r in hit if r["onset"] is not None and r["alarm"] < r["onset"]]
        lead = [r["onset"] - r["alarm"] for r in before]
        out["by_type"][t] = {"n": len(sub), "caught": len(hit), "before_contact": len(before),
                             "lead_median": float(np.median(lead)) if lead else None}
    return out


def intervals(rows, n=2000, seed=0):
    """Scene bootstrap: recall, false-alarm rate, F1, and the share of wrong target → other object failures
    caught before contact."""
    scenes = {}
    for i, r in enumerate(rows):
        scenes.setdefault((r["task"], r["init"]), []).append(i)
    groups = list(scenes.values())
    y = np.array([not r["success"] for r in rows])
    a = np.array([r["alarm"] is not None for r in rows])
    other = np.array([r["failure"] == TYPES[0] for r in rows])
    before = np.array([r["alarm"] is not None and r["onset"] is not None and r["alarm"] < r["onset"] for r in rows])
    rng = np.random.default_rng(seed)
    draws = {"recall": [], "false_alarm_rate": [], "F1": [], "other_object_before_contact": []}
    for _ in range(n):
        idx = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        yy, aa = y[idx], a[idx]
        tp, fp = (aa & yy).sum(), (aa & ~yy).sum()
        p, rc = tp / max(tp + fp, 1), tp / max(yy.sum(), 1)
        draws["recall"].append(rc)
        draws["false_alarm_rate"].append(fp / max((~yy).sum(), 1))
        draws["F1"].append(2 * p * rc / max(p + rc, 1e-12))
        if other[idx].any():
            draws["other_object_before_contact"].append(before[idx][other[idx]].mean())
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None for k, v in draws.items()}


def by_task(rows):
    out = {}
    for task in common.TASKS:
        s = [r for r in rows if r["task"] == task and r["success"]]
        f = [r for r in rows if r["task"] == task and not r["success"]]
        if s or f:
            out[task] = {"successes": len(s), "false_alarms": sum(r["alarm"] is not None for r in s),
                         "failures": len(f), "caught": sum(r["alarm"] is not None for r in f)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    run = common.Run(a.run)
    started = common.now()
    rows = classify(load(run))
    sentences = [r for r in rows if not r["original"]]
    views = {"all sentences": sentences,
             "sentences with a reference object": [r for r in sentences if r["has_reference"]]}
    stages = sorted((run.path / "stages").glob("*.json"))
    summary = {
        "run": run.path.name, "experiment": run.manifest["experiment"], "head": common.read_json(run.path / "head.json"),
        "views": {name: {**metrics(v), "CI95": intervals(v)} for name, v in views.items() if v},
        "by_task": by_task(sentences),
        "originals": {"n": sum(r["original"] for r in rows), "success": sum(r["original"] and r["success"] for r in rows),
                      "scenes_without_reference": sorted({f"{r['task']}/init{r['init']:02d}" for r in sentences
                                                          if not r["has_reference"]})},
        "task_object_status": dict(Counter(r["vlm"] for r in rows)),
        "label_reasons": dict(Counter(r["label_reason"] for r in rows)),
        "stage_code_hashes": sorted({common.read_json(s)["code_hash"] for s in stages}),
    }
    out = run.path / "results"
    common.write_json(out / "summary.json", summary)
    common.write_json(out / "rows.json", [{k: v for k, v in r.items() if k != "names"} for r in rows])
    (out / "summary.md").write_text(markdown(summary))
    figure(out / "summary.png", summary)
    run.record_stage("evaluate", started, rollouts=len(rows))
    print((out / "summary.md").read_text(), flush=True)


def fmt(x, digits=3):
    return "—" if x is None else f"{x:.{digits}f}" if isinstance(x, float) else str(x)


def markdown(s):
    lines = [f"# {s['experiment']} — {s['run']}", "",
             f"Readout head: layer {s['head']['layer']}, head {s['head']['head']}. One env step = {1 / HZ:.2f} s.", ""]
    for name, m in s["views"].items():
        ci = m["CI95"]
        lines += [f"## Table 1 — {name} (n = {m['n']}, failures {m['failures']}, successes {m['successes']})", "",
                  "| actual / alarm | alarm | no alarm | total |", "|---|---|---|---|",
                  f"| failure | {m['TP']} (TP) | {m['FN']} (FN) | {m['failures']} |",
                  f"| success | {m['FP']} (FP) | {m['TN']} (TN) | {m['successes']} |", "",
                  f"precision {m['precision']:.3f} · recall {m['recall']:.3f} [{fmt(ci['recall'][0])}, {fmt(ci['recall'][1])}]"
                  f" · F1 {m['F1']:.3f} [{fmt(ci['F1'][0])}, {fmt(ci['F1'][1])}]"
                  f" · false-alarm rate {100 * m['false_alarm_rate']:.1f}% [{100 * ci['false_alarm_rate'][0]:.1f}, "
                  f"{100 * ci['false_alarm_rate'][1]:.1f}] · alarm step median {fmt(m['alarm_step_median'], 1)}"
                  f" · success rate {100 * m['success_rate']:.1f}%", "",
                  f"## Table 2 — {name}", "", "| failure type | n | caught | caught before contact | median lead (steps) |",
                  "|---|---|---|---|---|"]
        for t, v in m["by_type"].items():
            if v["n"]:
                lines.append(f"| {t} | {v['n']} | {v['caught']} | {v['before_contact']} | {fmt(v['lead_median'], 1)} |")
        o = ci.get("other_object_before_contact")
        if o:
            lines.append(f"\nwrong target → other object caught before contact, 95% CI [{o[0]:.3f}, {o[1]:.3f}]")
        lines.append("")
    lines += ["## False alarms by task (all sentences)", "", "| task | successes | false alarms | rate | failures | caught |",
              "|---|---|---|---|---|---|"]
    for t, v in sorted(s["by_task"].items(), key=lambda kv: -kv[1]["false_alarms"] / max(kv[1]["successes"], 1)):
        rate = 100 * v["false_alarms"] / max(v["successes"], 1)
        lines.append(f"| {common.instruction(t)} | {v['successes']} | {v['false_alarms']} | {rate:.1f}% | {v['failures']} | {v['caught']} |")
    o = s["originals"]
    lines += ["", f"Original-instruction rollouts: {o['success']}/{o['n']} succeeded; scenes without a reference object: "
              f"{len(o['scenes_without_reference'])}.", f"Task-object replies: {s['task_object_status']}.",
              f"Labels: {s['label_reasons']}.", f"Stage code hashes: {', '.join(s['stage_code_hashes'])}.", ""]
    return "\n".join(lines)


def figure(path, s):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    m = s["views"]["all sentences"]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axs = plt.subplots(1, 3, figsize=(19, 5.6), gridspec_kw={"width_ratios": [1, 1.6, 1.9]})
    ax = axs[0]
    cm = np.array([[m["TP"], m["FN"]], [m["FP"], m["TN"]]])
    ax.imshow(cm / cm.sum(1, keepdims=True).clip(1), cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=16)
    ax.set_xticks([0, 1], ["alarm", "no alarm"])
    ax.set_yticks([0, 1], ["failure", "success"])
    ax.set_title(f"Alarms (n = {m['n']:,})\nrecall {m['recall']:.3f} · false alarms {100 * m['false_alarm_rate']:.1f}%",
                 fontsize=11)
    ax = axs[1]
    types = [t for t in TYPES if m["by_type"][t]["n"]]
    x = np.arange(len(types))
    for k, (field, colour) in enumerate((("n", "#c8c8c8"), ("caught", "#2a78d6"), ("before_contact", "#eb6834"))):
        ax.bar(x + (k - 1) * 0.26, [m["by_type"][t][field] for t in types], 0.26, color=colour,
               label={"n": "failures", "caught": "caught", "before_contact": "caught before contact"}[field])
    ax.set_xticks(x, [t.replace(" → ", "\n→ ").replace(", ", ",\n") for t in types], fontsize=9)
    ax.legend(frameon=False)
    ax.set_title("Failure types", fontsize=11)
    ax = axs[2]
    tasks = sorted(s["by_task"], key=lambda t: s["by_task"][t]["false_alarms"] / max(s["by_task"][t]["successes"], 1))
    rates = [100 * s["by_task"][t]["false_alarms"] / max(s["by_task"][t]["successes"], 1) for t in tasks]
    ax.barh(range(len(tasks)), rates, color="#eb6834")
    ax.set_yticks(range(len(tasks)), [common.instruction(t) for t in tasks], fontsize=9)
    ax.set_xlabel("false-alarm rate (% of successes)")
    ax.set_title("False alarms by task", fontsize=11)
    fig.suptitle(f"{s['experiment']} — {s['run']}", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
