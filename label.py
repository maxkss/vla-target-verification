#!/usr/bin/env python3
"""Which object the hand handled first, and when contact with it began. Scoring only: no detector input.

From each rollout's own per-step record (trajectory.npz, written while the rollout ran): the first object the
hand stays in contact with for `dwell_steps` consecutive steps AND that actually changes, i.e. its position or
one of its joints (drawer slide, knob hinge) moves by more than `min_change` (metres / radians) between just
before the contact and `settle_steps` after the last contact. Objects only brushed on the way do not count.
Two objects qualifying at the same step leave the rollout without a handled object.

  label.py RUN    -> <rollout>/label.json for every rollout of the run
"""

import argparse
import sys

import numpy as np

import common


def object_change(positions, joints, joint_owner):
    """(steps, objects): each object's largest coordinate change since the first recorded step."""
    out = np.abs(positions - positions[0]).max(-1)
    if joints.shape[1]:
        d = np.abs(joints - joints[0])
        for i in np.unique(joint_owner):
            out[:, i] = np.maximum(out[:, i], d[:, joint_owner == i].max(-1))
    return out


def first_handled(contacts, change, dwell, min_change, settle):
    """(object index or None, contact onset or None, reason, per-object summary)."""
    steps, n = contacts.shape
    summary, qualified = [], []
    for i in range(n):
        streak, onset = 0, None
        for t in range(steps):
            streak = streak + 1 if contacts[t, i] else 0
            if streak == dwell:
                onset = t - dwell + 1
                break
        if onset is None:
            continue
        touched = np.flatnonzero(contacts[:, i])
        end = min(steps, touched[-1] + settle + 1)
        before = change[max(onset - 1, 0), i]
        delta = float(np.abs(change[onset:end, i] - before).max())
        summary.append({"index": i, "onset": onset, "steps_in_contact": int(len(touched)), "change": delta})
        if delta > min_change:
            qualified.append((onset, i))
    if not summary:
        return None, None, "no stable contact", summary
    if not qualified:
        return None, None, "contact without change", summary
    qualified.sort()
    if len(qualified) > 1 and qualified[0][0] == qualified[1][0]:
        return None, qualified[0][0], "two objects at once", summary
    onset, i = qualified[0]
    return i, onset, "handled", summary


def label(trajectory, cfg=None):
    cfg = cfg or common.CONFIG["label"]
    names = [str(n) for n in trajectory["names"]]
    change = object_change(trajectory["positions"], trajectory["joints"], trajectory["joint_owner"])
    index, onset, reason, summary = first_handled(trajectory["contacts"], change, cfg["dwell_steps"],
                                                  cfg["min_change"], cfg["settle_steps"])
    return {"object": names[index] if index is not None else None, "index": index,
            "onset_step": onset if index is not None else None, "reason": reason, "object_names": names,
            "contacts": [{**s, "object": names[s["index"]]} for s in summary]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    run = common.Run(a.run)
    started = common.now()
    done = 0
    for item in run.plan:
        d = run.rollout_dir(item)
        if (d / "label.json").exists():
            continue
        if not (d / "episode.json").exists():
            sys.exit(f"{common.key(item)}: not collected")
        with np.load(d / "trajectory.npz") as z:
            common.write_json(d / "label.json", label(dict(z)))
        done += 1
    run.record_stage("label", started, labelled=done)
    print(f"label: {done} rollouts labelled", flush=True)


if __name__ == "__main__":
    main()
