#!/usr/bin/env python3
"""Per-call verdicts and the alarm of every rollout (paper sections 2.2-2.3), with the run's readout head.
Inputs are the rollout's own attention, tracked candidates and task objects only. Policy environment, CPU.

  alarm.py RUN [--worker i/n]   -> <rollout>/alarm.json
"""

import argparse

import numpy as np

import common
import detector


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    run = common.Run(a.run)
    started = common.now()
    head = common.read_json(run.path / "head.json")
    layer, h = head["layer"], head["head"]
    todo = [it for it in run.items(a.worker) if not (run.rollout_dir(it) / "alarm.json").exists()]
    for item in todo:
        d = run.rollout_dir(item)
        objects = common.read_json(d / "task_objects.json")["task_objects"]
        with np.load(d / "calls.npz") as z:
            steps, attention = z["env_step"], z["attention"][:, layer, h].astype(np.float32)
        record = {"head": [layer, h], "calls": int(len(steps))}
        if objects is None:   # the VLM gave no task objects: no verdict, no alarm
            record.update(status="no task objects", alarm_call=None, alarm_step=None)
        else:
            with np.load(d / "tracks.npz") as z:
                masks = z["masks"]
            r = detector.run(attention, masks, objects)
            fired = r["alarm_call"]
            record.update(status="ok", alarm_call=fired, alarm_step=int(steps[fired]) if fired is not None else None,
                          max_count=max(r["counts"], default=0), verdicts=r["verdicts"],
                          off_target=r["off_target"], counts=r["counts"],
                          top_k_mass=[round(m, 4) for m in r["top_k_mass"]],
                          shares=np.round(r["shares"], 4).tolist())
        common.write_json(d / "alarm.json", record)
    i, n = a.worker.split("/")
    run.record_stage(f"alarm.w{i}of{n}", started, rollouts=len(todo), head=[layer, h])
    print(f"alarm: {len(todo)} rollouts", flush=True)


if __name__ == "__main__":
    main()
