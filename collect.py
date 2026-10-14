#!/usr/bin/env python3
"""Rollouts of a run's plan, recording what every later stage needs (paper section 3). Policy environment.

Every rollout settles its init state, then runs its instruction until the task succeeds or `max_steps` pass.
  per call (every `actions_per_call` env steps): the action queries' attention to the agent view at the last
      denoising step for every layer and head, each head's total attention to the images, the policy frame
  per env step: the action, which objects the hand touches, every object's position and joint coordinates
Once per scene (task x init state): the first frame, robot and simulator object masks, and the first frame
re-rendered at 896 px for the VLM.

Checks (a failure stops the worker): the capture leaves the first call's action unchanged (policy.py); every
rollout's first frame equals its scene's; the 896-px render, downsampled, lies within 2 grey levels of it.

  collect.py RUN [--worker i/n]   -> <rollout>/{calls.npz, trajectory.npz, episode.json}, <task>/scenes/init<NN>/
"""

import argparse
import itertools
import time

import numpy as np

import common

HIRES_TOLERANCE = 2.0   # mean absolute grey levels between the 896-px render (downsampled) and the policy frame


def save_scene(run, task, init, scene, render_env, initial):
    """Write the scene once; afterwards every rollout of it must start from the identical frame."""
    from PIL import Image

    import policy as P

    d = run.scene_dir(task, init)
    if (d / "frame.png").exists():
        if not np.array_equal(np.asarray(Image.open(d / "frame.png").convert("RGB")), scene["frame"]):
            raise SystemExit(f"{task}/init{init:02d}: first frame differs from the scene's")
        return
    size = common.CONFIG["rollout"]["image_size"] * common.CONFIG["task_objects"]["scale"]
    hires = P.settle(render_env, run.seed, initial)["agentview_image"][::-1, ::-1]
    small = np.asarray(Image.fromarray(np.ascontiguousarray(hires)).resize(scene["frame"].shape[:2][::-1],
                                                                            Image.BILINEAR), float)
    diff = float(np.abs(small - scene["frame"].astype(float)).mean())
    if hires.shape[0] != size or diff > HIRES_TOLERANCE:
        raise SystemExit(f"{task}/init{init:02d}: {hires.shape[0]}-px render differs from the policy frame ({diff:.2f})")
    common.save_npz(d / "masks.npz", robot=scene["robot"], objects=scene["objects"], names=np.array(scene["names"]))
    common.save_png(d / "frame_896.png", np.ascontiguousarray(hires))
    common.save_png(d / "frame.png", scene["frame"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    common.setup_libero()
    import policy as P

    run = common.Run(a.run)
    started = common.now()
    todo = [it for it in run.items(a.worker) if not (run.rollout_dir(it) / "episode.json").exists()]
    todo.sort(key=lambda it: common.TASKS.index(it["task"]))
    pol = P.Policy(common.env_path("OPENPI_CHECKPOINT")) if todo else None
    size = common.CONFIG["rollout"]["image_size"] * common.CONFIG["task_objects"]["scale"]
    gate = 0.0
    for task, items in itertools.groupby(todo, key=lambda it: it["task"]):
        env, render_env = P.make_env(task), P.make_render_env(task, size)
        states = common.init_states(task)
        try:
            for item in items:
                t0 = time.time()
                scene, record = P.rollout(env, pol, item, states[item["init"]], run.seed)
                save_scene(run, task, item["init"], scene, render_env, states[item["init"]])
                d = run.rollout_dir(item)
                common.save_npz(d / "calls.npz", **record["calls"])
                common.save_npz(d / "trajectory.npz", **record["trajectory"])
                common.write_json(d / "episode.json", {**record["episode"], "seconds": round(time.time() - t0, 1)})
                gate = max(gate, record["episode"]["gate_max_abs"])
                print(f"{common.key(item)}: success={record['episode']['success']} steps={record['episode']['steps']} "
                      f"{time.time() - t0:.0f}s", flush=True)
        finally:
            env.close()
            render_env.close()
    i, n = a.worker.split("/")
    run.record_stage(f"collect.w{i}of{n}", started, rollouts=len(todo), gate_max_abs=gate)


if __name__ == "__main__":
    main()
