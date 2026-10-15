# Early Detection of Task Failures in VLA Models via Attention-Based Target Verification

대상 검증을 통한 VLA의 내부 어텐션 기반 작업 실패 조기 탐지

A frozen VLA policy (π0.5, LIBERO checkpoint) is watched while it acts. At every policy call, one attention
head of the action expert tells which object the policy is attending to; an external VLM has named, on the first
frame, the objects the task involves. Repeated attention on an object outside the task raises an alarm, often
before the gripper touches the wrong object. The policy's weights and actions are never changed.

## Method → code

| paper | step | code |
|---|---|---|
| 2.1 | object candidates: SAM 2.1 masks on the first frame, minus background and the robot body | `segment.py` |
| 2.1 | task objects: Qwen3.5-9B picks the candidates the gripper will act on and the destination | `task_objects.py` |
| 2.1 | candidates followed through the rollout (SAM 2.1 video, forward only) | `track.py` |
| 2.2 | action queries → image attention at the last denoising step, read-only | `policy.py`, `env/openpi_attention_capture.patch` |
| 2.2 | readout head: largest attention share on the handled object, LIBERO-Goal originals, init 0 of the 10 tasks | `select_head.py` |
| 2.2 | readout: top-8 patches, each patch to the candidate covering most of it (≥ 10 %), shares (eq. 1), verdict | `detector.py` |
| 2.3 | alarm: off-target calls counted over any other object, reset by an on-target call, kept through abstained calls, N = 5 | `detector.py`, `alarm.py` |
| 3 | rollouts, contact labels | `collect.py`, `label.py` |
| 4 | tables | `evaluate.py` |

Settings the paper describes in words only are fixed in `config.json`: a call abstains when its top-8 patches hold
less than 0.4 of the map ("weak attention"); the map is the average over the action queries at the last
(10th) denoising step, agent view; the task-object set also takes in candidates lying mostly inside the chosen
ones or covering most of them; candidate filters; the VLM prompt and image format.

## Setup

- Three Python 3.11 environments, versions in `env/`: policy (JAX, openpi, LIBERO), SAM 2.1 (package commit in
  `config.json`), VLM (transformers).
- openpi at the commit in `config.json` with `env/openpi_attention_capture.patch` applied (`git apply`).
- LIBERO-Para at the commit in `config.json`; the π0.5 LIBERO checkpoint (`pi05_libero`); SAM 2.1 hiera-small
  and Qwen3.5-9B at the revisions in `config.json`.
- Copy `env/local.sh.example` to `env/local.sh` and set the paths. A run checks the openpi commit and patch, the
  LIBERO-Para data and the checkpoint fingerprint, and stops if any differs.

## Run

```bash
./run.sh experiments/libero_para.json          # the paper's experiment: 4,092 sentences + 500 original rollouts
./run.sh experiments/comp50.json               # test set: 50 sentences per task on one scene each
./run.sh experiments/example.json              # figure 1: one sentence
./run.sh experiments/comp50.json --only list   # part of an experiment (task/sentence/initNN globs, one per line)
./run.sh <run folder>                          # continue an interrupted run
./run.sh <run folder> --from alarm             # redo a stage and every later one
GPUS=0 ./run.sh experiments/comp50.json        # restrict to some GPUs (2 policy processes of ~9 GB per GPU)
python compare.py <run A> <run B> <out>        # two runs, rollout by rollout and table by table
python visualize.py <run> <task/sentence/initNN>
```

Stages: `collect` → `label` → `select_head` → `segment` → `task_objects` → `track` → `alarm` → `evaluate`.
Results: `<run>/results/summary.md` (tables 1 and 2, false alarms by task), `summary.json`, `rows.json`, `summary.png`.

## Protocol

- Scenes: LIBERO-Goal task × init state (0–49). The original instruction runs on every scene a run uses; init 0
  of every task selects the head. LIBERO-Para sentences of a task go round-robin, in file-name order, over init
  1–49, whatever the outcomes, so a sentence always runs on the same scene.
- Same scene, same noise: the policy noise of a call comes from (seed, task, init, call) only.
- A call executes 5 of the 10 predicted actions (20 Hz control: one env step = 0.05 s); a rollout ends at
  success or after 300 steps.
- Scoring only (never an input of the detector): success, the first object the hand handled (≥ 10 steps of
  contact and a change > 0.02 m or rad), its contact onset, and the original rollout's handled object on the same
  scene as the reference for the failure type.

## Consistency

- XLA autotuning is off and XLA ops are deterministic (`run.sh`). With autotuning on, kernel choice varies between
  processes (6 processes fell into 2 groups of different actions); with it off, 6 processes on two GPUs gave
  bit-identical calls.
- A rollout depends only on (seed, task, init, sentence): not on the number of workers, the GPU or the order.
  Repeating 322 rollouts of the runs below in other processes, 12 of them in a fresh clone with environments
  installed by `env/setup.sh`, gave 317 bit-identical rollouts and the same success, handled object and alarm
  for all 322. The 5 others diverged at one call while the robot, the objects and the agent view were identical;
  in the case traced, a single wrist-camera pixel differed by one grey level. This is rare and its trigger is not
  known.
- `run.json` records the code and settings hashes, the external-data fingerprints, the machine and the
  determinism flags; every stage records the code hash it ran with.

## Results of an independent run

Runs of this repository at tag `runs-2026-10-14` on 2 × RTX A6000 (LIBERO-Para on one GPU: collection 14.7 h,
task objects 2.9 h, tracking 6.6 h). Sentence rollouts only; positive = failed rollout.

| LIBERO-Para (4,092 sentences) | paper | this run |
|---|---|---|
| success rate | 72.2 % | 70.8 % |
| TP / FN / FP / TN | 940 / 197 / 428 / 2,527 | 971 / 223 / 407 / 2,491 |
| precision · recall · F1 | 0.687 · 0.827 · 0.750 | 0.705 · 0.813 · 0.755 |
| false-alarm rate | 14.5 % | 14.0 % |
| alarm step, median | 30 (1.5 s) | 30 |
| wrong target → other object: n / caught / before contact | 497 / 476 / 395 | 523 / 499 / 423 |
| wrong target → task object | 168 / 128 / 61 | 169 / 127 / 63 |
| right object, not completed | 431 / 313 / 71 | 442 / 323 / 98 |
| nothing handled | 20 / 9 / – | 28 / 8 / – |
| type undetermined | 21 / 14 / 8 | 32 / 14 / 5 |

Criterion fixed before the run (sentences on scenes with a reference object, scene-bootstrap 95 % intervals): the
paper's recall 0.827, false-alarm rate 14.5 % and share of wrong target → other object failures caught before
contact 79.5 % all lie inside this run's intervals (0.824 [0.796, 0.849], 14.1 % [12.6, 15.6], 80.9 % [76.9, 84.4]).
False alarms concentrate in the same four tasks (put the bowl on the stove 30.5 %, put the cream cheese in the bowl
26.6 %, put the bowl on top of the cabinet 21.1 %, open the top drawer and put the bowl inside 20.8 %; the other six
2.1–8.5 %).

comp50 (test set, 500 sentences): success rate 83.2 %, TP / FN / FP / TN 72 / 12 / 11 / 405, F1 0.862,
false-alarm rate 2.6 %, alarm step median 70.

## Differences from the run behind the paper

| | paper's run | this pipeline |
|---|---|---|
| scenes of the sentences | 380 held out from development (9–12 sentences each) | init 1–49 by rule (7–9 sentences each) |
| reference rollouts | original instruction re-run in a second pass | one original rollout per scene in the same run |
| attention after the 5th call | stored actions replayed | recorded during the rollout |
| contact labels | stored actions replayed in the simulator | the rollout's own contact record (same definition) |
| XLA autotuning | on | off |

On identical inputs the stages reproduce the earlier outputs exactly (first frames, candidates, VLM replies,
tracks, labels, alarm steps); the policy's numerics differ by about 1e-3 because of the autotuning setting, so
single rollouts can end differently while the tables agree.
