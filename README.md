# Early Detection of Task Failures in VLA Models via Attention-Based Target Verification

대상 검증을 통한 VLA의 내부 어텐션 기반 작업 실패 조기 탐지

## Overview

This repository implements a non-intrusive failure detector for Vision-Language-Action (VLA) policies. While a frozen VLA policy ($\pi_{0.5}$, `pi05_libero` checkpoint) executes a manipulation task, we monitor its internal attention without modifying the policy's weights or actions:

1. **Task Object Identification (First Frame):** SAM 2.1 segments candidate objects in the initial scene (excluding the background and robot body), and an external VLM (Qwen3.5-9B) identifies which objects are relevant to the task instruction (the target object to manipulate and its destination). SAM 2.1 then tracks all candidate masks forward through the rollout.
2. **Attention Readout (Every Policy Call):** At the final denoising step of each policy call, we extract the action expert's image attention from a single pre-selected attention head and map the top-attended image patches to the tracked object masks.
3. **Early Alarm:** If the policy repeatedly attends to an object outside the task-relevant set ($N = 5$ off-target calls), the detector raises an alarm—frequently catching failures *before* the gripper makes physical contact with the wrong object.

---

## Method to Code Mapping

| Paper Section | Pipeline Step | Implementation |
|---|---|---|
| **2.1** | **Object Candidates:** Generate first-frame segmentation masks with SAM 2.1, filtering out the background and robot body. | `segment.py` |
| **2.1** | **Task Objects:** Prompt Qwen3.5-9B on the first frame to identify the target object to be grasped and the destination object. | `task_objects.py` |
| **2.1** | **Object Tracking:** Track all candidate masks forward through the rollout video using SAM 2.1. | `track.py` |
| **2.2** | **Attention Extraction:** Capture read-only action-query-to-image attention weights at the final (10th) denoising step. | `policy.py`, `env/openpi_attention_capture.patch` |
| **2.2** | **Head Selection:** Select the readout attention head with the highest attention share on the manipulated object using `init 0` of the 10 `LIBERO-Goal` tasks. | `select_head.py` |
| **2.2** | **Target Verification:** Take the top-8 attended patches, assign each patch to the candidate mask covering most of its area ($\ge 10\%$), compute object attention shares (Eq. 1), and issue a call verdict. | `detector.py` |
| **2.3** | **Alarm Rule:** Increment an off-target counter when attention falls on a non-task object, reset on an on-target call, carry the count across abstained calls, and trigger an alarm at $N = 5$. | `detector.py`, `alarm.py` |
| **3** | **Data Collection & Labeling:** Execute policy rollouts and extract ground-truth physical contact labels from simulator states. | `collect.py`, `label.py` |
| **4** | **Evaluation:** Compute detection metrics, failure breakdowns, and summary tables. | `evaluate.py` |

### Configuration Details (`config.json`)
Hyperparameters and heuristics described in prose in the paper are explicitly defined in `config.json`:
- **Attention Map:** Averaged across all action queries at the final (10th) denoising step on the agent-view camera image.
- **Abstention ("Weak Attention"):** A policy call abstains if its top-8 patches capture less than `0.4` of the total attention map.
- **Task-Object Expansion:** The valid task-object set automatically includes candidate masks that are mostly contained within, or mostly cover, the VLM-selected objects.
- **Prompts & Filters:** candidate filters, the exact VLM prompt, and image formatting parameters are locked in `config.json`.

---

## Setup

1. **Python Environments:** Set up three separate Python 3.11 environments (specifications in `env/`):
   - **Policy environment:** JAX, `openpi`, and `LIBERO`
   - **Tracker environment:** SAM 2.1 (pinned to the commit in `config.json`)
   - **VLM environment:** Hugging Face `transformers`
2. **Patch `openpi`:** Check out `openpi` at the commit specified in `config.json` and apply the attention capture patch:
   ```bash
   git apply env/openpi_attention_capture.patch
   ```
3. **Models & Datasets:** Download the required checkpoints and datasets at the exact revisions specified in `config.json`:
   - `LIBERO-Para` dataset
   - $\pi_{0.5}$ LIBERO checkpoint (`pi05_libero`)
   - SAM 2.1 (`hiera-small`)
   - `Qwen3.5-9B`
4. **Configure Paths:** Copy `env/local.sh.example` to `env/local.sh` and set your local directory paths. At startup, the pipeline verifies the `openpi` commit and patch status, dataset integrity, and checkpoint fingerprints, aborting immediately if any mismatch is detected.

---

## Running Experiments

```bash
# Run the main paper experiment (4,092 paraphrased instructions + 500 original rollouts)
./run.sh experiments/libero_para.json

# Run the 500-sentence test set (50 sentences per task, one scene each)
./run.sh experiments/comp50.json

# Run the single-sentence example from Figure 1
./run.sh experiments/example.json

# Run a subset of rollouts matching glob patterns (task/sentence/initNN, one per line)
./run.sh experiments/comp50.json --only list

# Resume an interrupted run
./run.sh <run_folder>

# Re-run from a specific stage onward
./run.sh <run_folder> --from alarmground-truth physical contact labels from simulator states

# Restrict execution to specific GPUs (launches 2 policy processes of ~9 GB each per GPU)
GPUS=0 ./run.sh experiments/comp50.json

# Compare two runs rollout-by-rollout and table-by-table
python compare.py <run_A> <run_B> <out_dir>

# Visualize attention, tracking
python visualize.py <run_folder> <task/sentence/initNN>
```

- **Pipeline Stages:** `collect` → `label` → `select_head` → `segment` → `task_objects` → `track` → `alarm` → `evaluate`
- **Outputs:** Saved under `<run_folder>/results/`, including `summary.md` (Tables 1 & 2 and per-task false-alarm rates), `summary.json`, `rows.json`, and `summary.png`.
