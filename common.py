"""Shared pieces: frozen settings, paths, LIBERO, experiment plans, run folders and their records.

Every stage reads and writes one run folder:

    <VTV_RUNS>/<experiment>_seed<seed>_<YYYYmmdd_HHMMSS>/
      run.json    settings, code and external-data fingerprints, machine, determinism flags
      plan.json   every rollout of the run: task, sentence id, instruction, init state
      stages/     one record per finished stage
      head.json   the readout head (select_head.py)
      results/    tables and figures (evaluate.py)
      <task>/scenes/init<NN>/          first frame and everything derived from it alone
      <task>/<sentence id>/init<NN>/   one rollout ("original" = the task's own instruction)
"""

import fnmatch
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text())
ORIGINAL = "original"
# LIBERO-Goal, in benchmark order (LIBERO-Para files name a task by this index).
TASKS = [
    "open_the_middle_drawer_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_bowl_on_the_stove",
    "put_the_bowl_on_top_of_the_cabinet",
    "put_the_cream_cheese_in_the_bowl",
    "put_the_wine_bottle_on_the_rack",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "turn_on_the_stove",
]


def env_path(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"{name} is not set (see env/local.sh.example)")
    return Path(value)


def instruction(task):
    """A LIBERO task name is its instruction with underscores."""
    return task.replace("_", " ")


# ---------------------------------------------------------------- files

def sha(data, n=16):
    return hashlib.sha256(data).hexdigest()[:n]


def config_hash():
    return sha(json.dumps(CONFIG, sort_keys=True).encode())


def code_hash():
    """Every file that can change a result: code, settings, experiment definitions, the openpi patch."""
    h = hashlib.sha256()
    for pattern in ("*.py", "*.sh", "config.json", "experiments/*.json", "env/*.patch"):
        for p in sorted(ROOT.glob(pattern)):
            h.update(p.relative_to(ROOT).as_posix().encode() + b"\0" + p.read_bytes())
    return h.hexdigest()[:16]


def read_json(path):
    return json.loads(Path(path).read_text())


def _atomic(path, write):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=path.suffix)
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def write_json(path, obj):
    _atomic(path, lambda tmp: Path(tmp).write_text(json.dumps(obj, indent=1, default=_plain) + "\n"))


def save_npz(path, **arrays):
    _atomic(path, lambda tmp: np.savez_compressed(tmp, **arrays))


def save_png(path, image):
    from PIL import Image

    _atomic(path, lambda tmp: Image.fromarray(np.asarray(image)).save(tmp, format="PNG"))


def _plain(x):
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x))


# ---------------------------------------------------------------- LIBERO

def setup_libero():
    """Make LIBERO-Para's `libero` importable without its interactive first-run prompt."""
    import yaml

    root = env_path("LIBERO_PARA_ROOT")
    base = root / "libero/libero"
    cfg = env_path("VTV_RUNS") / ".libero"
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / "config.yaml").write_text(yaml.safe_dump({
        "benchmark_root": str(base), "bddl_files": str(base / "bddl_files"),
        "init_states": str(base / "init_files"), "assets": str(base / "assets"),
        "datasets": str(root / "libero/datasets")}))
    os.environ["LIBERO_CONFIG_PATH"] = str(cfg)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def bddl(task):
    return env_path("LIBERO_PARA_ROOT") / "libero/libero/bddl_files/libero_goal" / f"{task}.bddl"


def init_states(task):
    import torch

    path = env_path("LIBERO_PARA_ROOT") / "libero/libero/init_files/libero_goal" / f"{task}.pruned_init"
    return torch.load(path, weights_only=False)


def libero_para_sentences(task):
    """[(sentence id, instruction)] of one task, in file-name order."""
    n = TASKS.index(task)
    folder = env_path("LIBERO_PARA_ROOT") / "libero/libero/bddl_files/libero_para"
    out = []
    for f in sorted(folder.glob(f"*_eval{n}_ver*.bddl"), key=lambda f: f.name):
        line = next(ln for ln in f.read_text().splitlines() if ":language" in ln)
        out.append((f.stem.replace(f"_eval{n}", ""), line.split(":language", 1)[1].strip().rstrip(")").strip()))
    return out


def task_objects(task, names):
    """Indices of the task's bddl obj_of_interest among the scene objects (scoring only). A region maps to
    the object it is named after (flat_stove_1_cook_region -> flat_stove_1), a table region to the object
    its name refers to (main_table_stove_front_region -> flat_stove_1)."""
    text = bddl(task).read_text()
    out = set()
    for item in re.search(r"\(:obj_of_interest(.*?)\)", text, re.S).group(1).split():
        hit = [i for i, n in enumerate(names) if item == n or item.startswith(n + "_")]
        if not hit:
            hit = [i for i, n in enumerate(names) if f"_{n.rsplit('_', 1)[0].split('_')[-1]}_" in f"_{item}_"]
        assert len(hit) == 1, (task, item, hit)
        out |= set(hit)
    return sorted(out)


def noise(seed, task, init, call, horizon=10, dim=32):
    """Policy noise of one call, from (seed, task, init state, call) only: every sentence run on a scene
    draws the same noise as the original instruction on it."""
    key = hashlib.sha256(f"{int(seed)}|{task}:{int(init)}|{int(call)}".encode()).hexdigest()
    return np.random.default_rng(int(key[:16], 16)).standard_normal((horizon, dim)).astype(np.float32)


def patch_coverage(masks, grid=16):
    """(..., H, W) bool -> (..., grid * grid): share of each image patch a mask covers."""
    m = np.asarray(masks, np.float32)
    h, w = m.shape[-2:]
    m = m.reshape(*m.shape[:-2], grid, h // grid, grid, w // grid).mean((-1, -3))
    return m.reshape(*m.shape[:-2], grid * grid)


# ---------------------------------------------------------------- plans

def key(item):
    return f"{item['task']}/{item['sentence']}/init{item['init']:02d}"


def make_plan(experiment, only=None):
    """Every rollout of an experiment. A sentence runs on one scene (task x init state); the original
    instruction runs on the head-selection scene of every task and on every scene a sentence uses."""
    src = experiment["sentences"]
    sentences = []
    for task in TASKS:
        if src["source"] == "libero_para":
            first, last = src["inits"]
            inits = list(range(first, last + 1))
            sentences += [{"task": task, "sentence": sid, "text": text, "init": inits[k % len(inits)]}
                          for k, (sid, text) in enumerate(libero_para_sentences(task))]
        elif task in src["tasks"]:
            entry = src["tasks"][task]
            sentences += [{"task": task, "sentence": s["id"], "text": s["text"], "init": entry["init"]}
                          for s in entry["sentences"]]
    if only is not None:
        sentences = [s for s in sentences if any(fnmatch.fnmatchcase(key(s), p) for p in only)]
        if not sentences:
            raise SystemExit("--only matched no sentence")
    head = CONFIG["head_selection"]["init"]
    scenes = sorted({(TASKS.index(t), i) for t, i in [(t, head) for t in TASKS] + [(s["task"], s["init"]) for s in sentences]})
    originals = [{"task": TASKS[t], "sentence": ORIGINAL, "text": instruction(TASKS[t]), "init": i} for t, i in scenes]
    return originals + sentences


# ---------------------------------------------------------------- runs

def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class Run:
    def __init__(self, path):
        self.path = Path(path)
        self.manifest = read_json(self.path / "run.json")
        self.plan = read_json(self.path / "plan.json")
        if self.manifest["config_hash"] != config_hash():
            raise SystemExit(f"{self.path.name}: config.json changed since the run was created — start a new run")

    @classmethod
    def create(cls, experiment_file, only=None):
        experiment = read_json(experiment_file)
        plan = make_plan(experiment, only)
        path = env_path("VTV_RUNS") / f"{experiment['name']}_seed{experiment['seed']}_{stamp()}"
        path.mkdir(parents=True)
        write_json(path / "plan.json", plan)
        write_json(path / "run.json", {
            "experiment": experiment["name"], "experiment_file": str(experiment_file), "seed": experiment["seed"],
            "only": only, "created": datetime.now().isoformat(timespec="seconds"),
            "rollouts": len(plan), "sentences": sum(p["sentence"] != ORIGINAL for p in plan),
            "plan_hash": sha(json.dumps(plan, sort_keys=True).encode()),
            "config_hash": config_hash(), "code_hash": code_hash(), "git": git_state(),
            "external": external_fingerprints(), "machine": machine(), "determinism": determinism_flags(),
            "config": CONFIG})
        return cls(path)

    @property
    def seed(self):
        return self.manifest["seed"]

    def items(self, worker="0/1"):
        i, n = map(int, worker.split("/"))
        return self.plan[i::n]

    def rollout_dir(self, item):
        return self.path / key(item)

    def scene_dir(self, task, init):
        return self.path / task / "scenes" / f"init{init:02d}"

    def scenes(self):
        return sorted({(p["task"], p["init"]) for p in self.plan}, key=lambda s: (TASKS.index(s[0]), s[1]))

    def record_stage(self, stage, started, **info):
        write_json(self.path / "stages" / f"{stage}.json", {
            "stage": stage, "started": started, "finished": datetime.now().isoformat(timespec="seconds"),
            "code_hash": code_hash(), "config_hash": config_hash(), "python": sys.executable,
            "packages": packages(), **info})


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------- provenance

def git_state():
    def git(*args):
        r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def external_fingerprints():
    """openpi checkout + patch, LIBERO-Para data, policy checkpoint: refuse anything but the pinned ones."""
    pin = CONFIG["pinned"]
    openpi = env_path("OPENPI_ROOT")
    head = subprocess.run(["git", "-C", str(openpi), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    diff = subprocess.run(["git", "-C", str(openpi), "diff"], capture_output=True, text=True).stdout
    patch = (ROOT / pin["openpi_patch"]).read_text()
    libero = env_path("LIBERO_PARA_ROOT")
    lp_head = subprocess.run(["git", "-C", str(libero), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    lp_dirty = subprocess.run(["git", "-C", str(libero), "status", "--porcelain", "--",
                               "libero/libero/bddl_files", "libero/libero/init_files"],
                              capture_output=True, text=True).stdout.strip()
    ck = checkpoint_fingerprint(env_path("OPENPI_CHECKPOINT"))
    problems = []
    if head != pin["openpi_commit"]:
        problems.append(f"openpi at {head}, pinned {pin['openpi_commit']}")
    if diff != patch:
        problems.append("openpi working tree differs from the capture patch")
    if lp_head != pin["libero_para_commit"] or lp_dirty:
        problems.append(f"LIBERO-Para at {lp_head} (data modified: {bool(lp_dirty)}), pinned {pin['libero_para_commit']}")
    if ck != pin["checkpoint_fingerprint"]:
        problems.append(f"checkpoint fingerprint {ck}, pinned {pin['checkpoint_fingerprint']}")
    if problems:
        raise SystemExit("external data not as pinned:\n  " + "\n  ".join(problems))
    return {"openpi_commit": head, "openpi_patch_hash": sha(patch.encode()), "libero_para_commit": lp_head,
            "checkpoint": str(env_path("OPENPI_CHECKPOINT")), "checkpoint_fingerprint": ck}


def checkpoint_fingerprint(path):
    """(relative path, size) of every file, plus the content of files under 1 MB."""
    h = hashlib.sha256()
    for p in sorted(q for q in Path(path).rglob("*") if q.is_file()):
        size = p.stat().st_size
        h.update(f"{p.relative_to(path)}\t{size}\n".encode())
        if size < 1_000_000:
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


def machine():
    gpus = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.strip().splitlines()
    return {"host": platform.node(), "platform": platform.platform(), "gpus": gpus}


def determinism_flags():
    names = ("XLA_FLAGS", "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED", "MUJOCO_GL")
    return {k: os.environ.get(k) for k in names}


def packages():
    from importlib import metadata

    out = {"python": platform.python_version()}
    for name in ("jax", "jaxlib", "flax", "numpy", "torch", "mujoco", "robosuite", "sam-2", "transformers",
                 "opencv-python", "pillow"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return out
