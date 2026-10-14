"""π0.5 (LIBERO checkpoint) in process: action chunks with read-only attention capture, and one rollout.

The action of every call comes from the capture pass itself: the model's own 10 denoising steps
(`denoise_prepare` / `denoise_step`, added by env/openpi_attention_capture.patch), with the attention of the
action queries to the image keys taken from the last step. The first call of every rollout is also run
through the unmodified `policy.infer`; the two chunks must agree (gate) or the rollout stops.

Import after common.setup_libero() and after the GPU / MuJoCo environment variables are set.
"""

from collections import deque

import jax
import jax.numpy as jnp
import numpy as np

import common

GRID = 16  # SigLIP patches per side at 224 px
HAND = ("gripper0_", "robot0_right_hand")  # body-name prefixes of the robot hand


class Policy:
    def __init__(self, checkpoint):
        from openpi.models.model import IMAGE_KEYS, Observation
        from openpi.policies.policy_config import create_trained_policy
        from openpi.shared.nnx_utils import module_jit
        from openpi.training.config import get_config

        self.policy = create_trained_policy(get_config("pi05_libero"), str(checkpoint))
        model = self.policy._model
        self.Observation = Observation
        self.cameras = len(IMAGE_KEYS)
        self.keys = jnp.arange(self.cameras * GRID * GRID, dtype=jnp.int32)
        self.steps = common.CONFIG["rollout"]["denoising_steps"]
        self.prepare = module_jit(model.denoise_prepare)
        self.step = module_jit(model.denoise_step, static_argnames=("num_steps",))

    def act(self, inputs, noise):
        """(actions (10, 7), agent-view attention (layer, head, 16, 16), image attention (layer, head)).

        Attention is the post-softmax weight of the action queries (averaged) at the last denoising step:
        the agent-view part per patch, and the total over every camera's image keys.
        """
        x = self.policy._input_transform(jax.tree.map(lambda v: v, inputs))
        x = jax.tree.map(lambda v: jnp.asarray(v)[np.newaxis, ...], x)
        state = self.prepare(self.Observation.from_dict(x), jnp.asarray(noise)[np.newaxis, ...])
        maps = None
        for k in range(self.steps):
            state, maps = self.step(state, self.keys if k == self.steps - 1 else None, num_steps=self.steps)
        out = self.policy._output_transform({"state": np.asarray(x["state"][0]), "actions": np.asarray(state["x"][0])})
        m = np.asarray(maps[:, 0], np.float32).mean(axis=2)          # layer, head, image key
        m = m.reshape(*m.shape[:2], self.cameras, GRID, GRID)
        return out["actions"], m[:, :, 0], m.sum(axis=(2, 3, 4))

    def reference(self, inputs, noise):
        """The unmodified openpi path."""
        return self.policy.infer(inputs, noise=noise)["actions"]


# ---------------------------------------------------------------- environment

def make_env(task, size=None):
    from libero.libero.envs.env_wrapper import SegmentationRenderEnv

    size = size or common.CONFIG["rollout"]["camera_size"]
    return SegmentationRenderEnv(bddl_file_name=str(common.bddl(task)), camera_heights=size, camera_widths=size)


def make_render_env(task, size):
    from libero.libero.envs.env_wrapper import OffScreenRenderEnv

    return OffScreenRenderEnv(bddl_file_name=str(common.bddl(task)), camera_heights=size, camera_widths=size)


def settle(env, seed, initial):
    """The scene the first policy call sees: the initial state after a few gripper-open steps."""
    env.seed(seed)
    env.reset()
    obs = env.set_init_state(initial)
    for _ in range(common.CONFIG["rollout"]["settle_steps"]):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs


def observation(obs, instruction):
    """(agent-view frame, policy inputs). Frames are rotated 180° as in openpi's LIBERO example."""
    from openpi_client import image_tools

    size = common.CONFIG["rollout"]["image_size"]
    frame = obs["agentview_image"][::-1, ::-1]
    wrist = obs["robot0_eye_in_hand_image"][::-1, ::-1]
    q = np.asarray(obs["robot0_eef_quat"]).copy()
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] ** 2)
    axis = np.zeros(3) if den < 1e-8 else q[:3] * (2 * np.arccos(q[3])) / den
    return frame, {
        "observation/image": image_tools.resize_with_pad(frame, size, size),
        "observation/wrist_image": image_tools.resize_with_pad(wrist, size, size),
        "observation/state": np.concatenate([obs["robot0_eef_pos"], axis, obs["robot0_gripper_qpos"]]),
        "prompt": instruction,
    }


def to_policy_frame(mask):
    """A camera-resolution mask in the policy's 224-px frame (rotated 180°, nearest-neighbour)."""
    import cv2

    size = common.CONFIG["rollout"]["image_size"]
    m = np.asarray(mask, np.uint8)[::-1, ::-1]
    assert m.shape[0] == m.shape[1], m.shape
    return cv2.resize(np.ascontiguousarray(m), (size, size), interpolation=cv2.INTER_NEAREST).astype(bool)


def scene_masks(env, obs, names):
    """Robot body and simulator object masks of the first frame. The robot comes from the robot models the
    environment was built with (not from instance names); object masks are for head selection only."""
    seg = np.squeeze(obs["agentview_segmentation_instance"])
    model = env.env.model
    own = {m.name for robot in model.mujoco_robots for m in (robot, *robot.models)}
    ids = [i + 1 for i, name in enumerate(model.instances_to_ids) if name in own]
    if not ids:
        raise RuntimeError("no robot instances in the segmentation")
    objects = np.array([to_policy_frame(seg == env.instance_to_id[n]) if n in env.instance_to_id
                        else np.zeros((224, 224), bool) for n in names])
    return to_policy_frame(np.isin(seg, ids)), objects


class Recorder:
    """Per env step: which objects the hand touches, and every object's position and joint coordinates."""

    def __init__(self, env, names):
        sim = env.env.sim
        m = sim.model
        bodies = [m.body_id2name(i) or "" for i in range(m.nbody)]
        self.owner = {}
        for b, name in enumerate(bodies):   # longest prefix: every sub-body (drawer, knob, ...) maps to its object
            hits = [i for i, n in enumerate(names) if name == n or name.startswith(n + "_")]
            if hits:
                self.owner[b] = max(hits, key=lambda i: len(names[i]))
        self.hand = {b for b, name in enumerate(bodies) if name.startswith(HAND)}
        self.bodies = [env.env.obj_body_id[n] for n in names]
        self.joints = []   # (object, qpos address, width): free joints give position only, slide / hinge one value
        for j in range(m.njnt):
            if m.jnt_bodyid[j] in self.owner:
                width = {0: 3, 1: 4}.get(int(m.jnt_type[j]), 1)
                self.joints.append((self.owner[m.jnt_bodyid[j]], int(m.jnt_qposadr[j]), width))
        self.n = len(names)
        self.contacts, self.positions, self.coords = [], [], []

    def record(self, env):
        sim = env.env.sim
        body = sim.model.geom_bodyid
        touch = np.zeros(self.n, bool)
        for c in sim.data.contact[: sim.data.ncon]:
            b1, b2 = body[c.geom1], body[c.geom2]
            for hand, other in ((b1, b2), (b2, b1)):
                if hand in self.hand and other in self.owner:
                    touch[self.owner[other]] = True
        self.contacts.append(touch)
        self.positions.append([sim.data.body_xpos[b].copy() for b in self.bodies])
        self.coords.append(np.concatenate([sim.data.qpos[a:a + w] for _, a, w in self.joints])
                           if self.joints else np.zeros(0))

    def arrays(self):
        width = [w for _, _, w in self.joints]
        owner = np.repeat([o for o, _, _ in self.joints], width).astype(int) if self.joints else np.zeros(0, int)
        return {"contacts": np.array(self.contacts, bool).reshape(-1, self.n),
                "positions": np.array(self.positions).reshape(-1, self.n, 3),
                "joints": np.array(self.coords).reshape(len(self.coords), -1), "joint_owner": owner}


# ---------------------------------------------------------------- one rollout

def rollout(env, policy, item, initial, seed):
    """Run one instruction from one init state. Returns (scene, record): the first-frame data shared by every
    rollout of the scene, and this rollout's arrays and summary."""
    cfg = common.CONFIG["rollout"]
    obs = settle(env, seed, initial)
    names = sorted({**env.env.objects_dict, **env.env.fixtures_dict})
    robot, objects = scene_masks(env, obs, names)
    recorder = Recorder(env, names)
    queue = deque()
    calls = {"env_step": [], "attention": [], "image_attention": [], "image": []}
    actions, eef = [], []
    success, gate = False, None
    first_frame = None
    for t in range(cfg["max_steps"]):
        _, inputs = observation(obs, item["text"])
        if not queue:
            call = len(calls["env_step"])
            z = common.noise(seed, item["task"], item["init"], call)
            chunk, attention, image_attention = policy.act(inputs, z)
            if call == 0:
                first_frame = inputs["observation/image"]
                reference = policy.reference(inputs, z)
                gate = float(np.abs(chunk - reference).max())
                tol = cfg["gate_tolerance"]
                if not np.allclose(chunk, reference, atol=tol, rtol=tol):
                    raise RuntimeError(f"{common.key(item)}: capture changed the action ({gate:.2e})")
            calls["env_step"].append(t)
            calls["attention"].append(attention.astype(np.float16))
            calls["image_attention"].append(image_attention)
            calls["image"].append(inputs["observation/image"])
            queue.extend(chunk[: cfg["actions_per_call"]])
        action = queue.popleft()
        obs, _, done, _ = env.step(action.tolist())
        actions.append(action)
        eef.append(np.asarray(obs["robot0_eef_pos"]).copy())
        recorder.record(env)
        if env.check_success():
            success = True
            break
        if done:
            break
    scene = {"frame": first_frame, "robot": robot, "objects": objects, "names": names}
    record = {"calls": {k: np.array(v) for k, v in calls.items()},
              "trajectory": {"actions": np.array(actions), "eef": np.array(eef), "names": np.array(names),
                             **recorder.arrays()},
              "episode": {"task": item["task"], "sentence": item["sentence"], "instruction": item["text"],
                          "init": item["init"], "seed": seed, "steps": len(actions),
                          "calls": len(calls["env_step"]), "success": success, "gate_max_abs": gate}}
    return scene, record
