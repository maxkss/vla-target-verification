#!/usr/bin/env python3
"""Task objects of every rollout (paper section 2.1): an external VLM (Qwen3.5-9B) names, on the first frame,
the candidates the gripper or the object it holds will touch — the object it acts on and its destination.
Asked once per rollout, before the policy moves; fixed for the whole rollout. Runs in the VLM environment.

Input: the first frame re-rendered at 896 px, the same image with every candidate outlined and numbered, and
the rollout's instruction. Never: labels, simulator object names, later frames, the policy's attention.
Greedy decoding, thinking off. The task-object set is the chosen candidates plus every candidate belonging to
their region (lying mostly inside it, or covering most of it). An unclear or unusable reply leaves the rollout
without task objects: no verdict, no alarm.

  task_objects.py RUN [--worker i/n]   -> <task>/scenes/init<NN>/numbered.png, <rollout>/task_objects.json
"""

import argparse
import json
import re
import time

import numpy as np

import common


def erode(mask):
    p = np.pad(mask, 1)
    return p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1] & p[1:-1, :-2] & p[1:-1, 2:]


def depth(mask):
    """How many 4-neighbour erosions each pixel survives (0 outside)."""
    out, cur = np.zeros(mask.shape, int), mask.copy()
    while cur.any():
        out += cur
        cur = erode(cur)
    return out


def numbered(base, masks, cfg):
    """The 896-px frame with every candidate outlined in its colour and numbered 1..N on a filled tag, placed
    deepest inside the part no smaller candidate covers, away from tags already placed (smallest first)."""
    from PIL import Image, ImageDraw, ImageFont

    scale, line, palette = cfg["scale"], cfg["outline"], [tuple(c) for c in cfg["palette"]]
    font = ImageFont.truetype(cfg["font"], cfg["font_size"])
    size = masks.shape[-1] * scale
    assert base.size == (size, size), base.size
    canvas = np.asarray(base).copy()
    areas = masks.reshape(len(masks), -1).sum(1)
    for i, m in enumerate(masks):
        big = np.kron(m, np.ones((scale, scale), bool))
        inner = big
        for _ in range(line):
            inner = erode(inner)
        canvas[big & ~inner] = palette[i % len(palette)]
    out = Image.fromarray(canvas)
    draw = ImageDraw.Draw(out)
    placed = []
    for i in np.argsort(areas, kind="stable"):
        smaller = np.zeros_like(masks[i])
        for j in np.flatnonzero(areas < areas[i]):
            smaller |= masks[j]
        own = masks[i] & ~smaller
        dep = depth(own if own.sum() >= 0.2 * areas[i] else masks[i])
        text = str(i + 1)
        w, h = font.getlength(text) + 8, font.size + 6
        spots = sorted(zip(*np.nonzero(dep)), key=lambda yx: -dep[yx])
        box = None
        for y, x in spots:
            cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
            left = max(0, min(cx - w / 2, size - w))
            top = max(0, min(cy - h / 2, size - h))
            b = (left, top, left + w, top + h)
            if all(b[2] <= q[0] or b[0] >= q[2] or b[3] <= q[1] or b[1] >= q[3] for q in placed):
                box = b
                break
        if box is None:   # nowhere free: the deepest spot regardless
            y, x = spots[0]
            cx, cy = (x + 0.5) * scale, (y + 0.5) * scale
            box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        placed.append(box)
        draw.rectangle(box, fill=palette[i % len(palette)], outline=(255, 255, 255), width=2)
        draw.text((box[0] + 4, box[1] + 2), text, font=font, fill=(255, 255, 255), stroke_width=2,
                  stroke_fill=(0, 0, 0))
    return out


def parse(text, n):
    """The reply's JSON -> (reply, status, chosen 0-based). status: ok / unclear / parse_error / invalid."""
    found = re.search(r"\{.*\}", text, re.S)
    if not found:
        return None, "parse_error", []
    try:
        reply = json.loads(found.group(0))
    except json.JSONDecodeError:
        return None, "parse_error", []
    try:
        chosen = sorted({int(k) for k in reply.get("contact") or []})
    except (TypeError, ValueError):
        return reply, "invalid", []
    if any(not 1 <= k <= n for k in chosen):
        return reply, "invalid", []
    if reply.get("unclear") is True:
        return reply, "unclear", []
    return reply, ("ok" if chosen else "invalid"), [k - 1 for k in chosen]


def region(masks, chosen, overlap):
    """The chosen candidates and every candidate lying mostly inside their union or covering most of it."""
    if not chosen:
        return []
    union = np.any(masks[chosen], axis=0)

    def belongs(k):
        inter = (masks[k] & union).sum()
        return inter > overlap * masks[k].sum() or inter > overlap * max(union.sum(), 1)

    return [k for k in range(len(masks)) if k in chosen or belongs(k)]


class Judge:
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.torch, self.cfg = torch, cfg
        self.processor = AutoProcessor.from_pretrained(cfg["model"], revision=cfg["revision"])
        self.model = AutoModelForMultimodalLM.from_pretrained(cfg["model"], revision=cfg["revision"],
                                                              dtype=torch.bfloat16, device_map="cuda").eval()

    def ask(self, images, text):
        messages = [{"role": "user", "content": [*({"type": "image", "image": im} for im in images),
                                                 {"type": "text", "text": text}]}]
        rendered = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False,
                                                      enable_thinking=False)
        assert rendered.endswith("<think>\n\n</think>\n\n"), rendered[-60:]   # thinking off
        inputs = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt",
                                                    enable_thinking=False).to(self.model.device)
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, do_sample=False, max_new_tokens=self.cfg["max_new_tokens"])
        new = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(new, skip_special_tokens=True).strip(), int(len(new))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--worker", default="0/1")
    a = p.parse_args()
    from PIL import Image

    run = common.Run(a.run)
    cfg = common.CONFIG["task_objects"]
    started = common.now()
    todo = [it for it in run.items(a.worker) if not (run.rollout_dir(it) / "task_objects.json").exists()]
    judge = Judge(cfg) if todo else None
    images = {}
    for item in todo:
        scene = run.scene_dir(item["task"], item["init"])
        with np.load(scene / "candidates.npz") as z:
            masks = z["masks"]
        if scene not in images:
            base = Image.open(scene / "frame_896.png").convert("RGB")
            marked = numbered(base, masks, cfg)
            if not (scene / "numbered.png").exists():
                common.save_png(scene / "numbered.png", marked)
            images[scene] = (base, marked)
        t0 = time.time()
        text, tokens = judge.ask(list(images[scene]), cfg["prompt"].replace("{instruction}", item["text"]))
        reply, status, chosen = parse(text, len(masks))
        record = {"status": status, "chosen": chosen,
                  "task_objects": region(masks, chosen, cfg["overlap"]) if status == "ok" else None,
                  "objects": (reply or {}).get("objects"), "reply": reply, "raw": text,
                  "candidates": int(len(masks)), "seconds": round(time.time() - t0, 3), "new_tokens": tokens}
        common.write_json(run.rollout_dir(item) / "task_objects.json", record)
        print(f"{common.key(item)}: {status} {[k + 1 for k in chosen]} ({record['objects']})", flush=True)
    i, n = a.worker.split("/")
    run.record_stage(f"task_objects.w{i}of{n}", started, rollouts=len(todo))


if __name__ == "__main__":
    main()
