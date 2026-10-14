#!/usr/bin/env bash
# Pinned third-party code and the three Python environments, under PREFIX (default: the folder above the repo).
#
#   env/setup.sh [PREFIX]
#
# Needs git, uv (https://docs.astral.sh/uv/) and an NVIDIA driver for CUDA 12. Writes PREFIX/third_party/{openpi,
# LIBERO-Para}, PREFIX/envs/{policy,sam,vlm} and env/local.sh (edit the checkpoint and Hugging Face paths there).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
prefix="$(mkdir -p "${1:-$(dirname "$repo")}" && cd "${1:-$(dirname "$repo")}" && pwd)"
pin() { python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['pinned'][sys.argv[2]])" "$repo/config.json" "$1"; }

# openpi at the pinned commit, with the read-only attention capture
if [ ! -d "$prefix/third_party/openpi" ]; then
  git clone https://github.com/Physical-Intelligence/openpi.git "$prefix/third_party/openpi"
  git -C "$prefix/third_party/openpi" checkout "$(pin openpi_commit)"
  git -C "$prefix/third_party/openpi" apply "$repo/env/openpi_attention_capture.patch"
fi
# LIBERO-Para at the pinned commit (LIBERO-Goal scenes and the 4,092 paraphrases)
if [ ! -d "$prefix/third_party/LIBERO-Para" ]; then
  git clone https://github.com/cau-hai-lab/LIBERO-Para.git "$prefix/third_party/LIBERO-Para"
  git -C "$prefix/third_party/LIBERO-Para" checkout "$(pin libero_para_commit)"
fi

# policy: JAX + openpi's dependencies + LIBERO's
uv venv --python 3.11 "$prefix/envs/policy"
uv pip install --python "$prefix/envs/policy/bin/python" -r "$repo/env/policy.txt"
# SAM 2.1: the policy environment's packages plus the SAM 2 package at the pinned commit (no CUDA extension)
uv venv --python 3.11 "$prefix/envs/sam"
uv pip install --python "$prefix/envs/sam/bin/python" -r "$repo/env/policy.txt" -r "$repo/env/sam.txt"
SAM2_BUILD_CUDA=0 uv pip install --python "$prefix/envs/sam/bin/python" --no-deps \
  "git+https://github.com/facebookresearch/sam2.git@$(pin sam2_package_commit)"
# VLM: transformers + torch (CUDA 12.6 wheels)
uv venv --python 3.11 "$prefix/envs/vlm"
uv pip install --python "$prefix/envs/vlm/bin/python" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu126 -r "$repo/env/vlm.txt"

if [ ! -f "$repo/env/local.sh" ]; then
  sed -e "s|/path/to/runs|$prefix/runs|" -e "s|/path/to/openpi|$prefix/third_party/openpi|" \
      -e "s|/path/to/LIBERO-Para|$prefix/third_party/LIBERO-Para|" \
      -e "s|/path/to/policy-env|$prefix/envs/policy|" -e "s|/path/to/sam-env|$prefix/envs/sam|" \
      -e "s|/path/to/vlm-env|$prefix/envs/vlm|" "$repo/env/local.sh.example" > "$repo/env/local.sh"
  echo "wrote env/local.sh — set OPENPI_CHECKPOINT and HF_HOME"
fi
