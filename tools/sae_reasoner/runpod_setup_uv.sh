#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"

uv venv .venv --python "${PYTHON_VERSION}" --seed
source .venv/bin/activate
uv pip install --torch-backend="${UV_TORCH_BACKEND}" -r tools/sae_reasoner/requirements-runpod.txt
python -m tools.sae_reasoner --help

