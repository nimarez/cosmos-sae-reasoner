#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-auto}"
COSMOS_SAE_USE_SYSTEM_TORCH="${COSMOS_SAE_USE_SYSTEM_TORCH:-1}"

venv_args=(.venv --python "${PYTHON_VERSION}" --seed)
requirements="tools/sae_reasoner/requirements-runpod.txt"
tmp_requirements=""
tmp_constraints=""

if [[ "${COSMOS_SAE_USE_SYSTEM_TORCH}" == "1" ]] && python - <<'PY'
import torch
import torchvision

print(f"system torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
print(f"system torchvision={torchvision.__version__}")
PY
then
  venv_args+=(--system-site-packages)
  tmp_requirements="$(mktemp)"
  tmp_constraints="$(mktemp)"
  grep -Ev '^(torch|torchvision)([<=> ].*)?$' tools/sae_reasoner/requirements-runpod.txt > "${tmp_requirements}"
  python - <<'PY' > "${tmp_constraints}"
import torch
import torchvision

print(f"torch=={torch.__version__}")
print(f"torchvision=={torchvision.__version__}")
PY
  requirements="${tmp_requirements}"
else
  echo "System torch not available; installing torch into the virtualenv."
fi

uv venv "${venv_args[@]}"
source .venv/bin/activate
install_args=(--torch-backend="${UV_TORCH_BACKEND}" -r "${requirements}")
if [[ -n "${tmp_constraints}" ]]; then
  install_args+=(--constraint "${tmp_constraints}")
fi
uv pip install "${install_args[@]}"
rm -f "${tmp_requirements}" "${tmp_constraints}"
python - <<'PY'
import torch

print(f"active torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
PY
python -m tools.sae_reasoner --help
