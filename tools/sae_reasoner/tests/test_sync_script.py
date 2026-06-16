import os
import subprocess
from pathlib import Path


def test_sync_to_runpod_requires_explicit_host_and_port():
    script = Path("tools/sae_reasoner/sync_to_runpod.sh")
    env = os.environ.copy()
    env.pop("COSMOS_SAE_RUNPOD_HOST", None)
    env.pop("COSMOS_SAE_RUNPOD_PORT", None)

    result = subprocess.run(["bash", str(script)], env=env, text=True, capture_output=True, check=False)

    assert result.returncode == 2
    assert "COSMOS_SAE_RUNPOD_HOST is required" in result.stderr
