#!/bin/bash
set -euo pipefail

# Assumes setup with: Python3.12, CUDA 12, transformers v5, torch 2.9
# This file will setup the repo's Python environment. It will:
#  1. Install UV
#  2. Install a Python 3.12 `venv` in the provided $VENV_DIR if it doesn't already exist
#  3. Install necessary packages
#  4. Setup Jupyter
#  5. Add $PROJECT_DIR to the working directory when the venv is activated, so that module imports work correctly 
#
# Before running, set PROJECT_DIR to this repo directory; VENV_DIR to the desired venv install location, and
# KERNEL_NAME to the desired Jupyter kernel name.

# Set constants
PROJECT_DIR="/workspace/deliberative-alignment-jailbreaks"
VENV_DIR="$PROJECT_DIR/.venv"
KERNEL_NAME="role-analysis-uv"

# Persist across restarts
export UV_CACHE_DIR="/workspace/.uv-cache"
export UV_PYTHON_INSTALL_DIR="/workspace/.uv-python"
export UV_HTTP_TIMEOUT=120


# ---------- 1. Install UV (idempotent) ----------
cd "$PROJECT_DIR"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
case ":$PATH:" in *":$HOME/.local/bin:"*) :;; *) export PATH="$HOME/.local/bin:$PATH";; esac # Make sure uv is in PATH


# ---------- 2. Install venv ----------
# Set Python 3.12 location (download if needed)
UVPY="$(uv python find 3.12 2>/dev/null || (uv python install 3.12 >/dev/null 2>&1 && uv python find 3.12))"

# Create the venv with Python 3.12 (uv will download 3.12 if needed)
if [ -x "$VENV_DIR/bin/python" ]; then
  # Check that it actually runs and is 3.12
  if ! "$VENV_DIR/bin/python" -c 'import sys; exit(0 if sys.version_info[:2]==(3,12) else 1)'; then
    echo "Repairing venv Python links (keeping installed packages)..."
    "$UVPY" -m venv "$VENV_DIR" --upgrade --without-pip   # relink python, keep site-packages
    "$VENV_DIR/bin/python" -m ensurepip --upgrade || true
  else
    echo "Using existing venv at $VENV_DIR"
  fi
else
  if [ -d "$VENV_DIR" ]; then
    echo "Repairing existing venv (python missing)…"
    "$UVPY" -m venv "$VENV_DIR" --upgrade --without-pip
    "$VENV_DIR/bin/python" -m ensurepip --upgrade || true
  else
    echo "Creating venv…"
    uv venv "$VENV_DIR" --python 3.12 --seed
  fi
fi


# ---------- 3. Install packages ----------
uv pip install --python "$VENV_DIR/bin/python" --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1

uv pip install --python "$VENV_DIR/bin/python" \
  transformers==4.57.5 hf_transfer==0.1.9 accelerate==1.12.0 triton==3.5.1 \
  tiktoken==0.12.0 blobfile==3.1.0 kernels==0.11.5 \
  compressed-tensors==0.13.0 \
  plotly pandas kaleido python-dotenv pyyaml tqdm termcolor \
  datasets

# Optional, needed for ReAct loop agent testing
uv pip install --python $VENV_DIR/bin/python openai

# Below needed for plotly exports
uv run --python "$VENV_DIR/bin/python" plotly_get_chrome -y
apt update && apt-get install libnss3 libatk-bridge2.0-0 libcups2 libxcomposite1 libxdamage1 libxfixes3 libxrandr2 libgbm1 libxkbcommon0 libpango-1.0-0 libcairo2 -y

# Flash-Attention (separate so we can set build flags independently if needed)
# uv pip install --python "$VENV_DIR/bin/python" --only-binary=:all: flash-attn==2.8.3 # Only use prebuilt wheels
# uv pip install --python "$VENV_DIR/bin/python" flash-attn==2.8.3 --no-build-isolation # Allow build
# FA_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
FA_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.5.4/flash_attn-2.8.3+cu128torch2.9-cp312-cp312-linux_x86_64.whl"
uv pip install --python "$VENV_DIR/bin/python" "$FA_URL"

# RAPIDS (search NVIDIA index for cudf/cuML — PyPI/extra index support is documented)
uv pip install --python "$VENV_DIR/bin/python" libucx-cu12==1.18.1 ucx-py-cu12==0.45.0 # Dependencies from pypi for RAPIDS - install separately to avoid error
uv pip install --python "$VENV_DIR/bin/python" --extra-index-url https://pypi.nvidia.com "cudf-cu12==25.9.*" "cuml-cu12==25.9.*"

# ---------- 4. Setup Jupyter ----------
# Jupyter (server + kernel + widgets + nbformat)
uv pip install --python "$VENV_DIR/bin/python" jupyterlab jupyter_server ipykernel ipywidgets nbformat notebook

# Jupyter kernel (visible to any server)
"$VENV_DIR/bin/python" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Role analysis (uv)"


# ---------- 5. Add import paths ----------
SITE_DIR="$("$VENV_DIR/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf "%s\n" "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"

# Final
echo "Done. Kernel: $KERNEL_NAME  |  Python: $("$VENV_DIR/bin/python" -V)"