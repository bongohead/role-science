# #!/usr/bin/env bash

# Use this as an alternative to runpod_setup.sh
# # Prep a Runpod box with uv + Python 3.12 venv + Jupyter kernel

# # 0) System prep
# apt update -y && apt upgrade -y
# apt install -y curl ca-certificates nano build-essential pkg-config

# # 1) Install uv for the current user
# curl -LsSf https://astral.sh/uv/install.sh | sh
# export PATH="$HOME/.local/bin:$PATH"

# # 2) Create a Python 3.12 environment (managed by uv)
# uv python install 3.12
# uv venv --python 3.12

# # 3) Jupyter stack + kernel
# uv pip install -U ipykernel ipywidgets # jupyterlab jupyterlab-widgets # jupyterlab optional if mostly using vscode
# python -m ipykernel install --user --name py312-uv --display-name "Python 3.12 (uv)"

# # 4) Path injection
# python - <<'PY'
# import sysconfig, pathlib
# pth = pathlib.Path(sysconfig.get_paths()['purelib']) / "add_path_analysis.pth"
# pth.write_text("/workspace/interpretable-moes-analysis\n")
# print(f"Wrote {pth}")
# PY

# # 5) Install packages
# # Speed + stability
# export UV_HTTP_TIMEOUT=600

# uv pip install --no-cache-dir  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# uv pip install --no-cache-dir \
#     transformers==4.55.3\
#     accelerate==1.10.0\
#     kernels\
#     triton==3.4.0\
#     datasets\
#     python-dotenv pyyaml tqdm\
#     pandas\
#     plotly
# # To reset
# # rm -rf .venv
# # jupyter kernelspec remove py312-uv -y # jupyter kernelspec list
# # uv cache clean
# # uv cache prune
# # uv cache dir


