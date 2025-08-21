#!/usr/bin/env bash
set -eu pipefail

if [ -z "${VENV:-}" ]; then
  . .venv/bin/activate
fi

# Misc
pip install plotly.express pandas kaleido
pip install python-dotenv pyyaml tqdm termcolor
pip install datasets

pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate==1.10.0

# Model specific
pip install kernels==0.9.0 # As of Aug 2025, dep for GPT-OSS (supports fa3 w/attention sinks)
pip install triton==3.4.0
pip install tiktoken


# Analysis
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.8.*" "cuml-cu12==25.8.*"
