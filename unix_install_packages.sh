pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/huggingface/transformers
pip install accelerate==1.10.0

# Model specific
pip install kernels # As of Aug 2025, dep for GPT-OSS (supports fa3 w/attention sinks)
pip install triton==3.4.0

# Vis
pip install plotly.express
pip install pyyaml
pip install pandas
pip install tqdm
pip install python-dotenv
pip install datasets