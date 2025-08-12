#!/bin/bash
# These are APT updates needed for use on Runpod

# Update & upgrade system packages
apt update -y \
&& apt upgrade -y

# Install nano
apt install -y nano

# Upgrade/install Jupyter Lab and widgets
pip install --upgrade jupyterlab ipywidgets jupyterlab-widgets

# # Install Python 3.12
# apt install -y python3.12
# apt-get install -y python3.12-dev

# # Set Python 3.12 as the default python
# update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# # Install python3.12-venv
# apt install -y python3.12-venv

# Ensure pip is up to date for Python
python -m ensurepip --upgrade

# Install packages
echo "/workspace/interpretable-moes-analysis" > $(python -c "import site; print(site.getsitepackages()[0])")/add_path_analysis.pth
sh unix_install_packages.sh