#!/bin/bash
set -eu pipefail

# These are APT updates needed for use on Runpod
KERNEL_NAME=redteam-venv
PROJECT_DIR="/workspace/deliberative-alignment-jailbreaks"

# System prepare
apt update -y && apt upgrade -y
apt install -y nano python3.12 python3.12-venv python3.12-dev python3-pip

# Make/activate a Python 3.12 venv for this project
python3.12 -m venv "$PROJECT_DIR/.venv"

"$PROJECT_DIR/.venv/bin/python" -m pip install -U pip setuptools wheel

# Upgrade/install Jupyter Lab and widgets within venv
"$PROJECT_DIR/.venv/bin/python"  -m pip install -U ipykernel ipywidgets # jupyterlab jupyterlab-widgets
"$PROJECT_DIR/.venv/bin/python"  -m ipykernel install --user --name "$KERNEL_NAME" --display-name $KERNEL_NAME

# Install packages
SITE_DIR="$("$PROJECT_DIR/.venv/bin/python"  -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "$PROJECT_DIR" > "$SITE_DIR/add_path_analysis.pth"

# echo $PROJECT_DIR > $(python -c "import site; print(site.getsitepackages()[0])")/add_path_analysis.pth
# sh unix_install_packages.sh