#!/bin/bash
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Installing mamba"
    conda install mamba -n base -c conda-forge
fi
set -e
echo "Pulling submodules"
bash pull_submodules.sh
set +e
echo "====================================================================="

echo "pull EMOCA and setup for py38 environment"
echo "EMOCA"
echo "1. pull submodules for emoca"
(cd external_libs/emoca/ && bash install_38.sh)
(cd external_libs/emoca/gdl_app/EMOCA/demo/ && bash download_assets.sh)
echo "======================================================================"

