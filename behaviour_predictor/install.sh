#!/bin/bash
echo "Pulling submodules"
bash pull_submodules.sh
echo "Installing mamba"
conda install mamba -n base -c conda-forge
if ! command -v mamba &> /dev/null
then
    echo "mamba could not be found. Please install mamba before running this script"
    exit
fi
echo "Creating conda environment"
mamba create -n behaviour python=3.8
eval "$(conda shell.bash hook)" # make sure conda works in the shell script
conda activate behaviour
if echo $CONDA_PREFIX | grep behaviour
then
    echo "Conda environment successfully activated"
else
    echo "Conda environment not activated. Probably it was not created successfully for some reason. Please activate the conda environment before running this script"
    exit
fi
echo "Installing conda packages"
mamba env update -n behaviour --file behaviour_predictor_py38_cu11.yml
