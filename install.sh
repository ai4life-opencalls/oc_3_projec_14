#! /bin/sh
# This script will install FeatureForest and its dependencies in a new conda environment.
echo "Create a new conda environment named tree with Python 3.10"
conda init
conda create -n tree -y python=3.10
echo "Activate the tree environment"
source activate base
conda activate tree
echo "Install PyTorch and torchvision using light-the-torch"
python -m pip install light-the-torch
ltt install 'torch>=2.5.1' 'torchvision>=0.20.1'
echo "Install additional dependencies"
conda install -c conda-forge dask -y
python -m pip install -r ./requirements.txt
echo "Install FeatureForest from GitHub"
python -m pip install git+https://github.com/juglab/featureforest.git
