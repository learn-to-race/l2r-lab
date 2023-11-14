#!/bin/bash

# Install Anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2023.09-0-Linux-x86_64.sh
chmod +x ./Anaconda3-2023.09-0-Linux-x86_64.sh
./Anaconda3-2023.09-0-Linux-x86_64.sh -b
source /root/anaconda3/bin/activate 

# Install mamba
conda create --name initialization python=3.8 -y
conda activate initialization
conda install -c conda-forge mamba -y

# Create l2r environment using mamba
mamba create --name l2r python=3.8 -y
mamba init
source /root/.bashrc 
mamba activate l2r

# Install necessary packages
mamba install git -y
mamba install -c conda-forge strictyaml -y
mamba install -c conda-forge pyyaml -y
mamba install -c conda-forge matplotlib -y
mamba install -c conda-forge wandb -y
mamba install -c conda-forge tensorboardx -y
mamba install -c conda-forge gym -y
mamba install -c conda-forge jsonpickle -y
mamba install -c conda-forge scipy -y
mamba install -c conda-forge gym-box2d -y
mamba install -c conda-forge opencv -y
mamba install -c conda-forge tianshou -y
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install git+https://github.com/learn-to-race/l2r@aicrowd-environment
pip install tianshou

# Set up Ubuntu update (run multiple times in case failed)
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC

# Solve "version `GLIBCXX_3.4.29' not found" error
# Ref: https://github.com/lhelontra/tensorflow-on-arm/issues/13#issuecomment-418202182
sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update
sudo apt-get upgrade libstdc++6 -y

# Prepare the repository
cd /
mkdir workspace
cd workspace
git clone https://github.com/BrandonBian/l2r-distributed.git
cd l2r-distributed
