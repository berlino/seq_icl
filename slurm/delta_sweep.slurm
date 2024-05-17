#!/bin/bash

#SBATCH --job-name=rnn # Job name
#SBATCH --partition=conference # Partition name
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --cpus-per-task=10            # Number of CPU cores per task
#SBATCH --output=slurm_log/regbench-rnn_%j.log  # Standard output and error log
#SBATCH --gres=gpu:1                 # Request GPU resource

source /data/cl/user/bailinw/miniconda3/etc/profile.d/conda.sh
conda activate seq_icl

export CUDA_HOME=/usr/local/cuda-12.1

export TRANSFORMERS_CACHE="./hg_cache"
export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES_INFO=0
export OMP_NUM_THREADS=8

python sweep.py --model_family delta --task dfa