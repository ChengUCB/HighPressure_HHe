#!/bin/bash
#
#----------------------------------
# single GPU + single CPU example
#----------------------------------
#
#SBATCH --job-name=test
#
#Define the number of hours the job should run.
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=24:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
#SBATCH --mem=32G
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning,
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Define the "gpu" partition for GPU-accelerated jobs
#SBATCH --partition=gpu
#SBATCH --exclude=gpu[113,114,118,119,123-127,136-139,144-148,150]
#
#Define the number of GPUs used by your job
#SBATCH --gres=gpu:1
#
#Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
##SBATCH --constraint=GTX980
#
#Do not export the local environment to the compute nodes
##SBATCH --export=NONE
##unset SLURM_EXPORT_ENV
#

source ~/.bashrc
#load an CUDA software module
module load cuda/11.8.0

python /nfs/scistore14/chenggrp/bcheng/programs/mace/scripts/run_train.py \
    --name="hhe_train_ist" \
    --train_file="../hhe-xyz/train-all-less.xyz" \
    --valid_fraction=0.02 \
    --test_file="../hhe-xyz/test-all.xyz" \
    --E0s='{1:0.0, 2:0.0}' \
    --model="MACE" \
    --num_interactions=2 \
    --num_channels=64 \
    --max_L=1 \
    --correlation=2 \
    --r_max=3.0 \
    --forces_weight=10 \
    --energy_weight=1000 \
    --stress_weight=0 \
    --energy_key="TotEnergy_VDW" \
    --forces_key="force_VDW" \
    --batch_size=12 \
    --valid_batch_size=8 \
    --max_num_epochs=650 \
    --start_swa=50 \
    --scheduler_patience=15 \
    --patience=30 \
    --eval_interval=4 \
    --ema \
    --swa \
    --error_table='PerAtomRMSE' \
    --default_dtype="float32"\
    --device=cuda \
    --seed=3 \
    --restart_latest \
    --save_cpu
