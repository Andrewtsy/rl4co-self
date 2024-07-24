#!/bin/bash

#SBATCH --output=results/%j.out
#SBATCH --error=results/%j.err
#SBATCH -t 96:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -J "decomposition"
#SBATCH --cpus-per-task=3
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu 8GB
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH -A hpc-prf-winf4gpu

export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

module load lang/Python/3.10.4-GCCcore-11.3.0
source /scratch/hpc-prf-winf4/atao/.venv/bin/activate

python /scratch/hpc-prf-winf4/atao/rl4co-self/run.py "$@"
