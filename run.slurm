#!/bin/bash
#SBATCH --mem=100g
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --account=bcuv-delta-gpu   # <- match to a "Project" returned by the "accounts" command
#SBATCH --job-name=run
#SBATCH --time=03:00:00      # hh:mm:ss for the job
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out


module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module load python  # ... or any appropriate modules
# module load anaconda3_gpu
module list  # job documentation and metadata
echo "job is starting on `hostname`"
conda init
conda activate oat
which conda
which python
which pip
./run_naive.sh
