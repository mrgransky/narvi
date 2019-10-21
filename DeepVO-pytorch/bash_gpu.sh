#!/bin/bash
                                                                                            
# GPU setup:
#SBATCH -J FRD_GPU
#SBATCH --output=output_%j.txt # STDOUT
#SBATCH --error=output_%j.txt # STDOUT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --mem=20000
#SBATCH --time=5-05:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2

module load CUDA/9.0
source activate py27
srun clear
srun echo "BATCH GPU ...!"
now="$(date)"
computer_name="$(hostname)"
CWD="$(pwd)"

srun echo "cur_dir: $CWD"
srun echo "home_dir: $HOME"
srun echo "Time: $now"

max_idx=0
max_mem=0
idx=0

{
  read _;                         # discard first line (header)
  while read -r mem _; do         # for each subsequent line, read first word into mem
    if (( mem > max_mem )); then  # compare against maximum mem value seen
      max_mem=$mem                # ...if greater, then update both that max value
      max_idx=$idx                # ...and our stored index value.
    fi
    ((++idx))
  done
} < <(nvidia-smi --query-gpu=memory.free --format=csv)

srun echo "Maximum memory seen is $max_mem, at processor $idx"

python -V
srun echo "#######################################"

python main.py

srun echo "All Done!"
