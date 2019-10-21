#!/bin/bash
                                                                                            
# GPU setup:
#SBATCH -J FRD_GPU
#SBATCH -o logs/output_%a.txt
#SBATCH -e logs/error_%a.txt
#SBATCH -n 1
#SBATCH -c 1

#SBATCH --mem=20000
#SBATCH --time=5-05:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load CUDA/9.0
source activate py27
srun clear
srun echo "BATCH GPU ...!"
now = "$(date)"
computer_name = "$(hostname)"
CWD="$(pwd)"
srun echo "cur_dir: $CWD"
srun echo "home_dir: $HOME"
srun echo "host: $hostname"
echo "Current time : $now"

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

echo "Maximum memory seen is $max_mem, at processor $idx"

python -V
srun echo "#######################################"

python main.py
