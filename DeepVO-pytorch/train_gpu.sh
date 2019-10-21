#!/bin/bash
                                                                                            
# GPU setup:
#SBATCH -J FRD_GPU

#SBATCH -o gpu_train_%j.txt
#SBATCH -e gpu_train_%j.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32768M
#SBATCH --time=5-05:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load CUDA/9.0
source activate py27
srun clear
srun echo "BATCH GPU ...!"
now="$(date)"
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
srun echo "#######################################"
srun echo "python main.py with GPU DONE!"
srun echo "#######################################"
