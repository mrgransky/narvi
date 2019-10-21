#!/bin/bash
                                                                                            
# GPU setup:
#SBATCH -J FRD_GPU
#SBATCH --output=log_%a.txt
#SBATCH --error=log_%a.txt

#SBATCH --ntasks=1
#SBATCH --cpu-per-task=1

#SBATCH --mem=20000
#SBATCH --time=5-05:50:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1


module load CUDA/9.0
source activate py27

srun echo "starting up...!"

CWD="$(pwd)"
srun echo "cur_dir:"
srun echo $CWD

srun echo "home_dir:"
srun echo $HOME

python -V
python main.py
