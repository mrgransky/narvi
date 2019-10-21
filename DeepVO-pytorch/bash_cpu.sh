#!/bin/bash
                                                                                            
# CPU setup:
#SBATCH -J FRD_CPU
#SBATCH --output=log_%a.txt
#SBATCH --error=log_%a.txt

#SBATCH --ntasks=1
#SBATCH --cpu-per-task=1

#SBATCH --mem=25000
#SBATCH --ntasks=5-05:50:00
#SBATCH --partition=normal

source activate py27

srun echo "starting up...!"

CWD="$(pwd)"
srun echo "cur_dir:"
srun echo $CWD

srun echo "home_dir:"
srun echo $HOME

python -V
python main.py
