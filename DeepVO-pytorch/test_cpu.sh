#!/bin/bash
                                                                                            
# CPU setup:
#SBATCH -J FRD_CPU

#SBATCH -o /home/alijani/Datasets/kitti_color/DeepVO_misc/logs/cpu_test_%j_%a.txt
#SBATCH -e /home/alijani/Datasets/kitti_color/DeepVO_misc/logs/cpu_test_%j_%a.txt

#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32768M
#SBATCH --time=5-05:50:00
#SBATCH --partition=normal

source activate py27

PYTHONDONTWRITEBYTECODE=True
export PYTHONDONTWRITEBYTECODE 
srun clear

srun echo ""
srun echo "BATCH CPU ...!"
srun echo ""

now="$(date)"
CWD="$(pwd)"

srun echo "cur_dir: $CWD"
srun echo "home_dir: $HOME"
srun echo "Time: $now"


python test.py
srun echo "#######################################"
srun echo "python test.py with CPU DONE!"
srun echo "#######################################"

