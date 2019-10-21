#!/bin/bash
                                                                                            
# CPU setup:
#SBATCH -J FRD_CPU

#SBATCH -o cpu_test_%j.txt
#SBATCH -e cpu_test_%j.txt

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32768M
#SBATCH --time=5-05:50:00
#SBATCH --partition=normal

source activate py27

srun clear
srun echo "BATCH CPU ...!"
now="$(date)"
CWD="$(pwd)"

srun echo "cur_dir: $CWD"
srun echo "home_dir: $HOME"
srun echo "Time: $now"

python -V
srun echo "#######################################"

python test.py
srun echo "#######################################"
srun echo "python test.py with CPU DONE!"
srun echo "#######################################"

