#!/bin/bash
#
# slurm specific parameters should be defined as comment line starting with #SBATCH
#SBATCH --job-name=coala
#SBATCH --gres=gpu:4g.40gb:1    # number of GPUs (type MIG 1g.10gb) 
#SBATCH --partition=luna-gpu-long    # using luna-short queue for a job that request up to 8h 
#SBATCH --mem=96G               # max memory per node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12       # max CPU cores per process
#SBATCH --time=01-12:00         # time limit (DD-HH:MM)
#SBATCH --mail-user=r.k.m.karimzadehmostafaabadi@amsterdamumc.nl
#SBATCH --mail-type=END
#SBATCH --output=/scratch/bmep/mastrampel/morphology/logs/siam_%A.out

module purge
module load GCCcore/11.2.0
module load Python/3.9.6

source /home/bmep/mastrampel/morphology/crlm/crlm/bin/activate


python rpnet3d_main.py
