#!/bin/sh
#PBS -l select=1:ncpus=1:ngpus=1:mem=16gb
#PBS -l walltime=00:10:00
#PBS -q gpu
#PBS -P 12001577
#PBS -joe

cd $PBS_O_WORKDIR
module load singularity
module load cuda/10.1
nvidia-smi

singularity exec --nv --bind myenv/:/mnt MyPyTorchSandBox/ python /mnt/fcalc/main.py --mode custom_sequence --submode 1