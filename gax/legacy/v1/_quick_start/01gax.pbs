#!/bin/sh
#PBS -l select=1:ncpus=5:ngpus=1:mem=16gb
#PBS -l walltime=00:10:00
#PBS -q gpu
#PBS -j oe
#PBS -P 12001577

cd $PBS_O_WORKDIR
module load cuda/10.1
module load singularity
singularity exec --nv --bind gax/:/mnt MainSB/ python /mnt/main_pneu.py --mode train --PROJECT_ID pneu_testrun --n_iter 64 --batch_size 4 --realtime_print 1 --n_debug 16 --ROOT_DIR gax 

