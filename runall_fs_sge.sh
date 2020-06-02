#!/bin/bash
#$ -cwd
#$ -l h_vmem=512M
#$ -pe smp 8
#$ -R y
# -o and -e need to different for each user.
#$ -o /home/guest10/BioProject/JOBZ/pig_o/$JOB_ID.o_$TASK_ID
#$ -e /home/guest10/BioProject/JOBZ/pig_e/$JOB_ID.e_$TASK_ID

python pig.py calcfl $SGE_TASK_ID
#qsub -V -t 1-35  runall_sge.sh
