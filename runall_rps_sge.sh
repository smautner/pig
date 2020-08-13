#!/bin/bash
#$ -cwd
#$ -l h_vmem=512M
#$ -pe smp 24
#$ -R y
# -o and -e need to different for each user.
#$ -o /scratch/bi01/mautner/guest10/JOBZ/pig_o/$JOB_ID.o_$TASK_ID
#$ -e /scratch/bi01/mautner/guest10/JOBZ/pig_e/$JOB_ID.e_$TASK_ID

python pig.py --calcrps $SGE_TASK_ID
