#!/bin/bash
#$ -cwd
#$ -l h_vmem=512M
#$ -M mautner@cs.uni-freiburg.de
#$ -m as
#$ -pe smp 24
#$ -R y
#$ -o /home/mautner/JOBZ/pig_o/$JOB_ID.o_$TASK_ID
#$ -e /home/mautner/JOBZ/pig_e/$JOB_ID.e_$TASK_ID

##mkdir -p /home/mautner/JOBZ/reconstr_o/
##mkdir -p /home/mautner/JOBZ/reconstr_e/
python pig.py $SGE_TASK_ID
#qsub -V -t 1-35  runall_sge.sh
