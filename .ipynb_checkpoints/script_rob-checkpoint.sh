#!/bin/bash -l
#$ -m ae
#$ -l h_rt=48:00:00
#$ -pe omp 16
# Join output and error streams to reduce the clutter
#$ -j y
# Specify the number of tasks (number of parallel jobs you want to run
#$ -t 1-31

module load gcc
module load python3/3.6.5

#myarray=(0, 1.0 1.1 1.2)
myarray=(0, 1.2  1.0  1.4  1.3  1.1  1.5  1.6  1.7  1.8  1.9  2.0  2.5  4.0  3.0  3.5  2.1  2.6  3.1  3.6  2.2  2.3  2.7  2.8  3.2  3.3  3.7  3.8  2.4  2.9  3.4  3.9)
python main.py ${myarray[$SGE_TASK_ID]} $INIT_DIR




