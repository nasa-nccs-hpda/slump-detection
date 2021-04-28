#!/bin/bash
#SBATCH -N1
#SBATCH -J gen_dataset
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

# Input: container location given as a parameter on the command line
singularity exec -B $NOBACKUP:$NOBACKUP,$TNOBACKUP:$TNOBACKUP $1 python $TNOBACKUP/slump-detection/projects/detectron2/gen_dataset.py -c config/slump_mask_rcnn_R_50_FPN_3x.yaml