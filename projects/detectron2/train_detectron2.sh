#!/bin/bash
#SBATCH -N1
#SBATCH -J train
#SBATCH -o slurm-%j.out
#SBATCH -e slurm-%j.err

CONTAINER_PATH='$NOBACKUP/slump-detectron2_11.1.sif'
CONFIG_FILE='$NOBACKUP/slump-detection/projects/detectron2/config/slump_mask_rcnn_R_50_FPN_3x.yaml'

singularity exec -B $NOBACKUP:$NOBACKUP $CONTAINER_PATH python $NOBACKUP/slump-detection/projects/detectron2/train_detectron2.py -c $CONFIG_FILE