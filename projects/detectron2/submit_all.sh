#!/bin/bash
#SBATCH -N1
#SBATCH -J slump_all
module load singularity
CONTAINER_PATH='/att/nobackup/ewebb5/slump-detectron2_11.1.sif'
CONFIG_FILE='/att/nobackup/ewebb5/slump-detection/projects/detectron2/config/slump_mask_rcnn_R_50_FPN_3x.yaml'

# Bandaid for removing existing files
rm -rf ../../data/slump-detection_trialrun_*
rm -rf ../../data/model/WV02_20160709_M1BS_10300100591D6600-toa_pansharpen_pred.tif

singularity exec --nv -B /att/nobackup/ewebb5:/att/nobackup/ewebb5 $CONTAINER_PATH python /att/nobackup/ewebb5/slump-detection/projects/detectron2/gen_dataset.py -c $CONFIG_FILE

singularity exec --nv -B /att/nobackup/ewebb5:/att/nobackup/ewebb5 $CONTAINER_PATH python /att/nobackup/ewebb5/slump-detection/projects/detectron2/train_detectron2.py -c $CONFIG_FILE

singularity exec --nv -B /att/nobackup/ewebb5:/att/nobackup/ewebb5 $CONTAINER_PATH python /att/nobackup/ewebb5/slump-detection/projects/detectron2/predict_detectron2.py -c $CONFIG_FILE