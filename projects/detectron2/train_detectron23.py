#Src: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0

import os, sys
sys.path.append('/att/gpfsfs/briskfs01/ppl/acartas/git/anikautil/')
from anikautil import CodeTimer

from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import pdb
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.data.datasets import register_coco_instances


baseDir = '/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health.v2/'
inSmall = baseDir+"/input/datasets/OR_20190630_Three_Creek_0p1m/test/c3r2_c16r38/images/OR_20190630_Three_Creek_c3r2_c16r38.png"
inLrg = "/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/OR_20190630_Three_Creek_c3r2_ortho.tif"
outDir = baseDir+"output"




inDir = baseDir+"input/"
tmpDir = baseDir+"tmp"

#basic config stuff during dev
doDeleteModel = False

input = inLrg
outPath = os.path.join(outDir,os.path.basename(input))

#imLrg = cv2.imread(inLrg)
'''
with CodeTimer('imread small'):
    im = cv2.imread(inSmall)

with CodeTimer('load config'):
    cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

with CodeTimer('predict'):
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)

with CodeTimer('draw pred'):
    v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(outPath,out.get_image()[:, :, ::-1])'''
    
###########################
### 2. Train on my data ###
###########################

from detectron2.engine import DefaultTrainer

flight='OR_20190630_Three_Creek'
trainDir = os.path.join(inDir,'datasets','OR_20190630_Three_Creek','train')
trainJson = os.path.join(inDir,'OR_20190630_Three_Creek_train.json')
valJson = os.path.join(inDir,'OR_20190630_Three_Creek_val.json')
valDir = os.path.join(inDir,'datasets','OR_20190630_Three_Creek','val')

for curType in ['train','val','test']:
  curJson = os.path.join(inDir,flight+'_'+curType+'.json')
  curDir = os.path.join(inDir,'datasets',flight,curType)
  register_coco_instances(flight+'_'+curType, {}, curJson,curDir)

#Fine-tune a COCO-pretrained R50-FPN Mask R-CNN model on the dataset. 
#It takes ~6 minutes to train 300 iterations on Colab's K80 GPU, or ~2 minutes on a P100 GPU.

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (flight+'_train',)
cfg.DATASETS.VAL = (flight+'_val',) #anika added - does this do anything?
cfg.DATASETS.TEST = () # no metrics implemented for this dataset (?is this correct?)
cfg.DATALOADER.NUM_WORKERS = 4 #can increase with better computer
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2 #dataset-dependent, try out 4-10, good computer start high (32), just for faster, maaaaybe less accurate
cfg.SOLVER.BASE_LR = 0.0025  # pick a good LR
cfg.SOLVER.MAX_ITER = 10    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512) #lowering can help for prod
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class
cfg.OUTPUT_DIR = outDir
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False #Src: https://github.com/facebookresearch/detectron2/issues/819

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

finalModelPath = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
if os.path.isfile(finalModelPath) and doDeleteModel:
    os.remove(finalModelPath)
    
if not os.path.isfile(finalModelPath): #don't rerun training unless I cleared the old one
  trainer = DefaultTrainer(cfg) 
  trainer.resume_or_load(resume=False)
  trainer.train()
  
#########################################################
### 3. Inference & evaluation using the trained model ###
#########################################################

#First, let's create a predictor using the model we just trained
#Inference should use the config with parameters that are used in training
#Changes for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold (from 0.63)
predictor = DefaultPredictor(cfg)


#When watching it run:
'''
Notes from convo with Jordan:
loss_mask -> should be decreaseing
total loss needs to be below 1
can set callback function to stop at some number of total_loss so you don't have to watching
can also set callback for loss_mask, but not as useful

'''
from detectron2.utils.visualizer import ColorMode

#dataset_dicts = get_balloon_dicts("balloon/val")
metadata = MetadataCatalog.get(flight+'_test')
dataset_dicts = DatasetCatalog.get(flight+'_test')

cfg.DATASETS.TEST = (flight+'_test',)

#for d in dataset_dicts:#random.sample(dataset_dicts, 3):    
with CodeTimer('predict '+ inLrg):# + os.path.basename(d["file_name"])):
  im = cv2.imread(inLrg)
  outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
  v = Visualizer(im[:, :, ::-1],
                 metadata=metadata, 
                 scale=1, 
                 #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
  )
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  predictFileName = os.path.basename(d['file_name'][:-4]+'_predict.png')
  fileOut = os.path.join(outDir,predictFileName)
  
  cv2.imwrite(fileOut,out.get_image()[:, :, ::-1])
    
    






