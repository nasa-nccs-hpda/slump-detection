import glob
import os
import glob
import time
import random
import torch
import numpy as np
import cv2
import xarray as xr
import rioxarray as rxr
from skimage.util import img_as_ubyte
from skimage import exposure
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from core.utils import arg_parser, get_bands, gen_data_png, gen_coco_dataset
from core.utils import predict_batch, arr_to_tif

CONFIG = 'config/slump_mask_rcnn_R_50_FPN_3x.yaml'
PREPROCESS = True
VIS = True
TRAIN = True
PREDICT = True

data_dir = '/att/gpfsfs/briskfs01/ppl/jacaraba/slump-detection-data/TEST/*.png'
data_files = glob.glob(data_dir)

cfg = get_cfg()  # get default configurations in place
cfg.set_new_allowed(True)  # allow for new configuration objects
cfg.INPUT.MIN_SIZE_TRAIN = 256  # small hack to allow merging new fields
cfg.merge_from_file(CONFIG)  # merge from file

# Path and directory configurations
cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# setting up model information
model_weights = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)
cfg.MODEL.WEIGHTS = model_weights

# build model and model metadata
model = build_model(cfg)  # build architecture and maps
model_dict = torch.load(model_weights, map_location=torch.device('cpu'))
model.load_state_dict(model_dict['model'])  # load metadata

# TODO: In a later version, parellize the model over several GPUs.
# import torch.nn as nn
# model = nn.DataParallel(model)
model.train(False)  # we are predicting, weights are already updated

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode

for d in random.sample(data_files, 3):
    print(d)

#    im = cv2.imread(d["file_name"])
#    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
#    v = Visualizer(im[:, :, ::-1],
#                   metadata=balloon_metadata, 
#                   scale=0.5, 
#                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
#    )
#    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
#    cv2_imshow(out.get_image()[:, :, ::-1])
