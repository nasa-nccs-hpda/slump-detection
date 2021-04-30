# ---------------------------------------------------------------------
# Training detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from core.utils import arg_parser

# import some common detectron2 utilities
import torch
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

setup_logger()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def run(cfg):
    """
    Train model using detectron2 framework.
    """
    """
    # Path and directory configurations
    input_dir = cfg.DATASETS.OUTPUT_DIRECTORY
    cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    dataset_name = cfg.DATASETS.COCO_METADATA.DESCRIPTION
    model_name = cfg.MODEL.MODEL_NAME

    # Registor COCO datasets for train, val, and test
    for curType in ['TRAIN', 'VAL', 'TEST']:
        curJson = os.path.join(
            input_dir, dataset_name + '_' + curType + '.json'
        )
        curDir = os.path.join(input_dir, curType)
        register_coco_instances(
            f'{dataset_name}_{curType}', {}, curJson, curDir
        )

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    predictor = DefaultPredictor(cfg)
    
    metadata = MetadataCatalog.get(dataset_name + '_TEST')
    dataset_dicts = DatasetCatalog.get(dataset_name + '_TEST')

    inLrg = '../../data/TEST/trialrun_data_img_20.png'

    im = cv2.imread(inLrg)
    print(im.shape)
    #print(type(im))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    print(type(outputs))
    print(type(outputs['instances']))

    v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=1,
            #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    #print(metadata)
    #out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #out = v.draw_binary_mask(outputs["instances"].to("cpu"))
    #out = v.draw_sem_seg(outputs["instances"].to("cpu"))
    predictFileName = os.path.basename('predict.png')
    file_out = os.path.join(cfg.OUTPUT_DIR, predictFileName)

    #cv2.imwrite(file_out, out.get_image()[:, :, ::-1])

    bin_mask= np.zeros((256, 256))

    for bin in outputs['instances'].pred_masks.to('cpu'):
         print('new')
         print(type(bin))
         print(np.unique(bin.numpy().astype(int)))
         v.draw_binary_mask(bin.numpy(), color='white', edge_color='white')
         bin_mask += bin.numpy()
         print(np.unique(bin_mask))
    v._create_grayscale_image()
    out = v.get_output()
    cv2.imwrite(file_out, out.get_image()[:, :, ::-1])
    bin_mask[bin_mask > 0] = 255
    cv2.imwrite('pp.png', bin_mask)
    """

    # Path and directory configurations
    input_dir = cfg.DATASETS.OUTPUT_DIRECTORY
    cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    dataset_name = cfg.DATASETS.COCO_METADATA.DESCRIPTION
    model_name = cfg.MODEL.MODEL_NAME

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

    model = build_model(cfg)

    model_dict = torch.load(os.path.join(cfg.OUTPUT_DIR, "model_final.pth"), map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['model'] )
    model.train(False)

    inLrg = '../../data/TEST/trialrun_data_img_20.png'
    img = np.transpose(cv2.imread(inLrg),(2,0,1))
    img_tensor = torch.from_numpy(img)

    inputs = [{"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}, {"image":img_tensor}]

    outputs = model(inputs)
    print(outputs)


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # Initialize args parser
    # ---------------------------------------------------------------------------
    args = arg_parser()

    # ---------------------------------------------------------------------------
    # Initialize configuration object
    # ---------------------------------------------------------------------------
    cfg = get_cfg()  # get default configurations in place
    cfg.set_new_allowed(True)  # allow for new configuration objects
    cfg.INPUT.MIN_SIZE_TRAIN = 256  # small hack to allow merging new fields
    cfg.merge_from_file(args.config_filename)  # merge from file

    # ---------------------------------------------------------------------------
    # Run the main
    # ---------------------------------------------------------------------------
    run(cfg)
