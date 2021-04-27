# ---------------------------------------------------------------------
# Training detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
import cv2
import matplotlib.pyplot as plt
from core.utils import arg_parser

# import some common detectron2 utilities
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

    # Path and directory configurations
    input_dir = cfg.DATASET.OUTPUT_DIRECTORY
    cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    dataset_name = cfg.DATASET.COCO_METADATA.DESCRIPTION

    # Registor COCO datasets for train, val, and test
    for curType in ['TRAIN', 'VAL', 'TEST']:
        curJson = os.path.join(
            input_dir, dataset_name + '_' + curType + '.json'
        )
        curDir = os.path.join(input_dir, curType)
        register_coco_instances(
            f'{dataset_name}_{curType}', {}, curJson, curDir
        )

    # Setup configurations
    #cfg = get_cfg()
    #cfg.merge_from_file(
    #    model_zoo.get_config_file(
    #        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #    )
    #)
    #cfg.DATASETS.TRAIN = (dataset_name + '_train')
    #cfg.DATASETS.VAL = (dataset_name + '_val')
    #cfg.DATASETS.TEST = ()

    #cfg.DATALOADER.NUM_WORKERS = 4
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    #    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    #)
    #cfg.SOLVER.IMS_PER_BATCH = 2
    #cfg.SOLVER.BASE_LR = 0.0025
    #cfg.SOLVER.MAX_ITER = 300
    #cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    #cfg.INPUT.MIN_SIZE_TRAIN = (256,)
    #cfg.INPUT.MAX_SIZE_TRAIN = (256,)
    ##cfg.INPUT.MIN_SIZE_TEST = (256,)
    ##cfg.INPUT.MAX_SIZE_TEST = (256,)
    ##cfg.INPUT.RANDOM_FLIP = "vertical"

    #cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    #cfg.MODEL.MASK_ON = True
    #cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    #cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    finalModelPath = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)

    if os.path.isfile(finalModelPath) and cfg.MODEL.DELETE_MODEL:
        os.remove(finalModelPath)
        print("WARNING: deleted existing model, re-training")

    if not os.path.isfile(finalModelPath):  # no rerun training unless cleared
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()

    """
    #########################################################
    # 3. Inference & evaluation using the trained model ###
    #########################################################

    # First, let's create a predictor using the model we just trained
    # Inference should use the config with parameters that are used in training
    # Changes for inference:

    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom threshold

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(dataset_name + '_test')
    dataset_dicts = DatasetCatalog.get(dataset_name + '_test')
    #cfg.DATASETS.TEST = (dataset_name + '_test')

    inLrg = '../../data/test/trialrun_data_d_20.png'

    #for d in random.sample(dataset_dicts, 3):
    #    im = cv2.imread(d["file_name"])
    #    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    #    v = Visualizer(
    #            im[:, :, ::-1],
    #            metadata=metadata, scale=0.5,
    #            instance_mode=ColorMode.IMAGE_BW  # remove the colors of unsegmented pixels. This option is only available for segmentation models
    #    )
    #    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #    #cv2_imshow(out.get_image()[:, :, ::-1])
    #    cv2.imwrite(fileOut,out.get_image()[:, :, ::-1])

    im = cv2.imread(inLrg)
    print(im.shape)
    #print(type(im))
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
            scale=1,
            #instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    predictFileName = os.path.basename('predict.png')
    fileOut = os.path.join(outDir,predictFileName)

    cv2.imwrite(fileOut,out.get_image()[:, :, ::-1])
    """

    """
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer

    model = build_model(cfg)
    #print(model)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    print(model.eval())
    #model.eval()
    """

    """
    #When watching it run:
    '''
    Notes from convo with Jordan:
    loss_mask -> should be decreaseing
    total loss needs to be below 1
    can set callback function to stop at some number of total_loss so you don't have to watching
    can also set callback for loss_mask, but not as useful
    '''

    # dataset_dicts = get_balloon_dicts("balloon/val")
    metadata = MetadataCatalog.get(dataset_name + '_test')
    dataset_dicts = DatasetCatalog.get(dataset_name + '_test')

    cfg.DATASETS.TEST = (dataset_name + '_test')

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
    """


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
