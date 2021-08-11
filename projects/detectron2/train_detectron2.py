# ---------------------------------------------------------------------
# Training detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
from core.utils import arg_parser
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data.datasets import register_coco_instances

setup_logger()
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


def run(cfg):
    """
    Train model using detectron2 framework.
    """
    # Path and directory configurations
    input_dir = cfg.DATASETS.OUTPUT_DIRECTORY
    cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    dataset_name = cfg.DATASETS.COCO_METADATA.DESCRIPTION

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
    finalModelPath = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)

    if os.path.isfile(finalModelPath) and cfg.MODEL.DELETE_MODEL:
        os.remove(finalModelPath)
        print("WARNING: deleted existing model, re-training")

    if not os.path.isfile(finalModelPath):  # no rerun training unless cleared
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()


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
