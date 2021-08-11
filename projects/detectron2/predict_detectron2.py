# ---------------------------------------------------------------------
# Predicting detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
import glob
import time
import torch
import numpy as np
import xarray as xr
from core.utils import arg_parser, get_bands
from core.utils import predict_batch, arr_to_tif
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

setup_logger()
np.random.seed(22)

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def run(cfg):
    """
    Predict imagery using detectron2 framework.
    """
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

    # Get list of files to predict
    if isinstance(cfg.PREDICTOR.PRED_FILENAMES, str):
        cfg.PREDICTOR.PRED_FILENAMES = glob.glob(cfg.PREDICTOR.PRED_FILENAMES)
    print(f'Number of files to predict: {len(cfg.PREDICTOR.PRED_FILENAMES)}')

    # Tterate over files and predict them
    for fname in cfg.PREDICTOR.PRED_FILENAMES:

        # measure execution time
        start_time = time.perf_counter()

        # path + name to store prediction into
        save_image = \
            cfg.OUTPUT_DIR + '/' + fname[:-4].split('/')[-1] + '_pred.tif'

        # --------------------------------------------------------------------------------
        # if prediction is not on directory, start predicting
        # (allows for restarting script if it was interrupted at some point)
        # --------------------------------------------------------------------------------
        if not os.path.isfile(save_image):

            print(f'Starting to predict {fname}')
            # --------------------------------------------------------------------------------
            # Extracting and resizing test and validation data
            # --------------------------------------------------------------------------------
            x_data = xr.open_rasterio(
                fname, chunks=dict(cfg.DATALOADER.DASK_SIZE)
            )
            x_data = get_bands(
                x_data, cfg.INPUT.INPUT_BANDS, cfg.INPUT.OUTPUT_BANDS
            )

            # --------------------------------------------------------------------------------
            # Calculate missing indices
            # --------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------
            # Getting predicted labels
            # --------------------------------------------------------------------------------
            prediction = predict_batch(x_data=x_data, model=model, config=cfg)
            prediction[prediction > 1] = 1
            prediction = prediction.astype(np.int8)  # type to int8

            # --------------------------------------------------------------------------------
            # Generating visualization from prediction
            # --------------------------------------------------------------------------------
            arr_to_tif(raster_f=fname, segments=prediction, out_tif=save_image)
            del prediction

        # This is the case where the prediction was already saved
        else:
            print(f'{save_image} already predicted.')

        print(f'Time: {(time.perf_counter() - start_time)}')


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
    # Set GPU devices
    # ---------------------------------------------------------------------------
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(
        map(str, cfg.SOLVER.CUDA_DEVICES)
    )

    # ---------------------------------------------------------------------------
    # Run the main
    # ---------------------------------------------------------------------------
    run(cfg)
