# ---------------------------------------------------------------------
# Predicting detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
import time
import numpy as np
import xarray as xr
from core.utils import arg_parser, get_bands
from core.utils import predict_batch, arr_to_tif

# import some common detectron2 utilities
import torch
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg

setup_logger()


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
    # TODO: if type is string, glob.glob, else you return the list
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
            prediction = prediction.astype(np.int8)  # type to int16

            # --------------------------------------------------------------------------------
            # Generating visualization from prediction
            # --------------------------------------------------------------------------------
            arr_to_tif(raster_f=fname, segments=prediction, out_tif=save_image)
            # np.save(save_segment, prediction)
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
    print(cfg.SOLVER.CUDA_DEVICES, f'{cfg.SOLVER.CUDA_DEVICES}')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

    # ---------------------------------------------------------------------------
    # Run the main
    # ---------------------------------------------------------------------------
    run(cfg)
