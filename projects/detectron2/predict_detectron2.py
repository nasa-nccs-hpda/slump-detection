# ---------------------------------------------------------------------
# Predicting detectron2 model for the task of instance segmentation.
# ---------------------------------------------------------------------
import os
import cv2
import time
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from core.utils import arg_parser, get_bands, predict_batch

# import some common detectron2 utilities
import torch
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import ColorMode

setup_logger()

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
    cfg.OUTPUT_DIR = cfg.MODEL.OUTPUT_DIRECTORY
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # setting up model information
    model_weights = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.MODEL_NAME)
    cfg.MODEL.WEIGHTS = model_weights

    # build model and model metadata
    model = build_model(cfg)  # build architecture and maps

    import torch.nn as nn

    model_dict = torch.load(model_weights, map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['model'])  # load metadata

    #model = nn.DataParallel(model)


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
            # x_data = x_data.transpose("y", "x", "band")
            print(x_data.shape, type(x_data), x_data.sizes)            

            # --------------------------------------------------------------------------------
            # Calculate missing indices
            # --------------------------------------------------------------------------------

            # --------------------------------------------------------------------------------
            # Getting predicted labels
            # --------------------------------------------------------------------------------
            # ME QUEDE AQUI
            prediction = predict_batch(x_data=x_data, model=model, config=cfg)
            
            #print("Prediction shape", prediction.shape, prediction.min(), prediction.max())

            """
            prediction = np.squeeze(prediction)
            prediction[prediction < 0.90] = 0.0
            prediction[prediction > 0.0] = 1.0

            prediction = prediction.astype(np.int8)  # type to int16

            # --------------------------------------------------------------------------------
            # Generating visualization from prediction
            # --------------------------------------------------------------------------------
            arr_to_tif(raster_f=fname, segments=prediction, out_tif=save_image)
            np.save(save_segment, prediction)
            del prediction 
            """

        # This is the case where the prediction was already saved
        else:
            print(f'{save_image} already predicted.')

        print(f'Time: {(time.perf_counter() - start_time)}')

    """
    inLrg = '../../data/TEST/trialrun_data_img_20.png'
    img = np.transpose(cv2.imread(inLrg), (2, 0, 1))
    img_tensor = torch.from_numpy(img)

    inputs = [
        {"image": img_tensor}, {"image": img_tensor}, {"image": img_tensor}, 
        {"image": img_tensor}, {"image": img_tensor}, {"image": img_tensor}, {"image": img_tensor},
        {"image": img_tensor}, {"image": img_tensor}, {"image": img_tensor}
    ]

    outputs = model(inputs)
    print(outputs)

    predictFileName = os.path.basename('predict.png')
    file_out = os.path.join(cfg.OUTPUT_DIR, predictFileName)
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
    # Set GPU devices
    # ---------------------------------------------------------------------------
    print(cfg.SOLVER.CUDA_DEVICES, f'{cfg.SOLVER.CUDA_DEVICES}')
    os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"

    # ---------------------------------------------------------------------------
    # Run the main
    # ---------------------------------------------------------------------------
    run(cfg)
