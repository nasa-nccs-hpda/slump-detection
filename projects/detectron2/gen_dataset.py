# --------------------------------------------------------------------------
# Extract dataset tiles from label and imagery rasters.
# --------------------------------------------------------------------------
import argparse
import xarray as xr
import numpy as np
from detectron2.config import get_cfg
# from skimage import img_as_ubyte
from core.utils import get_bands, gen_data_png


__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def arg_parser():
    """
    Argparser function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', action='store', dest='config_filename', type=str,
        help='configuration filename', required=True
    )
    return parser.parse_args()


def run(cfg):
    """
    Generate dataset for training detectron2 and/or Mask
    """

    # Process each image individually
    for image, label in zip(cfg.DATASET.IMAGES, cfg.DATASET.LABELS):

        # read input data
        image_data = xr.open_rasterio(image).transpose("y", "x", "band")
        label_data = xr.open_rasterio(label).squeeze().values
        print("Image and label shapes: ", image_data.shape, label_data.shape)

        # drop bands we are not interested in given the ones we actually want
        image_data = get_bands(
            image_data, cfg.INPUT.INPUT_BANDS, cfg.INPUT.OUTPUT_BANDS
        ).values
        print("Image after get_bands: ", image_data.shape, label_data.shape)

        # lower resolution here
        # from skimage.util import img_as_ubyte
        # from skimage import exposure
        # image_data = exposure.rescale_intensity(image_data, out_range=(0, 2**31 - 1))
        # print(image_data)
        # image_data = img_as_ubyte(image_data)

        # transforming 1 to 255 for now to visualize locally
        label_data[label_data == 1] = 255

        # extract tiles from the imagery and generate masks
        gen_data_png(
            image, image_data, label_data, cfg, set='TRAIN'
        )
        gen_data_png(
            image, image_data, label_data, cfg, set='TEST'
        )
        gen_data_png(
            image, image_data, label_data, cfg, set='VAL'
        )


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
