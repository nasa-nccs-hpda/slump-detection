# --------------------------------------------------------------------------
# Extract dataset tiles from label and imagery rasters.
# --------------------------------------------------------------------------
import argparse
import xarray as xr
from skimage import img_as_ubyte
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
        '-l', action='store', dest='label_filename', type=str,
        help='label raster filename'
    )
    parser.add_argument(
        '-i', action='store', dest='image_filename', type=str,
        help='image raster filename'
    )
    parser.add_argument(
        '-o', action='store', dest='out_dir', type=str,
        help='output directory to store dataset'
    )
    parser.add_argument(
        '-b', action='store', dest='bands', type=int, nargs='+',
        help='band indices to save into png file'
    )
    return parser.parse_args()


def run(config):
    """
    Generate dataset for training detectron2 and/or Mask
    """
    for image, label in zip(config['images'], config['labels']):

        # read input data
        image_data = xr.open_rasterio(image).transpose("y", "x", "band")
        label_data = xr.open_rasterio(label).squeeze().values
        print("Image and label shapes: ", image_data.shape, label_data.shape)

        # drop bands we are not interested in given the ones we actually want
        image_data = get_bands(
            image_data, config['input_bands'], config['output_bands']
        ).values
        print("Image after get_bands: ", image_data.shape, label_data.shape)

        # lower resolution here
        image_data = img_as_ubyte(image_data)

        # transforming 1 to 255 for now to visualize locally
        label_data[label_data == 1] = 255

        # extract tiles from the imagery and generate masks
        gen_data_png(
            image, image_data, label_data, config, set='train'
        )
        gen_data_png(
            image, image_data, label_data, config, set='test'
        )


if __name__ == "__main__":

    # ---------------------------------------------------------------------------
    # Initialize args parser
    # ---------------------------------------------------------------------------
    args = arg_parser()

    config = {
        'images': [
            '../../data/trialrun_data.tif'
        ],
        'labels': [
            '../../data/trialrun_label.tif'
        ],
        'input_bands': [
            'CoastalBlue', 'Blue', 'Green', 'Yellow', 'Red',
            'RedEdge', 'NIR1', 'NIR2'
        ],
        'output_bands': ['Blue', 'Green', 'Red'],
        'tile_size': 256,
        'n_tiles_train': 75,
        'n_tiles_test': 25,
        'n_true_pixels': 0,
        'out_dir': '../../data'
    }

    run(config)
