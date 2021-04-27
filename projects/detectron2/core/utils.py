# --------------------------------------------------------------------------
# Utilities directory for slump detection models generation.
# --------------------------------------------------------------------------
import os
from tqdm import tqdm
import random            # for random integers
import numpy as np       # for arrays modifications
import imageio

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Production"


def get_bands(data, input_bands, output_bands, drop_bands=[]):
    """
    Drop multiple bands to existing rasterio object
    Args:
        input_bands (str list): list of input bands
        output_bands (str list): list of bands to keep
    """
    for ind_id in list(set(input_bands) - set(output_bands)):
        drop_bands.append(input_bands.index(ind_id)+1)
    return data.drop(dim="band", labels=drop_bands, drop=True)


def gen_data_png(fimg, img, label, config, set='train'):

    # set dimensions of the input image array, and get desired tile size
    y_dim, x_dim, z_dim = img.shape
    tsz = config.INPUT.MIN_SIZE_TRAIN
    n_true_pixels = config.DATASET.NUMBER_OF_TRUE_PIXELS
    fimg = fimg.split('/')[-1][:-4]
    save_dir = config.DATASET.OUTPUT_DIRECTORY + f'/{set}'

    print(save_dir)

    """
    os.system(f'mkdir -p {save_dir}')

    # generate n number of tiles
    for i in tqdm(range(config[f'n_tiles_{set}'])):

        # Generate random integers from image
        yc = random.randint(0, y_dim - 2 * tsz)
        xc = random.randint(0, x_dim - 2 * tsz)

        # verify data is not on nodata region - maybe later
        # add data augmentation in this section

        while np.count_nonzero(
                    label[yc:(yc + tsz), xc:(xc + tsz)] == 255
                ) < n_true_pixels:
            yc = random.randint(0, y_dim - 2 * tsz)
            xc = random.randint(0, x_dim - 2 * tsz)

        # change order to (h, w, c)
        tile_img = img[yc:(yc + tsz), xc:(xc + tsz), :]
        tile_lab = label[yc:(yc + tsz), xc:(xc + tsz)]

        imageio.imwrite(f'{save_dir}/{fimg}_d_{i+1}.png', tile_img)
        imageio.imwrite(f'{save_dir}/{fimg}_l_{i+1}.png', tile_lab)

        #os.system(f'mkdir -p {save_dir}/{fimg}_d_{i}/images {save_dir}/{fimg}_d_{i}/masks')
        #imageio.imwrite(f'{save_dir}/{fimg}_d_{i}/images/{fimg}_d_{i}.png', tile_img)
        #imageio.imwrite(f'{save_dir}/{fimg}_d_{i}/masks/{fimg}_l_{i}.png', tile_lab)
    """