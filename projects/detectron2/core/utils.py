# --------------------------------------------------------------------------
# Utilities directory for slump detection models generation.
# --------------------------------------------------------------------------
import os                            # for os utilities
import sys                           # for os utilities
from tqdm import tqdm                # for progress bar
import argparse                      # for arguments parsing
import datetime                      # for dates manipulation
import glob                          # for local files manipulation
import json                          # for json handling
import random                        # for random integers
import numpy as np                   # for arrays modifications
import imageio                       # for managing images
from core import pycococreatortools  # for coco utilities
from PIL import Image                # for managing images

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


def gen_data_png(fimg, img, label, cfg, set='train'):
    """
    Save png images on disk
    Args:
        fimg (str): list of input bands
        img (np array): array with imagery values
        label (np array): array with labels values
        cfg (CfgNode obj): configuration object
        set (str): dataset to prepare
    """
    # set dimensions of the input image array, and get desired tile size
    y_dim, x_dim, z_dim = img.shape  # dimensions of imagery
    tsz = cfg.INPUT.MIN_SIZE_TRAIN  # tile size to extract from imagery

    n_true_pixels = cfg.DATASETS.NUM_TRUE_PIXELS  # num of true pixels per tile
    fimg = fimg.split('/')[-1][:-4]  # image filename for output
    save_dir = cfg.DATASETS.OUTPUT_DIRECTORY + f'/{set}'  # output directory

    n_tiles = cfg.DATASETS[f'NUM_{set}_TILES']  # number of tiles to extract
    os.system(f'mkdir -p {save_dir}')  # create saving directory

    # generate n number of tiles
    for i in tqdm(range(n_tiles)):

        # Generate random integers from image
        yc = random.randint(0, y_dim - 2 * tsz)
        xc = random.randint(0, x_dim - 2 * tsz)

        # verify data is not on nodata region - maybe later
        # add additional data augmentation in this section
        while np.count_nonzero(
                    label[yc:(yc + tsz), xc:(xc + tsz)] == 255
                ) < n_true_pixels:
            yc = random.randint(0, y_dim - 2 * tsz)
            xc = random.randint(0, x_dim - 2 * tsz)

        # change order to (h, w, c)
        tile_img = img[yc:(yc + tsz), xc:(xc + tsz), :]
        tile_lab = label[yc:(yc + tsz), xc:(xc + tsz)]

        # save png images
        imageio.imwrite(f'{save_dir}/{fimg}_img_{i+1}.png', tile_img)
        imageio.imwrite(f'{save_dir}/{fimg}_lbl_{i+1}.png', tile_lab)


def gen_coco_dataset(
        cfg, set='TRAIN', img_reg='*_img_*.png', label_reg='*_lbl_*.png'
     ):
    """
    Save JSON file with COCO formatted dataset
    Args:
        cfg (CfgNode obj): configuration object
        set (str): dataset to prepare
        img_reg (str): image filename regex
        label_reg (str): label filename regex
    """
    data_dir = cfg.DATASETS.OUTPUT_DIRECTORY  # root directory
    input_dir = os.path.join(data_dir, set)  # directory where images reside
    dataset_name = cfg.DATASETS.COCO_METADATA.DESCRIPTION
    json_out = f'{data_dir}/{dataset_name}_{set}.json'  # output

    if not os.path.isfile(json_out):

        # src: https://patrickwasp.com/create-your-own-coco-style-dataset/
        # Define several sections of the COCO Dataset Format

        # General Information
        INFO = dict(cfg.DATASETS.COCO_METADATA.INFO)
        INFO["date_created"] = datetime.datetime.utcnow().isoformat(' ')

        # Licenses and categories
        LICENSES = [dict(cfg.DATASETS.COCO_METADATA.LICENSES)]
        CATEGORIES = [dict(cfg.DATASETS.COCO_METADATA.CATEGORIES)]
        CATEGORY_INFO = dict(cfg.DATASETS.COCO_METADATA.CATEGORY_INFO)

        # Retrieve filenames from local storage
        train_names = sorted(glob.glob(f'{input_dir}/{img_reg}'))
        mask_names = sorted(glob.glob(f'{input_dir}/{label_reg}'))
        print(f"Number of train and mask images: {len(train_names)}")

        # place holders to store dataset metadata
        images = list()
        annotations = list()
        pastId = 0

        # go through each image
        for curImgName, curMaskName in zip(train_names, mask_names):

            curImgFile = curImgName
            curMaskFile = curMaskName

            # taking care of the images
            curImg = Image.open(curImgFile)
            curImgId = pastId + 1  # make sure it's properly unique
            pastId = curImgId
            curImgInfo = pycococreatortools.create_image_info(
                curImgId, os.path.basename(curImgFile), curImg.size
            )
            images.append(curImgInfo)

            # taking care of the annotations
            curAnnotationId = str(curImgId)
            binaryMask = np.asarray(
                Image.open(curMaskFile).convert('1')
            ).astype(np.uint8)

            annotationInfo = pycococreatortools.create_annotation_info(
                curAnnotationId, curImgId, CATEGORY_INFO, binaryMask,
                curImg.size, tolerance=2
            )

            if annotationInfo is not None:
                annotations.append(annotationInfo)
            else:
                print('\tNo annotationInfo found for: ' + curMaskName)

        coco_info = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": images,
            "annotations": annotations,
        }

        with open(json_out, 'w') as f:
            f.write(json.dumps(coco_info))
    else:
        sys.exit(f'{json_out} already exists. Please remove it and re-run.')
