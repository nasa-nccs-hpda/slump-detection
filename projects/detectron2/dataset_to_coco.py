import os
import sys
import datetime
import glob
import json
import numpy as np
from PIL import Image
import pycococreatortools

__author__ = "Jordan A Caraballo-Vega, Science Data Processing Branch"
__email__ = "jordan.a.caraballo-vega@nasa.gov"
__status__ = "Development"


def run():

    inputType = 'test'  # 'train', 'test'
    dataDir = '../../data'
    jsonOut = f'slump-detection_trialrun_{inputType}.json'

    inputTypeDir = os.path.join(dataDir, inputType)
    tmpDir = os.path.join(
        os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp'))
    )

    if not os.path.isfile(jsonOut):

        # src: https://patrickwasp.com/create-your-own-coco-style-dataset/
        INFO = {
            "description": "Slump Detection in World View VHR Imagery",
            "url": "https://www.nccs.nasa.gov",
            "version": "0.1.0",
            "year": "2021",
            "contributor": "Jordan A. Caraballo-Vega",
            "date_created": datetime.datetime.utcnow().isoformat(' ')
        }

        LICENSES = [
            {
                "id": 1,
                "name": "World View Maxar Agreement",
                "url": "https://www.nccs.nasa.gov"
            }
        ]

        CATEGORIES = [
            {
                "id": 1,
                "name": "Slump",
                "supercategory": "slump"
            }
        ]

        classId = 1  # only have slumps for now
        isCrowd = False
        categoryInfo = {
            'id': classId,
            'is_crowd': isCrowd,
        }

        trainNames = sorted(glob.glob(inputTypeDir + '/*_d_*.png'))
        maskNames = sorted(glob.glob(inputTypeDir + '/*_l_*.png'))
        print(f"Number of train and mask images: {len(trainNames)}")

        images = []
        annotations = []

        pastId = 0

        # go through each image
        for curImgName, curMaskName in zip(trainNames, maskNames):

            curImgFile = curImgName
            curMaskFile = curMaskName
            print(curImgFile, curMaskFile)

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
                curAnnotationId, curImgId, categoryInfo, binaryMask,
                curImg.size, tolerance=2
            )

            if annotationInfo is not None:
                annotations.append(annotationInfo)
            else:
                print('\t\tNo annotationInfo found for: ' + curMaskName)

        cocoInfo = {
            "info": INFO,
            "licenses": LICENSES,
            "categories": CATEGORIES,
            "images": images,
            "annotations": annotations,
        }

        with open(jsonOut, 'w') as f:
            f.write(json.dumps(cocoInfo))
    else:
        sys.exit(f'{jsonOut} already exists. Please remove it.')


if __name__ == "__main__":

    run()
