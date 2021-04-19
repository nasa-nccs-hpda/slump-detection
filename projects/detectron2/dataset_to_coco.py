import datetime
import glob
import json
import numpy as np
import os
import pdb
from PIL import Image
import pycococreatortools

from pycocotools.coco import COCO
import pycocotools.mask as maskUtils
import skimage.io as io

import matplotlib 
matplotlib.use('Agg') #Need this until ADAPT gpunodes fix display issue
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..','anikautil')))
import anikautil as anu

#TODO: Args

inputType = 'train' #'train', 'test'

# dataDir = '/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/datasets/OR_20190630_Three_Creek/'
dataDir = '../../data'
prefix = os.path.basename(os.path.normpath(dataDir))
inputTypeDir = os.path.join(dataDir, inputType)
# jsonOut = '/att/gpfsfs/briskfs01/ppl/acartas/git/forest.health/input/OR_20190630_Three_Creek_'+inputType+'.json'
jsonOut = f'slump-detection_trialrun_{inputType}.json'
tmpDir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'tmp')))

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

    trainNames = os.listdir(inputTypeDir)
"""
    images = []
    annotations = []

    pastId = 0
    #go through each image
    for curName in trainNames:
        print(curName)
        curImgFile = os.path.join(inputTypeDir, curName, 'images', prefix+'_' + curName + '.png')
        if not os.path.isfile(curImgFile):
          print('Cant find: ' + curImgFile)
          pdb.set_trace()

        curImg = Image.open(curImgFile)
        curImgId = pastId+1 #make sure it's properly unique
        pastId = curImgId
        curImgInfo = pycococreatortools.create_image_info(curImgId,os.path.basename(curImgFile),curImg.size)

        images.append(curImgInfo)

        #go through annotations
        maskDir = os.path.join(inputTypeDir,curName,'masks')
        maskNames = os.listdir(maskDir)
        for curMaskName in maskNames:
            print('\t'+curMaskName)
            curMaskFile = os.path.join(maskDir,curMaskName)
            curAnnotationId = str(curImgId)+curMaskName[len(prefix+'_'+curName+'_'):-4] #make sure it's properly unique
            classId = 1 #only have dead for now
            isCrowd = False #Could maybe do this later, or for bigger trees
            categoryInfo = {
                'id' : classId,
                'is_crowd' : isCrowd,
            }
            binaryMask = np.asarray(Image.open(curMaskFile).convert('1')).astype(np.uint8)
            annotationInfo = pycococreatortools.create_annotation_info(curAnnotationId,curImgId,categoryInfo,binaryMask,curImg.size,tolerance=2)
            if annotationInfo is not None:
                annotations.append(annotationInfo)
            else:
                print('\t\tNo annotationInfo found for: ' + curMaskName)
                #pdb.set_trace()

    cocoInfo = {
        "info" : INFO,
        "licenses" : LICENSES,
        "categories" : CATEGORIES,
        "images" : images,
        "annotations" : annotations,
    }

    with open(jsonOut,'w') as f:
        f.write(json.dumps(cocoInfo))

#Otherwise just display
example_coco = COCO(jsonOut)

categories = example_coco.loadCats(example_coco.getCatIds())
category_names = [category['name'] for category in categories]
print('Custom COCO categories: \n{}\n'.format(' '.join(category_names)))
category_names = set([category['supercategory'] for category in categories])
print('Custom COCO supercategories: \n{}'.format(' '.join(category_names)))

category_ids = example_coco.getCatIds(catNms=['Dead Tree'])
image_ids = example_coco.getImgIds(catIds=category_ids)
for i in range(len(image_ids)):
  image_data = example_coco.loadImgs(image_ids[i])[0]

  #load and display instance annotations
  image_directory = os.path.join(inputTypeDir,image_data['file_name'][len(prefix+'_'):-4],'images')
  image = io.imread(os.path.join(image_directory,image_data['file_name']))
  plt.imshow(image)
  
  plt.axis('off')
  
  annsDst = os.path.join(tmpDir,image_data['file_name'][:-4]+'_annotated.png')
  pylab.rcParams['figure.figsize'] = (8.0, 10.0)
  annotation_ids = example_coco.getAnnIds(imgIds=image_data['id'], catIds=category_ids, iscrowd=None)
  annotations = example_coco.loadAnns(annotation_ids)
  example_coco.showAnns(annotations)
  #ax=saveAnns(annsDst,example_coco,annotations,draw_bbox=True)
  plt.savefig(annsDst)
  plt.clf()
  plt.cla()
"""