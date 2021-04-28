# Mask RCNN Configuration Files

Additional information can be found in the [detectron2 configuration site](https://detectron2.readthedocs.io/en/latest/modules/config.html#detectron2.config.CfgNode).

## GENERAL

This section includes general configurations that can be extracted as default values.

| Option | Description |
| :---  | :-----      |
| \_BASE\_ | points to a main configuration file that has default configurations     |


## INPUT

This section includes configurations related to the data that is being used in the training and generation of the model.

| Option | Description |
| :---  | :-----      |
|INPUT.INPUT_BANDS| available bands from the original imagery, should be in the same order as they appear in the raster.     |
|INPUT.OUTPUT_BANDS| bands to extract from the imagery (3 for this architecture) any choice from the INPUT.INPUT_BANDS    |
|INPUT.MIN_SIZE_TRAIN| tile size to train on (normally 256, 512, 1024)   |
|INPUT.MAX_SIZE_TRAIN| tile size to train on (normally 256, 512, 1024)   |
|INPUT.MIN_SIZE_TEST| tile size to test on (normally 256, 512, 1024)   |
|INPUT.MAX_SIZE_TEST| tile size to test on (normally 256, 512, 1024)   |
|INPUT.RANDOM_FLIP| random flip data augmentation (vertical, horizontal)  |

## DATASETS

| Option | Description |
| :--- | :-----      |
|DATASETS.IMAGES| a list of images to use to generate the dataset  |
|DATASETS.LABELS| a list of label images, should match DATASETS.IMAGES images  |
|DATASETS.NUM_TRAIN_TILES| number of training tiles to retrieve from each image (n tiles per image) |
|DATASETS.NUM_TEST_TILES| number of test tiles to retrieve from each image (n tiles per image) |
|DATASETS.NUM_VAL_TILES| number of validation tiles to retrieve from each image (n tiles per image) |
|DATASETS.NUM_TRUE_PIXELS| minimum number of true pixels a tile needs to have in order to be included in the dataset |
|DATASETS.OUTPUT_DIRECTORY| directory to store dataset on |
|DATASETS.TRAIN| name of trainining dataset |
|DATASETS.VAL| name of validation dataset |
|DATASETS.COCO_METADATA.DESCRIPTION| dataset name to use when loading the data into the model |
|DATASETS.COCO_METADATA.INFO| small description of the dataset for documentation purposes. Must include: description, an url, the version of the dataset, a year, and a contributor |
|DATASETS.COCO_METADATA.LICENSES| small description of dataset licenses for documentation purposes. Must include: an id, the name of the license, and an url |
|DATASETS.COCO_METADATA.CATEGORIES| set the categories that are being identified by the model. Each class will have a different id|
|DATASETS.COCO_METADATA.CATEGORY_INFO| metadata information about the category |

## SOLVER

| Option | Description |
| :--- | :-----      |
|SOLVER.IMS_PER_BATCH| number of images that will be processed concurrently; this value will depend on the GPU memory available, 16 is normally an ok value  |
|SOLVER.BASE_LR| learning rate for training the model, the higher the learning rate, the more proabable the model will overfit; a value between 0.001-0.0001 is an okay value |
|SOLVER.MAX_ITER| number of iterations to train the model; this value should be higher than 15 in most cases, monitor the validation loss to account for the best value |

## MODEL

| Option | Description |
| :--- | :-----      |
|MODEL.ROI_HEADS.NAME| name for the middle convolutional layers  |
|MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE| batch size per image during training  |
|MODEL.ROI_HEADS.NUM_CLASSES| number of classes to train on; 1 for binary problems  |
|MODEL.OUTPUT_DIRECTORY| directory to store model output  |
|MODEL.MODEL_NAME| model filename to store on disk  |
|MODEL.DELETE_MODEL| defines if the model needs to be deleted on each new run  |

## DATALOADER

| Option | Description |
| :--- | :-----      |
|DATALOADER.FILTER_EMPTY_ANNOTATIONS| remove or not empty annotations from the dataset  |