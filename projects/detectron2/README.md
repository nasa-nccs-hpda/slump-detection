# Detectron2 for Instance Segmentation of Slumps

Using the detectron2 framework for the task of instance segmentation.

## Table of Contents

1. [Expected Input Data](#Expected_Input_Data)
2. [Configuration File](#Container_Environment_Installation)
3. [Working Inside a Container](#Working_Inside_Container)
4. [Authors](#Authors)
5. [References](#References)

## Expected Input Data

The following scripts expect data in GeoTIF format. There should be a file with raw data, preferebly TOA corrected, and a file with
labels or the mask to map with. The expected data file can have anywhere from 3 to N number of channels/bands, while the mask file should
have a single channel/band with integer values. Each integer value representing a class to classify. At the end of the day, the directories
should look similar to the following example:

```bash
/att/data/data01.tif, /att/data/data02.tif, /att/data/data03.tif
/att/labels/labels01.tif, /att/labels/labels02.tif, /att/labels/labels03.tif
```

Example input shape are (5000,5000,6) for data rasters, and (5000,5000) for mask files. The information regarding the data will be stored
in a CSV file located under scripts/config/data.csv, with the following format:

```bash
data,label,ntiles_train,ntiles_val,ymin,ymax,xmin,xmax
Keelin00_20120130_data.tif,cm_Keelin00_20120130_new_2.tif,500,1000,0,50,0,50
Keelin00_20180306_data.tif,cm_Keelin00_20180306_new_2.tif,500,1000,0,50,0,50
Keelin01_20151117_data.tif,cm_Keelin01_20151117_new.tif,500,1000,0,50,0,50
```

Where:

* data: is the filename with the data values
* label: is the filename with the mask values
* ntiles_train: number of training tiles to extract from the data file
* ntiles_val: number of validation tiles to extract from the data file
* ymin,ymax,xmin,xmax: values to exclude from training dataset to include in validation tiles

## Configuration File

### Preprocess

In this section of the pipeline we take raw raster and masks, and we generate training and validation
datasets to work with. The following script will save NPZ files into local storage with the train and mask
mappings taken from the data.csv file. Parameters are taken from the Configuration File. Expect to find two
new directories on the ROOT_DIR specified on the Configuration file with an NPZ file per raster. These NPZ
files contain the dataset mappings with 'x' for data, and 'y' for the label.

if specified, data will be stanrdardized using local standardization.

```bash
python preprocessing.py
```

### Train

In this section of the pipeline we proceed to train the model. Please refer to the Configuration file for more
details on parameters required for training. The main script will: read the data files, map them into TensorFlow
datasets, initialize the UNet model, and proceed with training. A model (.h5 file) will be saved for each epoch
that improved model performance.

You might notice that the first epoch of the model takes a long time to train. This is expected since the model
is being optimized during the first epoch. Distributed training across multiple GPUs is enabled, together with
mixed precission for additional performance enhancements. Upcoming modifications will include A100 performance
enhancements and Horovod for multi-node training.

```bash
python train.py
```

### Predict

In this section we proceed to train the desired data taking one of the saved models, and using it for predictions.
Please refer to the Configuration file for more details on parameters required for predicting. Ideally, all images to
predict will be stored in the same directory ending on .tif extensions. The script will go through every file, predict
it and output both the predicted GeoTIF and the probabilities arrays.

```bash
python predict.py
```


## Clip Data to Match Labels

```bash
rio clip WV02_20160709_M1BS_10300100591D6600-toa_pansharpen.tif trialrun_data.tif --like trialrun_label.tif
```

## Generate Small Tiles

```bash
python gen_dataset.py -l ../../data/trialrun_label.tif -i ../../data/trialrun_data.tif -o ../../data -b 0 1 2
```

[x] open imagery
[ ] get tiles
[ ] convert to png in directory format

docker build .

## References
