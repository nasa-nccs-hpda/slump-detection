# Detectron2 for Instance Segmentation of Slumps

Using the detectron2 framework for the task of instance segmentation.

## Table of Contents

1. [Expected Input Data](#Expected_Input_Data)
2. [Configuration Files](#Container_Environment_Installation)
3. [Working Inside a Container](#Working_Inside_Container)
4. [Authors](#Authors)
5. [References](#References)

## Expected Input Data

The following scripts expect data in GeoTIF format. There should be a file with raw data, preferebly TOA corrected, and a file with
labels or the mask to map with. The expected data file can have anywhere from 3 to N number of channels/bands, while the mask file should
have a single channel/band with integer values. Each integer value representing a class to classify.

Example input shape are (5000,5000,6) for data rasters, and (5000,5000) for mask files. Note that both the image and mask files have the same height and width dimensions. Another example could be (3456, 986, 6) and (3456, 986). The information regarding the data will be stored in a configuration file under the config/ directory of this project. In the case where a label file was generated without its matching data file, the following rasterio script can be executed to extract its matching data file.

```bash
rio clip WV02_20160709_M1BS_10300100591D6600-toa_pansharpen.tif trialrun_data.tif --like trialrun_label.tif
```

## Configuration Files

Once data files and labels are available, we can proceed to configure our datasets and models. There are two configuration files under the config/ directory. The Base**.yaml file has the default configurations for the model. The file slump**.yaml has specific configurations targeted to the problem in question. For any changes to the model, feel free to modify the slump**.yaml. A README has been included in the directory for additional information.

The current set of configuration files has the following information:

- data and label files are stored as TIF files under the data/ directory at the top of the project.
- the data file has 8-bands and bands RGB are taken for training
- the datasets generated will be stored in the data/ directory
- the models trained will be stored in the data/ directory
- transfer learning from ImageNet models is applied for faster training
- instance segmentation using Fast Mask RCNN is applied

## Generate Dataset

In this section of the pipeline we take raw raster and masks, and we generate training and validation
datasets to work with. The following script will save NPZ files into local storage with the train and mask
mappings taken from the data.csv file. Parameters are taken from the Configuration File. Expect to find two
new directories on the ROOT_DIR specified on the Configuration file with an NPZ file per raster. These NPZ
files contain the dataset mappings with 'x' for data, and 'y' for the label.

if specified, data will be stanrdardized using local standardization.

```bash
python preprocessing.py
```

## Train

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




## Generate Small Tiles

```bash
python gen_dataset.py -l ../../data/trialrun_label.tif -i ../../data/trialrun_data.tif -o ../../data -b 0 1 2
```

[x] open imagery
[ ] get tiles
[ ] convert to png in directory format

docker build .

## References

[1] Facebook Research, Detectron2: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=HUjkwRsOn1O0