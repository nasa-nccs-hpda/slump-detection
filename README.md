# Slump Detection

Slump Detection as an instance segmentation problem.

## Create Environment

```bash
```

## Starting Container

```bash
singularity shell --nv -B /gpfsm/ccds01/nobackup/temp/jacaraba:/gpfsm/ccds01/nobackup/temp/jacaraba,/att/nobackup/jacaraba:/att/nobackup/jacaraba /gpfsm/ccds01/nobackup/temp/jacaraba/slump-detection/requirements/slump-detectron2_latest.sif
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
