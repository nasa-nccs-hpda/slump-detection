# Slump Detection

Slump Detection as an instance segmentation problem.

## Create Environment

```bash
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

