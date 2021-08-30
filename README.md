# Slump Detection

Slump Detection as an instance segmentation problem.

## Business Case

The following repository stores several experiments for the task of instance and semantic
segmentation of slumps in very high-resolution satellite imagery. Many of the instructions
listed below are guided towards utilizing GSFC NASA Center for Climate Simulation (NCCS)
computing resources, particularly the PRISM GPU cluster.

A system with NVIDIA GPUs is required to run the scripts located in this repository.

- projects/detectron2: utilizes the detectron2 framework for the task of instance segmentation
leveraging MaskRCNN and Fast RCNN. The backend engine is PyTorch.

## Summarized Steps

```bash
ssh adaptlogin.nccs.nasa.gov
ssh gpulogin1
cd $NOBACKUP/slump-detection/projects/detectron2
```

## Table of Contents

1. [Logging-In](#Logging_In)
2. [Container Environment Installation](#Container_Environment_Installation)
3. [Working Inside a Container](#Working_Inside_Container)
4. [Getting Started](#Getting_Started)
5. [Authors](#Authors)
6. [References](#References)

## Logging-In <a name="Logging_In"></a>

You will need an activate NCCS account together with a PIV Card or an RSA Token. Please refer
to the following link for instructions on setting up login or any login related questions:
[NCCS Logging-In](https://www.nccs.nasa.gov/nccs-users/instructional/logging-in/bastion-host).
Once you are all setup, you may login to the PRISM GPU cluster.

```bash
ssh adaptlogin.nccs.nasa.gov
ssh gpulogin1
```

## Container Environment Installation <a name="Container_Environment_Installation"></a>

All the software and scripts from this repository can be ran within a container. Containers are
small versions of operating systems that are meant to speed up the process of software development.
These containers are simply a binary file which has all the executables needed to run the software included.

The NCCS provides Singularity as the default container runtime tool. In order to configure your
environment to run Singularity containers, you will need to setup the environment variables listed below.
For this, you can simply add the following lines to your ~/.bashrc file.

```bash
echo "export SINGULARITY_CACHEDIR=$NOBACKUP/.singularity" >> ~/.bashrc
echo "export SINGULARITY_TMPDIR=$NOBACKUP/.singularity" >> ~/.bashrc
source ~/.bashrc
```

Test the environment variables with the following command:

```bash
[username@gpulogin1 ~]$ echo $SINGULARITY_CACHEDIR $SINGULARITY_TMPDIR
/att/nobackup/username/.singularity /att/nobackup/username/.singularity
```

In order to utilize the container for this project, we first need to download the image from a container
registry. The image for this project is located in [NASA NCCS DockerHub Repository](https://hub.docker.com/repository/docker/nasanccs/slump-detectron2). Docker containers can be pulled as Singularity containers to be executed on HPC
environments. The following commands allow the download of the container from DockerHub and generates a
file with a .sif extension. Depending on the file system, this step can take several minutes.

```bash
cd $NOBACKUP
module load singularity
singularity pull docker://docker.io/nasanccs/slump-detectron2:latest
singularity build --sandbox slump-detectron2_latest slump-detectron2_latest.sif
```

## Working Inside a Container <a name="Working_Inside_Container"></a>

Each project provides a set of Slurm scripts that will execute code inside the container without having
to login inside the image. You may skip this step and go straight to the project README if you are only
interested in running scripts from outside the container. This section is meant to help users developing
and testing code inside the container to facilitate the development process.

To get a session in one of the PRISM GPU nodes, you can run the following command. Additional instructions
regarding Slurm can be found in the [NCCS website](https://www.nccs.nasa.gov/nccs-users/instructional/adapt-instructional/using-prism).

```bash
salloc
```

You will notice that the hostname will change to something similar to gpu***. This means that you are now
logged into one of the GPU nodes. To access the container image, you can run the command listed below:

```bash
singularity shell --nv -B /att/nobackup/username:/att/nobackup/username slump-detectron2_latest.sif
```

where username is your NASA auid. From here, you can run any command inside the container image. Note that
for Singularity containers to have access to other paths within the HPC environment, we need to bind
directories to particular locations in the container. The command above is binding your $NOBACKUP directory
to be visible from inside the container.

## Getting Started <a name="Getting_Started"></a>

The following is a summarized set of steps to get started and running in less than 5 minutes once the container image has been downloaded.

1. Clone this repository into your ADAPT space

```bash
cd $NOBACKUP
git clone https://github.com/jordancaraballo/slump-detection.git
```

2. Copy the data into the data/ directory

```bash
cp /data/location/.tif $NOBACKUP/slump-detection/data
```

3. Generate train, test, and validation datasets

```bash
cd $NOBACKUP/slump-detection/projects/detectron2
sbatch gen_dataset.sh
```

4. Train a new model

```bash
cd $NOBACKUP/slump-detection/projects/detectron2
sbatch train_detectron2.sh
```

5. Classify given imagery

```bash
cd $NOBACKUP/slump-detection/projects/detectron2
sbatch predict_detectron2.sh
```

## Project Specific Information

Data resides under:

``` bash
/att/nobackup/mwooten3/EVHR_requests/_deliver/EWebbRequest
```

```bash
[1:27 PM] Caraballo-Vega, Jordan Alexis (GSFC-5870)
ssh adaptlogin

[1:27 PM] Caraballo-Vega, Jordan Alexis (GSFC-5870)
ssh gpulogin1

[1:28 PM] Caraballo-Vega, Jordan Alexis (GSFC-5870)
module load anaconda

[1:28 PM] Caraballo-Vega, Jordan Alexis (GSFC-5870)
conda create --name slump-detection-11.1 --clone /att/nobackup/jacaraba/.conda/envs/slump-detection-11.1


conda create --name slump-detection-11.1 --clone /att/nobackup/jacaraba/.conda/envs/slump-detection-11.1
```

## Anaconda environment

```bash
module load anaconda
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -n slump-detection rioxarray cupy cudatoolkit=11.2 dask-cuda cudnn cutensor nccl ipykernel ipywidgets matplotlib geopandas iteration_utilities
```

Install pip dependencies

```bash
conda activate slump-detection
pip install -r requirements.txt
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
git clone https://github.com/facebookresearch/detectron2 detectron2_repo && pip install -e detectron2_repo
```
I think that we need to add nvcc


conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge     rapids-blazing=21.06 python=3.7 cudatoolkit=11.2 nvcc_linux-64 nccl ipykernel ipywidgets matplotlib geopandas iteration_utilities


conda create -n rapids-21.06 -c rapidsai -c nvidia -c conda-forge -c pytorch rapids-blazing=21.06 python=3.7 cudatoolkit=11.1 ipykernel ipywidgets matplotlib geopandas pytorch torchvision torchaudio cudatoolkit=11.1 

## =======================

Trying this one

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
conda create -y -n slump-detection-11.1 rioxarray cupy cudatoolkit=11.1 dask-cuda cudnn cutensor nccl ipykernel ipywidgets matplotlib geopandas iteration_utilities gcc_linux-64
```

```bash
pip install cython
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install 'git+https://github.com/facebookresearch/fvcore'
pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
pip install opencv-python scikit-image
```

## Authors

- Jordan Alexis Caraballo-Vega, <jordan.a.caraballo-vega@nasa.gov>

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, https://github.com/keras-team/keras. Accessed 13 February 2020.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, https://github.com/pytorch/pytorch. Accessed 13 February 2020.

[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, https://github.com/tensorflow/tensorflow. Accessed 13 February 2020.

