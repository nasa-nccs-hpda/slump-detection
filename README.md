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

## Table of Contents

1. [Logging-In](#Logging_In)
2. [Container Environment Installation](#Container_Environment_Installation)
3. [Working Inside a Container](#Working_Inside_Container)
4. [Authors](#Authors)
5. [References](#References)

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
singularity pull docker://docker.io/nasanccs/slump-detectron2:11.1
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

## Authors

- Jordan Alexis Caraballo-Vega, <jordan.a.caraballo-vega@nasa.gov>

## References

[1] Chollet, François; et all, Keras, (2015), GitHub repository, https://github.com/keras-team/keras. Accessed 13 February 2020.

[2] Paszke, Adam; Gross, Sam; Chintala, Soumith; Chanan, Gregory; et all, PyTorch, (2016), GitHub repository, https://github.com/pytorch/pytorch. Accessed 13 February 2020.

[3] Google Brain Team; et all, TensorFlow, (2015), GitHub repository, https://github.com/tensorflow/tensorflow. Accessed 13 February 2020.