# Building Docker Container

This repository includes a Docker file to build the container. The base image is taken
from NVIDIA DockerHub container repository with the CUDA stack installed.
In order to build the container image, you will need Docker installed in your system.
The main README of the repository provides additional instructions for downloading and
setting up the image in the ADAPT environment.

```bash
cd slump-detection/requirements/
docker build --tag slump-detection:latest .
docker login
docker tag slump-detection:latest nasanccs/slump-detection:latest
docker push nasanccs/slump-detection
```
