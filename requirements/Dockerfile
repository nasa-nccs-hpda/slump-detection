FROM nvidia/cuda:11.1.1-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

# Ubuntu needs noninteractive to be forced
ENV DEBIAN_FRONTEND noninteractive

# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"

# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"

COPY requirements.txt ./

# Install OS and Python core dependencies
RUN apt-get update && apt-get install -y python3-opencv ca-certificates python3-dev \
    git wget sudo ninja-build vim && \
    ln -sv /usr/bin/python3 /usr/bin/python && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# install dependencies
# See https://pytorch.org/ for other options if you use a different version of CUDA
RUN pip install -r requirements.txt && \
    pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install 'git+https://github.com/facebookresearch/fvcore' && \
    pip install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"

# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo && \
    pip install -e detectron2_repo
