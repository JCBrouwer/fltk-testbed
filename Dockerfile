FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
ENV TORCH_CUDA_ARCH_LIST="6.1;8.6"

# Who maintains this DockerFile
MAINTAINER Jeroen Galjaard <J.M.Galjaard-1@student.tudelft.nl>

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository and fetch packages
RUN apt-get update \
  && apt-get install -y curl python3 python3-pip net-tools iproute2 git

# Add Pre-downloaded models (otherwise needs be run every-time)
ADD data/ data/

# Use cache for pip, otherwise we repeatedly pull from repository
# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install numpy Cython cmake ninja
RUN python3 -m pip install numpy Cython cmake ninja
ADD requirements.txt ./
# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install torch==1.9+cu111 torchvision==0.10+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install torch==1.9+cu111 torchvision==0.10+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt
RUN python3 -m pip install -r requirements.txt

# Add FLTK and configurations
ADD fltk fltk
ADD configs configs
ADD charts charts

RUN python3 -m fltk.nets.gan.compile