FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

# Run build without interactive dialogue
ARG DEBIAN_FRONTEND=noninteractive

# Define the working directory of the current Docker container
WORKDIR /opt/federation-lab

# Update the Ubuntu software repository and fetch packages
RUN apt-get update && apt-get install -y curl python3 python3-pip net-tools iproute2 git

# Use cache for pip, otherwise we repeatedly pull from repository
RUN python3 -m pip install numpy Cython cmake ninja
RUN python3 -m pip install torch==1.9+cu111 torchvision==0.10+cu111 -f https://download.pytorch.org/whl/torch_stable.html
ADD requirements.txt ./
RUN python3 -m pip install -r requirements.txt

# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install numpy Cython cmake ninja
# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install torch==1.9+cu111 torchvision==0.10+cu111 -f https://download.pytorch.org/whl/torch_stable.html
# ADD requirements.txt ./
# RUN --mount=type=cache,target=/root/.cache/pip python3 -m pip install -r requirements.txt

# Add FLTK and configurations
ADD fltk fltk
ADD configs configs
ADD charts charts
ADD allresults.csv allresults.csv
RUN sed -i 's|client.V1DeleteOptions()||g' /usr/local/lib/python3.8/dist-packages/kubeflow/pytorchjob/api/py_torch_job_client.py
# ENV TORCH_CUDA_ARCH_LIST="8.6;6.1"
# RUN python3 -m fltk.nets.gan.compile