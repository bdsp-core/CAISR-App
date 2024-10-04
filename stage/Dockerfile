FROM ubuntu:20.04
#ARG CUDA_VERSION=12.01
#FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04
# An ARG declared before a FROM is outside of a build stage,
# so it can’t be used in any instruction after a FROM

#ARG PY_VER=3.6.10
#ARG TF_VER=2.5.0
#ARG CUDA_VERSION

# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        sudo \
        locales \
        openssh-server \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Install miniconda (python)
# Referenced PyTorch's Dockerfile:
#   https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile

# Install Miniconda (Python)
ARG CONDA_VER=latest
ARG OS_TYPE=x86_64
ARG PY_VER=3.6.10

RUN curl -LO "http://repo.continuum.io/miniconda/Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh"
RUN bash Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh -p /miniconda -b
RUN rm Miniconda3-${CONDA_VER}-Linux-${OS_TYPE}.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Set directory and copy required files and folders
WORKDIR /data
COPY caisr_stage.py /data/caisr_stage.py
COPY stage /data/stage

# Pull the environment name out of the environment.yml
RUN conda env create -f /data/stage/caisr_stage.yml
RUN echo "source activate $(head -1 /data/stage/caisr_stage.yml | cut -d' ' -f2)" > ~/.bashrc
ENV CONDA_DEFAULT_ENV $(head -1 /data/stage/caisr_stage.yml | cut -d' ' -f2)
ENV PATH /opt/conda/envs/${CONDA_DEFAULT_ENV}/bin:$PATH

# Set the ENTRYPOINT to run the script automatically when booting the docker
ENTRYPOINT ["/bin/bash", "-c", "source ~/.bashrc && conda activate ${CONDA_DEFAULT_ENV} && exec python caisr_stage.py"]
