FROM python:3.6.9

MAINTAINER Samaneh Nasiri, PhD

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-venv python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set directory and copy required files and folders
WORKDIR /data
COPY caisr_limb.py /data/caisr_limb.py
COPY limb /data/limb

RUN python3 -m venv /caisr_limb && \
    source /caisr_limb/bin/activate && \
    /caisr_limb/bin/pip install --upgrade pip && \
    /caisr_limb/bin/pip install -r limb/limb_requirements.txt

# Set the ENTRYPOINT to run the script automatically
ENTRYPOINT ["/bin/bash", "-c", "source /caisr_limb/bin/activate && exec python caisr_limb.py"]
