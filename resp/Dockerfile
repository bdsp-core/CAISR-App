FROM python:3.8.3

MAINTAINER Thijs-Enagnon Nassi, PhD

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-venv python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY caisr_resp.py /data/
COPY resp /data/resp/

WORKDIR /data

RUN python3 -m venv /caisr_resp && \
    source /caisr_resp/bin/activate && \
    /caisr_resp/bin/pip install --upgrade pip && \
    /caisr_resp/bin/pip install -r resp/resp_requirements.txt

# Set the ENTRYPOINT to run the script automatically
ENTRYPOINT ["/bin/bash", "-c", "source /caisr_resp/bin/activate && exec python caisr_resp.py"]

