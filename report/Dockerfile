FROM python:3.12.0

MAINTAINER WolfGang Ganglberger, PhD

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-venv python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY caisr_report.py /data/
COPY preprocess /data/preprocess/
COPY report /data/report/

WORKDIR /data

RUN python3 -m venv /caisr_report && \
    source /caisr_report/bin/activate && \
    /caisr_report/bin/pip install --upgrade pip && \
    /caisr_report/bin/pip install -r report/requirements_report.txt

# Set the ENTRYPOINT to run the script automatically
ENTRYPOINT ["/bin/bash", "-c", "source /caisr_report/bin/activate && exec python caisr_report.py"]

