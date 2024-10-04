FROM python:3.9

MAINTAINER Erik-Jan Meulenbrugge

SHELL ["/bin/bash", "-c"]

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python3-venv python3-tk && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY caisr_arousal.py /data/
COPY arousal /data/arousal/

WORKDIR /data

RUN python3 -m venv /caisr_arousal && \
    source /caisr_arousal/bin/activate && \
    /caisr_arousal/bin/pip install --upgrade pip && \
    /caisr_arousal/bin/pip install -r arousal/arousal_requirements.txt

# Set the ENTRYPOINT to run the script automatically
ENTRYPOINT ["/bin/bash", "-c", "source /caisr_arousal/bin/activate && exec python caisr_arousal.py"]

