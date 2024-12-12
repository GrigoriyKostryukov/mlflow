FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04 as base

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -yqq && apt-get install -yqq software-properties-common

RUN apt-get update -yqq && \
    apt-get install -yqq software-properties-common wget curl && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update -yqq && \
    apt-get install -yqq python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN python3.10 -m pip install --upgrade pip
RUN python3 -m pip install --no-cache-dir torch==2.0.0+cu118 torchvision==0.15.1+cu118 --index-url https://download.pytorch.org/whl/cu118

COPY requirements.txt /tmp/app_r.txt
RUN python3 -m pip install -r /tmp/app_r.txt --no-cache-dir \
    && rm -rf /root/.cache \
    && rm /tmp/app_r.txt

WORKDIR /src
# Install Jupyter Notebook
# RUN apt-get update && apt-get install -y jupyter-notebook && apt-get clean

# CMD ["python", "app/main.py"]
