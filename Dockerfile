FROM ubuntu:22.04
# FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
FROM tensorflow/tensorflow:latest-gpu
#  
# -- if you are using docker behind proxy, please set ENV --
#

ENV http_proxy "http://proxy.anan-nct.ac.jp:8080/"
ENV https_proxy "http://proxy.anan-nct.ac.jp:8080/"

ENV DEBIAN_FRONTEND nonineractive
ENV PATH=/usr/local/cuda/bin:/usr/local/nvidia/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/nvidia/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# -- build step --
#LABEL Description "Python 3 Tensorflow image with GPU support and Keras"
#LABEL Vendor "Matteo Ragni"
#LABEL maintainer "info@ragni.me"
#LABEL Version "1.0"

RUN pip install --upgrade pip
RUN apt-get update
RUN apt-get install -y libgl1-mesa-dev

# Keras dependencies installation:
#  - hdf5 : save and restore trained weights
#  - graphviz, pydot : display the model
RUN apt-get install -y --no-upgrade --no-install-recommends  \
  graphviz \
  libtiff5 \
  hdf5-helpers \
  hdf5-tools \
  python3-hdf5storage \
  python3-pydot \
  python3-tk

RUN pip install matplotlib keras pandas scikit-learn ipdb
RUN mkdir /nn
RUN mkdir /log

EXPOSE 8888
EXPOSE 6006
ENV KERAS_BACKEND tensorflow

WORKDIR /src
CMD (tensorboard --logdir=/log --port=6006 --host=0.0.0.0 --purge-orphaned-data >/dev/null 2>/dev/null &) && \
    (jupyter notebook --port=8888 --ip=0.0.0.0 >/tmp/jupyter.stdout 2>/tmp/jupyter.stderr &) && \
    bash
RUN pip install -q Pillow
RUN apt -y install vim
RUN pip install opencv-contrib-python
RUN pip install hyperas
RUN pip install optuna
RUN pip install pdf2image
RUN apt-get install -y poppler-utils poppler-data
RUN apt-get install -y python3 python3-pip
RUN pip3 install torch torchvision
RUN pip install pycocotools
RUN pip install timm

