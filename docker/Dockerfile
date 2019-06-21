# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

ARG http_proxy=http://www-cache.rd.bbc.co.uk:8080/
ARG https_proxy=https://www-cache.rd.bbc.co.uk:8080/
ARG no_proxy=.rd.bbc.co.uk,127.0.0.1,localhost,mirror.rd.bbc.co.uk 

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

# Clone the Detectron repository
RUN git clone https://github.com/AstronomerJoe/DensePose /densepose

# Install Python dependencies
RUN pip install -r /densepose/requirements.txt

# Get wget as required at a later stage
RUN apt-get clean && apt-get update
RUN apt-get install wget

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /densepose

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops
