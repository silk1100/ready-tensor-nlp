FROM ubuntu:latest

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends git wget g++ gcc ca-certificates && rm -rf /var/lib/apt/lists/*
RUN mkdir /opt/ml_vol && \
    mkdir /opt/ml_vol/inputs && \
    mkdir /opt/ml_vol/inputs/data_config/ && \
    mkdir /opt/ml_vol/inputs/data/ && \
    mkdir /opt/ml_vol/models && \
    mkdir /opt/ml_vol/models/artifacts/ && \
    mkdir /opt/ml_vol/outputs && \
    mkdir /opt/ml_vol/outputs/hpt_outputs && \
    mkdir /opt/ml_vol/outputs/testing_outputs && \
    mkdir /opt/ml_vol/outputs/errors && \
    mkdir /opt/ml_vol/src/

# COPY ./inputs/data/clickbait_schema.json /opt/ml_vol/inputs/data_config/clickbait_schema.json
# COPY ./inputs/data/training /opt/ml_vol/inputs/data/training
# COPY ./inputs/data/testing /opt/ml_vol/inputs/data/testing
# COPY ./src/ /opt/ml_vol/src/
COPY ./inputs /opt/ml_vol/inputs
COPY ./models /opt/ml_vol/models
COPY ./outputs /opt/ml_vol/outputs
COPY ./src /opt/ml_vol/src

# ENV PATH="/root/.bashrc"

# ENV PATH="/root/miniconda3/bin:${PATH}"
# ARG PATH="/root/miniconda3/bin:${PATH}"
# RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && mkdir /root/.conda \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh \
#     && echo "Running $(conda --version)" && \
#     conda init bash && \
#     . /root/.bashrc && \
#     conda update conda && \
#     conda create -n app python=3.9 && \
#     conda activate app &&\
#     conda install scikit-learn -y && \
#     conda install matplotlib seaborn pandas -y && \
#     conda install -c anaconda joblib -y

RUN apt-get update && apt-get upgrade -y && apt-get install python3-pip -y && \
    pip install scikit-learn && \
    pip install pandas

RUN echo '#!/bin/bash\ncd /opt/ml_vol\npython3 src/train.py' > /usr/bin/train && chmod +x /usr/bin/train
RUN echo '#!/bin/bash\ncd /opt/ml_vol\npython3 src/testing.py' > /usr/bin/test && chmod +x /usr/bin/test


WORKDIR /opt/ml_vol

# CMD cd /opt/ml_vol

# CMD conda activate app