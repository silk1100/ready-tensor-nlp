FROM ubuntu:latest

RUN apt-get update && apt-get -y upgrade
RUN apt-get install -y --no-install-recommends git wget g++ gcc ca-certificates && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get upgrade -y && apt-get install python3-pip -y && \
    pip install scikit-learn && \
    pip install pandas && \
    pip install Flask

COPY ./src/ /usr/src/
RUN echo '#!/bin/bash\npython3 /usr/src/backend/train.py' > /usr/bin/train && chmod +x /usr/bin/train
RUN echo '#!/bin/bash\npython3 /usr/src/backend/testing.py' > /usr/bin/test && chmod +x /usr/bin/test
RUN echo '#!/bin/bash\npython3 /usr/src/frontend/app.py' > /usr/bin/serv && chmod +x /usr/bin/serv


WORKDIR /opt/ml_vol