FROM ubuntu:20.04

USER root

ENV PYTHONUNBUFFERED=1

# set proper timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw
RUN apt-get update && apt-get install -y tzdata
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update
RUN apt-get install -y python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache --upgrade pip setuptools

RUN pip3 install --no-cache --upgrade torch==2.0.1 pandas==2.0.3 scikit-learn==1.3.2 numpy==1.22.1 matplotlib==3.5.2