FROM ubuntu:20.04

USER root

ENV PYTHONUNBUFFERED=1

# Set proper timezone
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Warsaw

# Install system dependencies in single layer for efficiency
RUN apt-get update && apt-get install -y \
    tzdata \
    python3-pip \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and setuptools
RUN pip3 install --upgrade pip setuptools

# Copy requirements files for dependency installation
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies (cached layer)
RUN pip3 install --no-cache-dir -r requirements.txt \
    && pip3 install --no-cache-dir -r requirements-dev.txt

# Create necessary directories
RUN mkdir -p /bachelor_thesis/data/input \
             /bachelor_thesis/data/processed/periods \
             /bachelor_thesis/data/processed/periods/unique \
             /bachelor_thesis/data/processed/temp_fasta \
             /bachelor_thesis/data/processed/temp_csv \
             /bachelor_thesis/data/processed/sequences_as_vectors \
             /bachelor_thesis/data/processed/datasets \
             /bachelor_thesis/models \
             /bachelor_thesis/results