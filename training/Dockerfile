FROM ubuntu:18.04

ENV DEBIAN_FRONTEND=noninteractive

# Instalar OpenCV 3 y sus herramientas desde los repos oficiales de Ubuntu 18.04
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    opencv-data \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work