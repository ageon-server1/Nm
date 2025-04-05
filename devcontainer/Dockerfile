# Use an official CUDA image (modify version as needed)
FROM nvidia/cuda:11.7.0-devel-ubuntu20.04

# Install required packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    net-tools \
    iproute2 \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user "vscode"
ARG USERNAME=vscode
RUN useradd -m ${USERNAME} && echo "${USERNAME}:${USERNAME}" | chpasswd && adduser ${USERNAME} sudo

USER ${USERNAME}
WORKDIR /workspace
