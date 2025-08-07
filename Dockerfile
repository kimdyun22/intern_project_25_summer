# Base image with CUDA 12.1 + cuDNN + PyTorch
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /workspace

# Install dependencies (libGL for cv2, libglib2.0 for GUI support, tzdata to suppress tz warning)
RUN apt update && \
    apt install -y libgl1 libglib2.0-0 tzdata

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --upgrade pip && \
    pip install numpy==1.24.4 --force-reinstall && \
    pip install -r /tmp/requirements.txt
