FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

WORKDIR /app

# Prevent timezone prompt
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and build requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev && \
    rm -rf /var/lib/apt/lists/*

# Install Python 3.12.3
RUN wget https://www.python.org/ftp/python/3.12.3/Python-3.12.3.tgz && \
    tar xzf Python-3.12.3.tgz && \
    cd Python-3.12.3 && \
    ./configure --enable-optimizations --prefix=/usr/local && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && \
    rm -rf Python-3.12.3 Python-3.12.3.tgz

# Install pip for the newly installed Python
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Create symbolic links for python3 and pip3
RUN ln -sf /usr/local/bin/python3.12 /usr/local/bin/python && \
    ln -sf /usr/local/bin/pip3.12 /usr/local/bin/pip

# Set PATH to prioritize the Python 3.12.3 installation
ENV PATH="/usr/local/bin:${PATH}"

# Clone repositories using HTTPS
RUN git clone https://github.com/groundlight/r1_vlm.git && \
    git clone --branch release_2025_03_06 --single-branch https://github.com/groundlight/verifiers.git && \
    git clone --branch release_2025_03_06 --single-branch https://github.com/groundlight/trl.git

WORKDIR /app/r1_vlm

# Install Python dependencies with proper build isolation settings and essential build dependencies
RUN python3.12 -m pip install setuptools wheel build hatchling editables && \
    python3.12 -m pip install numpy psutil && \
    python3.12 -m pip install 'torch==2.5.1' --index-url https://download.pytorch.org/whl/cu124 && \
    python3.12 -m pip install --no-build-isolation -e .  && \
    cd ../trl && python3.12 -m pip install --no-build-isolation -e . && \
    cd ../verifiers && python3.12 -m pip install --no-build-isolation -e . && \
    cd ../r1_vlm

# Default command
CMD ["bash"]
