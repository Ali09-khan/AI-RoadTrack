# syntax = docker/dockerfile:experimental
# Base image and Python version arguments
ARG BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04
ARG PYTHON_VERSION=3.12

########################################
#            Compile Stage             #
########################################
FROM ${BASE_IMAGE} AS compile-image
ARG PYTHON_VERSION
ARG CUDA_VERSION="12.4"
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    USE_CUDA=1

# Install system dependencies and Python 3.12 packages
RUN --mount=type=cache,id=apt-dev,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        g++ \
        python3-distutils \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        openjdk-17-jdk \
        curl \
        git \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

#Setting up env
RUN python${PYTHON_VERSION} -m venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

RUN python -m pip install --upgrade pip setuptools wheel

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1

# Installing additional system dependencies
RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
WORKDIR /app


RUN --mount=type=cache,target=/root/.cache/pip \
    python3.12 -m ensurepip --upgrade && \
    python3.12 -m pip install --upgrade setuptools && \
    python3.12 -m pip install -r requirements.txt

########################################
#         Production Stage             #
########################################
FROM ${BASE_IMAGE} AS production-image
ARG PYTHON_VERSION
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies for production
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-distutils \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        openjdk-17-jdk \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m model-server && \
    mkdir -p /home/model-server/{tmp,model-store,logs}

# Copy virtual environment from the compile stage
COPY --chown=model-server --from=compile-image /home/venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

# Create the entrypoint script for TorchServe
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
if [ "$1" = "serve" ] || [ "$1" = "" ]; then\n\
    torchserve --start --ts-config config.properties --model-store model-store --models laneLpService.mar,carDetectService.mar --disable-token-auth --ncs\n\
    # Prevent container exit\n\
    tail -f /dev/null\n\
elif [ "$1" = "serve-debug" ]; then\n\
    torchserve --start --ts-config config.properties --model-store model-store --models laneLpService.mar,carDetectService.mar --disable-token-auth --ncs --debug\n\
    # Prevent container exit\n\
    tail -f /dev/null\n\
else\n\
    exec "$@"\n\
fi' > /usr/local/bin/dockerd-entrypoint.sh

RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    chown -R model-server /home/model-server

COPY --chown=model-server service_kickoff/config.properties /home/model-server/config.properties
COPY --chown=model-server service_holder/ /home/model-server/model-store/

EXPOSE 8080 8081 8082 8888 7070 7071

USER model-server
WORKDIR /home/model-server
ENV TEMP=/home/model-server/tmp

# Define container entrypoint and default command
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

########################################
#         Development/Testing Stage    #
########################################
FROM ${BASE_IMAGE} AS dev-image
ARG PYTHON_VERSION
ARG BRANCH_NAME="master"
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TS_RUN_IN_DOCKER=TRUE

# Install system dependencies and development tools
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python${PYTHON_VERSION} \
        python3-distutils \
        python${PYTHON_VERSION}-dev \
        python${PYTHON_VERSION}-venv \
        openjdk-17-jdk \
        build-essential \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
        numactl \
        nodejs \
        npm \
        zip \
        unzip \
        git \
    && npm install -g newman@5.3.2 newman-reporter-htmlextra markdown-link-check && \
    rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /home/venv /home/venv
ENV PATH="/home/venv/bin:$PATH"

#this is so developer version will not cry
RUN python -m pip install intel_extension_for_pytorch==2.5.0 

# Install developer requirements from TorchServe's developer.txt
# Download the developer requirements file, patch it, and install from the patched file
RUN wget -O /tmp/developer.txt https://raw.githubusercontent.com/pytorch/serve/master/requirements/developer.txt && \
    wget -O /tmp/common.txt https://raw.githubusercontent.com/pytorch/serve/master/requirements/common.txt && \
    sed -i 's/intel_extension_for_pytorch==2\.3\.0/intel_extension_for_pytorch==2.5.0/' /tmp/developer.txt && \
    python -m pip install --no-cache-dir -r /tmp/developer.txt


RUN python -m pip install --no-cache-dir pytest pytest-cov pylint black isort mypy

RUN mkdir -p /home/dev-workspace/tests
WORKDIR /home/dev-workspace

CMD ["serve"] #CMD ["python", "-m", "pytest", "tests/"]

