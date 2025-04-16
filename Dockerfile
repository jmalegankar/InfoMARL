FROM nvidia/cuda:12.6.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=America/Los_Angeles \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# Install Python 3.12 and pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-dev python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && ln -s /usr/bin/python3.12 /usr/local/bin/python3 \
    && ln -s /usr/bin/python3.12 /usr/local/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/l

# Create non-root user
RUN useradd -m -s /bin/bash user
USER user
WORKDIR /home/user

# Install Python packages
RUN python3 -m pip install --no-cache-dir \
    absl-py==2.2.2 \
    antlr4-python3-runtime==4.9.3 \
    av==13.1.0 \
    certifi==2025.1.31 \
    charset-normalizer==3.4.1 \
    cloudpickle==3.1.1 \
    decorator==4.4.2 \
    filelock==3.17.0 \
    fsspec==2025.2.0 \
    grpcio==1.71.0 \
    gym==0.26.2 \
    gym-notices==0.0.8 \
    hydra-core==1.3.2 \
    idna==3.10 \
    imageio==2.37.0 \
    imageio-ffmpeg==0.6.0 \
    Jinja2==3.1.5 \
    joblib==1.4.2 \
    Markdown==3.8 \
    MarkupSafe==3.0.2 \
    moviepy==1.0.3 \
    mpmath==1.3.0 \
    networkx==3.4.2 \
    numpy==2.2.2 \
    omegaconf==2.3.0 \
    orjson==3.10.15 \
    packaging==24.2 \
    pillow==10.4.0 \
    proglog==0.1.11 \
    protobuf==6.30.2 \
    pyglet==1.5.27 \
    python-dotenv==1.1.0 \
    PyYAML==6.0.2 \
    requests==2.32.3 \
    setuptools==75.8.0 \
    six==1.17.0 \
    sympy==1.13.1 \
    tensorboard==2.19.0 \
    tensorboard-data-server==0.7.2 \
    tensordict==0.7.0 \
    torch==2.6.0 \
    torchrl==0.7.0 \
    torchvision==0.21.0 \
    tqdm==4.67.1 \
    typing_extensions==4.12.2 \
    urllib3==2.4.0 \
    vmas==1.5.0 \
    Werkzeug==3.1.3


COPY ./new_arch /home/user/new_arch

WORKDIR /home/user/new_arch
