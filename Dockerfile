FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libosmesa6-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /ws

RUN git clone https://github.com/jmalegankar/InfoMARL.git .

CMD ["/bin/bash"]