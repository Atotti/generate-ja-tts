FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ARG PYTHON_VERSION=3.10
ENV DEBIAN_FRONTEND=noninteractive

ENV HOME /app
WORKDIR $HOME

RUN apt-get -y update && apt-get -y upgrade && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        liblzma-dev \
        liblzma-dev \
        libffi-dev \
        curl \
        clang \
        git-lfs \
        cmake \
        make \
        pkg-config \
        libgoogle-perftools-dev

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

CMD ["sleep", "INF"]
