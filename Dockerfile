FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /src

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jdk \
    bash \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /src/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /src/requirements.txt

COPY . /src