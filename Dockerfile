FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    vim \
    nano \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY Person_counting_outside/ ./Person_counting_outside/
COPY beep_counting/ ./beep_counting/
COPY person_counter_machineROI/ ./person_counter_machineROI/

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["/bin/bash"]