# Use official Python base image
FROM python:3.10-slim

# Install system-level dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    vim \ 
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only the necessary module and files
COPY requirements.txt .
COPY Person_counting_outside/ ./Person_counting_outside/
COPY beep_counting/ 
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set default command to bash
CMD ["bash"]
