# Election Code Suite

This repository contains three Python scripts for real-time video/audio analytics in election monitoring:

1. **`h_final.py`** – Multi-camera YOLOv8 RTMP processor with Azure image upload and API alerts.
2. **`final4.py`** – Per-camera two-person-near-EVM RTMP processor with dynamic ROI detection, Azure upload, and API alerts.
3. **`pitch-csv.py`** – Real-time EVM-beep (audio) detector from RTMP feeds, logging beeps to SQL Server.

---

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Scripts & Usage](#scripts--usage)

  * [h\_final.py](#h_finalpy)
  * [final4.py](#final4py)
  * [pitch-csv.py](#pitch-csvpy)
* [Contributing](#contributing)
* [License](#license)

---

## Features

* **YOLOv8** based person detection & supervision
* **Dynamic ROI** detection (EVM area)
* **Multi-RTMP** ingest and output via FFmpeg
* **Azure Blob Storage** integration for frame uploads
* **HTTP API** alerts on defined events
* **Sound detection** (EVM beep) via `librosa` and FFmpeg
* **Thread-/process-safe** design for high-throughput streams

---

## Prerequisites

* **OS**: Linux / Windows / macOS
* **Python**: ≥ 3.8
* **FFmpeg**: installed and on your `PATH`
* **Azure**: storage account + connection string
* **SQL Server**: for beep logging

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/VmuktiAI-ML/election-code.git
   cd election-code
   ```

2. Create & activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```

3. Install Python dependencies:

   ```bash
   pip install --upgrade pip
   pip install \
     opencv-python \
     torch torchvision \
     ultralytics \
     supervision \
     azure-storage-blob \
     shapely \
     requests \
     pandas \
     numpy \
     librosa \
     pyodbc
   ```

---

## Configuration

Before running, edit the following constants inside each script (or override via CLI flags):

* **Azure**

  * `AZURE_CONNECTION_STRING`
  * `AZURE_CONTAINER_NAME`, `AZURE_BLOB_PREFIX`
* **API URL**: `API_URL` / `ALERT_URL`
* **RTMP base URLs**
* **YOLO model paths** (`.pt` files)
* **Detection thresholds** & cooldown intervals

---

## Scripts & Usage

### `h_final.py`

Multi-camera RTMP ingest → YOLOv8 detection → annotated RTMP out → Azure upload + API alert.

```bash
usage: h_final.py [-h] [--yolo_model PATH] [-n NUM_STREAMS] [--offset OFFSET] [--max_workers MAX]

options:
  --yolo_model PATH    Path to YOLOv8 .pt weights (default: ./yolov8l.pt)
  -n, --num_streams    Number of streams to process in this instance
  --offset OFFSET      CSV row-offset for this instance
  --max_workers MAX    Max parallel streams (tune to your GPU/CPU)

# Prepare a CSV (default name: camera_10.csv) with one RTMP URL per line.
# Then launch:
python h_final.py --yolo_model ./yolov8l.pt -n 50 --max_workers 32
```

---

### `final4.py`

Per-camera threaded RTMP → dynamic ROI detection (first frame) → two-person-in-ROI alert → Azure + API.

```bash
# Ensure streams.csv exists, each line: rtmp://.../CAMERA_ID
python final4.py
```

* Spins up one daemon thread per line in `streams.csv`
* Uploads alert frames to Azure & issues HTTP POST alerts

---

### `pitch-csv.py`

Audio-only EVM-beep detection from RTMP streams, logs to SQL Server via stored proc.

```bash
usage: pitch-csv.py

# Prepare rtmp_urls.csv with one RTMP URL per line.
python pitch-csv.py
```

* Extracts raw audio via FFmpeg
* Computes pitch via `librosa.piptrack`
* Detects “beeps” based on thresholds & duration
* Logs each beep to SQL Server using pyodbc and `SaveBeepfromCamera`

---

## Contributing

1. Fork the repo
2. Create a feature branch (`git checkout -b feature/XYZ`)
3. Commit your changes (`git commit -am 'Add XYZ'`)
4. Push (`git push origin feature/XYZ`) and open a Pull Request

---

## License

[MIT License](LICENSE)

---

*Happy monitoring!*
