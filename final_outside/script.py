#!/usr/bin/env python3
import os
# --- Control thread usage to limit threads for libraries ---
os.environ["OMP_NUM_THREADS"]        = "1"
os.environ["OPENBLAS_NUM_THREADS"]   = "1"
os.environ["MKL_NUM_THREADS"]        = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"]    = "1"
os.environ["OPENCV_FFMPEG_READ_ATTEMPTS"] = "999999" # Increased attempts for reading frames
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp;timeout;60000000" # Timeout in microseconds (60 seconds)
import cv2
import torch
import time
import argparse
import multiprocessing as mp
import subprocess
import pandas as pd
import requests
import threading
import signal
import logging
import queue
import gc
from datetime import datetime

# YOLO & Supervision
from ultralytics import YOLO
import supervision as sv

# Azure
from azure.storage.blob import BlobServiceClient

# ==== Configuration ====
STATIC_THRESHOLD       = 0.3
MAX_PERSON_THRESHOLD   = 14
# throttle alerts: minimum seconds between alerts from same camera
ALERT_INTERVAL_SECONDS = 50  # 5 minutes

OUTPUT_HOST            = "rtmp://aielection.vmukti.com:80/live-record"
RETRY_INTERVAL         = 1
DEFAULT_FPS            = 25
API_URL                = "https://tn2023demo.vmukti.com/api/analytics"

AZURE_CONNECTION_STRING = (
    "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;"
    "QueueEndpoint=https://nvrdatashinobi.queue.core.windows.net/;"
    "FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;"
    "TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;"
    "SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&"
    "se=2025-07-31T13:32:09Z&st=2025-03-31T05:32:09Z&spr=https,http&"
    "sig=lxI3Z67F40w8c2M3i%2FAvx7dJQNo6LU%2Bx3TVE2XM0qws%3D"
)
AZURE_CONTAINER_NAME = "nvrdatashinobi"
AZURE_BLOB_PREFIX    = "live-record/frimages"
STATIC_IMAGE_URL     = (
    f"https://nvrdatashinobi.blob.core.windows.net/"
    f"{AZURE_CONTAINER_NAME}/{AZURE_BLOB_PREFIX}"
)

# camera identifier, used in blob filenames
camera_id = "ANYK-804908-AAAAA"

# default YOLO weights—will override via CLI
YOLO_MODEL_PATH = "./yolov8l.pt"

# thread-safe shutdown flag
shutdown_event = threading.Event()
camera_last_alert_time = {}

# Global process management
active_processes = []
process_lock = threading.Lock()

# Resource management for 500 cameras
MAX_CONCURRENT_PROCESSES = 64  # Adjust based on your system
PROCESS_BATCH_SIZE = 8  # Number of cameras per process batch
MEMORY_CLEANUP_INTERVAL = 300  # 5 minutes

# Connection pool for better resource management
connection_pool_size = 100
available_connections = queue.Queue(maxsize=connection_pool_size)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(processName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("camera_streams_500.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def signal_handler(signum, frame):
    logger.info(f"Received shutdown signal ({signum}), shutting down 500 cameras...")
    shutdown_event.set()
    cleanup_processes()

def cleanup_processes():
    """Clean up all active processes"""
    with process_lock:
        for proc in active_processes:
            try:
                if proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=5)
                    if proc.is_alive():
                        proc.kill()
            except:
                pass
        active_processes.clear()

def initialize_yolo_model(weights_path, device, stream_name):
    try:
        model = YOLO(weights_path)
        model.model.to(device)
        # Enable model optimizations for batch processing
        model.model.eval()
        torch.cuda.empty_cache() if device.type == 'cuda' else None
        logger.info(f"[{stream_name}] Loaded YOLOv8 model from {weights_path} on {device}")
        return model
    except Exception as e:
        logger.error(f"[{stream_name}] YOLO init failed: {e}", exc_info=True)
        return None

def initialize_azure_blob_client():
    try:
        return BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
    except Exception as e:
        logger.error(f"Azure init error: {e}", exc_info=True)
        return None

def upload_image_to_azure(blob_service_client, cam_id, frame_img):
    try:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{cam_id}_{timestamp}.png"
        blob_name = f"{AZURE_BLOB_PREFIX}/{filename}"
        # Optimize image encoding for speed
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Fast compression
        _, buf = cv2.imencode('.png', frame_img, encode_param)
        blob_client = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME, blob=blob_name
        )
        blob_client.upload_blob(buf.tobytes(), overwrite=True)
        return f"{STATIC_IMAGE_URL}/{filename}"
    except Exception as e:
        logger.error(f"[{cam_id}] Azure upload failed: {e}", exc_info=True)
        return None

def should_send_alert(cam_id):
    now = time.time()
    last = camera_last_alert_time.get(cam_id, 0)
    if now - last >= ALERT_INTERVAL_SECONDS:
        camera_last_alert_time[cam_id] = now
        return True
    return False

def send_alert_to_api(cam_id, frame_img, count, blob_service_client):
    if shutdown_event.is_set() or not should_send_alert(cam_id):
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
    img_url = upload_image_to_azure(blob_service_client, cam_id, frame_img)
    if not img_url:
        return
    payload = {
        "cameradid":  cam_id,
        "sendtime":   timestamp,
        "imgurl":     img_url,
        "an_id":      20,
        "ImgCount":   4,
        "totalcount": count
    }
    try:
        resp = requests.post(API_URL, json=payload, timeout=(5, 15))
        if resp.status_code == 200:
            logger.info(f"[{cam_id}] API alert sent.")
        else:
            logger.warning(f"[{cam_id}] API responded {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"[{cam_id}] API call error: {e}", exc_info=True)

def connect_to_stream_with_retry(rtmp_url, stream_name, max_retries=5):
    retries = 0
    while not shutdown_event.is_set() and retries < max_retries:
        try:
            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            # Set buffer size to reduce latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS)
           
            if cap.isOpened():
                logger.info(f"[{stream_name}] Connected to {rtmp_url}")
                return cap
           
            retries += 1
            logger.warning(f"[{stream_name}] Connection failed (attempt {retries}/{max_retries}), retrying in {RETRY_INTERVAL}s")
            cap.release()
            time.sleep(RETRY_INTERVAL)
        except Exception as e:
            logger.error(f"[{stream_name}] Connection error: {e}")
            retries += 1
            time.sleep(RETRY_INTERVAL)
   
    logger.error(f"[{stream_name}] Failed to connect after {max_retries} attempts")
    return None

def create_ffmpeg_process(width, height, fps, output_url, stream_name):
    while not shutdown_event.is_set():
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}", '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p', '-f', 'flv',
            '-bufsize', '2M', '-maxrate', '2M',  # Add rate limiting for stability
            output_url
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[{stream_name}] FFmpeg streaming to {output_url}")
            time.sleep(0.1)
            return proc
        except Exception as e:
            logger.warning(f"[{stream_name}] FFmpeg start error: {e}", exc_info=True)
            time.sleep(RETRY_INTERVAL)
    return None

def memory_cleanup_worker():
    """Periodic memory cleanup for long-running processes"""
    while not shutdown_event.is_set():
        time.sleep(MEMORY_CLEANUP_INTERVAL)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("Performed memory cleanup")

def process_stream_with_retry(rtmp_url):
    pname = rtmp_url.split('/')[-1].replace('.', '_')
    mp.current_process().name      = f"{pname}"
    threading.current_thread().name = f"Thread-{pname}"
    stream_name = mp.current_process().name

    torch.set_num_threads(1)
    logger.info(f"[{stream_name}] Starting")

    # Start memory cleanup thread
    cleanup_thread = threading.Thread(target=memory_cleanup_worker, daemon=True)
    cleanup_thread.start()

    # Distribute across 8 H100 GPUs
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()  # Should be 8
        # Assign GPU based on a hash of the stream name to ensure even distribution
        gpu_id = hash(stream_name) % gpu_count
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"[{stream_name}] Device: {device}")

    model = initialize_yolo_model(YOLO_MODEL_PATH, device, stream_name)
    if not model:
        return
    box_annotator = sv.RoundBoxAnnotator(
        thickness=2,
    )


    blob_client = initialize_azure_blob_client()

    box_annotator = sv.RoundBoxAnnotator(thickness=2)
    key           = rtmp_url.split('/')[-1] + "-AI"
    output_url    = f"{OUTPUT_HOST}/{key}"

    # Connect & grab video properties (retry until available)
    cap = connect_to_stream_with_retry(rtmp_url, stream_name)
    if not cap or shutdown_event.is_set():
        logger.info(f"[{stream_name}] Shutdown during initial connect, exiting")
        return
   
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps    = min(int(cap.get(cv2.CAP_PROP_FPS)) or DEFAULT_FPS, DEFAULT_FPS)
    cap.release()

    ffmpeg_proc = create_ffmpeg_process(width, height, fps, output_url, stream_name)
    frame_delay = 1.0 / fps
   
    # Frame skip logic for performance
    frame_skip_counter = 0
    frame_skip_interval = 2  # Process every 2nd frame for 500 cameras

    while not shutdown_event.is_set():
        cap = connect_to_stream_with_retry(rtmp_url, stream_name)
        if not cap:
            time.sleep(RETRY_INTERVAL)
            continue

        consecutive_failures = 0
        max_consecutive_failures = 50

        while not shutdown_event.is_set():
            start_time = time.time()
            ret, frame = cap.read()
           
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"[{stream_name}] Too many consecutive failures, reconnecting")
                    break
                continue
           
            consecutive_failures = 0
            frame_skip_counter += 1

            # Skip frames for better performance with 500 cameras
            if frame_skip_counter % frame_skip_interval != 0:
                if ffmpeg_proc and ffmpeg_proc.poll() is None:
                    try:
                        ffmpeg_proc.stdin.write(frame.tobytes())
                    except:
                        break
                continue

            try:
                # inference with explicit conf & iou, no unpack
                results = model(
                    frame,
                    device=str(device),
                    imgsz=640,  # Reduced from 1280 for better performance
                    conf=STATIC_THRESHOLD,
                    iou=0.45,
                    classes=[0],
                    augment=False,  # Disabled for speed
                    verbose=False
                )
                boxes       = results[0].boxes.xyxy.cpu().numpy()
                confidences = results[0].boxes.conf.cpu().numpy()
                class_ids   = results[0].boxes.cls.cpu().numpy().astype(int)
                count       = len(boxes)

                detections      = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
                annotated_frame = box_annotator.annotate(scene=frame, detections=detections)

                cv2.putText(annotated_frame, f"People: {count}", (520, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 218, 25), 2, cv2.LINE_AA)
               
                if count > MAX_PERSON_THRESHOLD:
                    cv2.putText(annotated_frame, "!!ALERT: MAX PERSONS !!!", (420, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
                    threading.Thread(
                        target=send_alert_to_api,
                        args=(stream_name, annotated_frame.copy(), count, blob_client),
                        daemon=True
                    ).start()

                if ffmpeg_proc and ffmpeg_proc.poll() is None:
                    try:
                        ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                    except:
                        break
                else:
                    break

            except Exception as e:
                logger.error(f"[{stream_name}] Processing error: {e}")
                continue

            elapsed = time.time() - start_time
            if (sleep_time := (frame_delay - elapsed)) > 0:
                time.sleep(sleep_time)

        cap.release()
        try:
            if ffmpeg_proc:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.terminate()
                ffmpeg_proc.wait(timeout=5)
        except:
            if ffmpeg_proc:
                ffmpeg_proc.kill()
        time.sleep(RETRY_INTERVAL)

    logger.info(f"[{stream_name}] Exiting")

def launch_cameras_batch(rtmp_urls_batch, batch_id):
    """Launch a batch of cameras in a single process"""
    logger.info(f"Batch {batch_id}: Starting {len(rtmp_urls_batch)} cameras")
   
    # Use threading for cameras within a batch
    threads = []
    for url in rtmp_urls_batch:
        thread = threading.Thread(target=process_stream_with_retry, args=(url,), daemon=True)
        threads.append(thread)
        thread.start()
        time.sleep(0.1)  # Small delay to stagger startup
   
    # Wait for all threads in this batch
    for thread in threads:
        thread.join()
   
    logger.info(f"Batch {batch_id}: All cameras finished")

def launch_cameras(rtmp_urls, max_workers):
    logger.info(f"Launching {len(rtmp_urls)} camera streams across {max_workers} workers")
    logger.info(f"Batch size: {PROCESS_BATCH_SIZE} cameras per batch")
   
    # Split URLs into batches
    batches = []
    for i in range(0, len(rtmp_urls), PROCESS_BATCH_SIZE):
        batch = rtmp_urls[i:i + PROCESS_BATCH_SIZE]
        batches.append(batch)
   
    logger.info(f"Created {len(batches)} batches")
   
    # Launch batches using process pool
    with mp.get_context("spawn").Pool(max_workers) as pool:
        # Create arguments for each batch
        batch_args = [(batch, i) for i, batch in enumerate(batches)]
       
        # Use starmap to launch batches
        pool.starmap(launch_cameras_batch, batch_args)

def main():
    global YOLO_MODEL_PATH, PROCESS_BATCH_SIZE
    signal.signal(signal.SIGINT,  signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Multi-camera YOLOv8 + Supervision RTMP Processor - 500 Cameras"
    )
    parser.add_argument(
        '--yolo_model', type=str, default=YOLO_MODEL_PATH,
        help="Path to YOLOv8 .pt weights"
    )
    parser.add_argument(
        '-n', '--num_streams', type=int, default=500,
        help="How many streams this instance should handle (default: 500)"
    )
    parser.add_argument(
        '-o', '--offset', type=int, default=0,
        help="Index offset into the CSV (for this instance)"
    )
    parser.add_argument(
        '--max_workers', type=int, default=MAX_CONCURRENT_PROCESSES, # Set default to optimal 64
        help=f"Max concurrent batch processes (default: {MAX_CONCURRENT_PROCESSES})"
    )
    parser.add_argument(
        '--batch_size', type=int, default=PROCESS_BATCH_SIZE, # Set default to optimal 8
        help=f"Number of cameras per batch (default: {PROCESS_BATCH_SIZE})"
    )
    args = parser.parse_args()

    YOLO_MODEL_PATH = args.yolo_model
    PROCESS_BATCH_SIZE = args.batch_size

    csv_path = "camera_10.csv"
    if not os.path.exists(csv_path):
        logger.error(f"CSV not found: {csv_path}")
        return

    df       = pd.read_csv(csv_path, header=None)
    all_urls = df[0].dropna().tolist()
   
    # Handle 500 cameras
    if args.num_streams is not None:
        start = args.offset
        end   = start + args.num_streams
        urls  = all_urls[start:end]
    else:
        urls = all_urls

    if len(urls) > 500:
        logger.warning(f"Limiting to 500 cameras (requested: {len(urls)})")
        urls = urls[:500]
   
    logger.info(f"Processing {len(urls)} cameras with {args.max_workers} worker processes")
    logger.info(f"Batch size: {PROCESS_BATCH_SIZE} cameras per batch")
   
    try:
        launch_cameras(urls, args.max_workers)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        cleanup_processes()
        logger.info("Shutdown complete")

if __name__ == "__main__":
    main()


