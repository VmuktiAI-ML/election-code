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
import threading # Still used for threads within individual processes
import signal
import logging
import queue # Unused, but kept as per "don't change functionality"
import gc
from datetime import datetime, timedelta

# YOLO & Supervision
from ultralytics import YOLO
import supervision as sv

# Azure
from azure.storage.blob import BlobServiceClient

# ==== Configuration ====
STATIC_THRESHOLD       = 0.3
MAX_PERSON_THRESHOLD   = 10
ALERT_INTERVAL_SECONDS = 50

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

camera_id = "ANYK-804908-AAAAA" # Global, seems unused for per-stream identification in alerts

YOLO_MODEL_PATH = "./yolov8m.pt" # Default, overridden by CLI

# These will be initialized in main as multiprocessing-safe objects
shutdown_event = None 
camera_last_alert_time = None 

active_processes = [] # List to keep track of started multiprocessing.Process objects
process_lock = threading.Lock() # To protect access to active_processes list

# Default values for CLI arguments
MAX_CONCURRENT_PROCESSES_DEFAULT = 64
PROCESS_BATCH_SIZE_DEFAULT = 8 # CLI arg kept, but its specific use in launching logic is removed
MEMORY_CLEANUP_INTERVAL = 300  # 5 minutes

# Unused connection pool, kept as is
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

# Signal handler function (will be wrapped in main to use mp.Event)
def main_signal_handler(signum, frame, mp_shutdown_event):
    logger.info(f"Signal {signum} received in main process. Setting shutdown event.")
    if mp_shutdown_event:
        mp_shutdown_event.set()

def cleanup_processes():
    logger.info("Cleaning up active processes...")
    with process_lock:
        # Iterate over a copy for safe removal if needed, though clear() is used
        for proc in list(active_processes): 
            try:
                if proc.is_alive():
                    logger.info(f"Terminating process {proc.name} (PID {proc.pid})")
                    proc.terminate() # Send SIGTERM
                    proc.join(timeout=5) # Wait for clean exit
                    if proc.is_alive():
                        logger.warning(f"Process {proc.name} (PID {proc.pid}) did not terminate, killing.")
                        proc.kill() # Send SIGKILL
                        proc.join(timeout=1) # Wait for kill
            except Exception as e:
                logger.error(f"Error terminating process {proc.name if proc and hasattr(proc, 'name') else 'Unknown'}: {e}")
        active_processes.clear()
    logger.info("Process cleanup complete.")


def initialize_yolo_model(weights_path, device, stream_name):
    try:
        model = YOLO(weights_path)
        model.to(device) # Ultralytics YOLO objects have a .to() method
        # model.model.eval() # model.eval() is usually handled by YOLO, but can be explicit if needed
        if device.type == 'cuda':
            torch.cuda.empty_cache()
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
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Fast compression
        _, buf = cv2.imencode('.png', frame_img, encode_param)
        blob_client_obj = blob_service_client.get_blob_client(
            container=AZURE_CONTAINER_NAME, blob=blob_name
        )
        blob_client_obj.upload_blob(buf.tobytes(), overwrite=True)
        return f"{STATIC_IMAGE_URL}/{filename}"
    except Exception as e:
        logger.error(f"[{cam_id}] Azure upload failed: {e}", exc_info=True)
        return None

# Modified to accept shared multiprocessing.Manager.dict
def should_send_alert(cam_id, camera_last_alert_time_shared):
    now = time.time()
    # Use .get(key, default) for dictionary access
    last = camera_last_alert_time_shared.get(cam_id, 0)
    if now - last >= ALERT_INTERVAL_SECONDS:
        camera_last_alert_time_shared[cam_id] = now
        return True
    return False

# Modified to accept shared multiprocessing objects
def send_alert_to_api(cam_id, frame_img, count, blob_service_client, 
                      camera_last_alert_time_shared, shutdown_event_shared):
    if shutdown_event_shared.is_set():
        logger.info(f"[{cam_id}] Shutdown in progress, skipping alert.")
        return
    
    if not should_send_alert(cam_id, camera_last_alert_time_shared):
        logger.debug(f"[{cam_id}] Alert throttled for {cam_id}.")
        return

    #timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.000")
    timestamp = (datetime.now() + timedelta(hours=5, minutes=30)).strftime("%Y-%m-%d %H:%M:%S.000")

    img_url = upload_image_to_azure(blob_service_client, cam_id, frame_img)
    if not img_url:
        logger.warning(f"[{cam_id}] Image upload failed for {cam_id}, cannot send alert.")
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
        if shutdown_event_shared.is_set(): # Check again before network call
            logger.info(f"[{cam_id}] Shutdown just before API call for {cam_id}, skipping alert.")
            return
        resp = requests.post(API_URL, json=payload, timeout=(5, 15)) # (connect_timeout, read_timeout)
        if resp.status_code == 200:
            logger.info(f"[{cam_id}] API alert sent successfully for {cam_id}.")
        else:
            logger.warning(f"[{cam_id}] API responded {resp.status_code} for {cam_id}: {resp.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"[{cam_id}] API call error for {cam_id}: {e}", exc_info=True)


# Modified to accept shutdown_event (mp.Event)
# Modified to accept shutdown_event (mp.Event) and retry indefinitely
def connect_to_stream_with_retry(rtmp_url, stream_name, shutdown_event_obj):
    attempt_count = 0
    while not shutdown_event_obj.is_set(): # Loop indefinitely until shutdown or success
        try:
            attempt_count += 1
            # For logging purposes, indicate which attempt this is
            if attempt_count > 1: # Don't log "attempt 1" before the first try
                 logger.info(f"[{stream_name}] Attempting to connect to {rtmp_url} (attempt {attempt_count})...")

            cap = cv2.VideoCapture(rtmp_url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce latency
            cap.set(cv2.CAP_PROP_FPS, DEFAULT_FPS) # Suggest FPS to backend
           
            if cap.isOpened():
                logger.info(f"[{stream_name}] Connected to {rtmp_url} on attempt {attempt_count}")
                return cap
            
            cap.release() # Explicitly release if not opened
            logger.warning(f"[{stream_name}] Connection to {rtmp_url} failed (attempt {attempt_count}), retrying in {RETRY_INTERVAL}s")
            
            # Check shutdown event before sleeping
            if shutdown_event_obj.is_set(): 
                logger.info(f"[{stream_name}] Shutdown signaled during connection retry to {rtmp_url}.")
                break 
            time.sleep(RETRY_INTERVAL)

        except Exception as e:
            logger.error(f"[{stream_name}] Error during VideoCapture for {rtmp_url} on attempt {attempt_count}: {e}")
            # Check shutdown event before sleeping after an exception
            if shutdown_event_obj.is_set():
                logger.info(f"[{stream_name}] Shutdown signaled after connection error for {rtmp_url}.")
                break
            time.sleep(RETRY_INTERVAL)
   
    if shutdown_event_obj.is_set():
        logger.info(f"[{stream_name}] Connection attempts to {rtmp_url} stopped due to shutdown signal.")
    # If the loop exits due to shutdown, it will return None
    return None

# Modified to accept shutdown_event (mp.Event)
def create_ffmpeg_process(width, height, fps, output_url, stream_name, shutdown_event_obj):
    while not shutdown_event_obj.is_set():
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', f"{width}x{height}", '-r', str(fps), '-i', '-',
            '-c:v', 'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p', '-f', 'flv',
            '-bufsize', '2M', '-maxrate', '2M',
            output_url
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            logger.info(f"[{stream_name}] FFmpeg process started, streaming to {output_url}")
            time.sleep(0.2) # Give FFmpeg a moment to start or fail
            if proc.poll() is None: # Check if FFmpeg is still running
                 return proc
            else:
                 logger.warning(f"[{stream_name}] FFmpeg process for {output_url} exited immediately with code {proc.poll()}. Retrying...")
                 if proc.stdin: # Ensure stdin is closed if open
                     try: proc.stdin.close()
                     except: pass
                 proc.wait() # Wait for process to clean up
        except Exception as e:
            logger.warning(f"[{stream_name}] FFmpeg start error for {output_url}: {e}", exc_info=True)
        
        if shutdown_event_obj.is_set(): break
        logger.info(f"[{stream_name}] Retrying FFmpeg setup for {output_url} in {RETRY_INTERVAL} seconds...")
        time.sleep(RETRY_INTERVAL)
    return None


# Modified to accept shutdown_event (mp.Event)
def memory_cleanup_worker(shutdown_event_obj, process_name):
    """Periodic memory cleanup for long-running processes"""
    while not shutdown_event_obj.is_set():
        # Wait for MEMORY_CLEANUP_INTERVAL seconds, but check shutdown_event frequently
        for _ in range(MEMORY_CLEANUP_INTERVAL): # Check every second
            if shutdown_event_obj.is_set():
                break
            time.sleep(1)
        
        if shutdown_event_obj.is_set():
            break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"[{process_name}] Performed memory cleanup")

# Modified signature to accept shared multiprocessing objects
# YOLO_MODEL_PATH is accessed as a global within this function
def process_stream_with_retry(rtmp_url, shutdown_event_obj, camera_last_alert_time_shared):
    pname_base = rtmp_url.split('/')[-1].replace('.', '_') if rtmp_url else "unknown_stream"
    pname_base = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in pname_base) # Sanitize
    if not pname_base: pname_base = f"stream_{hash(rtmp_url)}"
    
    mp.current_process().name = f"Proc-{pname_base}"
    threading.current_thread().name = f"Thread-{pname_base}-Main"
    stream_name_for_log = mp.current_process().name

    torch.set_num_threads(1)
    logger.info(f"[{stream_name_for_log}] Starting processing for {rtmp_url}")

    cleanup_thread = threading.Thread(target=memory_cleanup_worker, args=(shutdown_event_obj, stream_name_for_log,), daemon=True)
    cleanup_thread.name = f"Thread-{pname_base}-MemClean"
    cleanup_thread.start()

    device = None
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        if gpu_count > 0:
            gpu_id = hash(pname_base) % gpu_count 
            device = torch.device(f"cuda:{gpu_id}")
            try:
                torch.cuda.set_device(device)
            except RuntimeError as e: # Catch specific error for invalid device
                logger.error(f"[{stream_name_for_log}] Failed to set CUDA device cuda:{gpu_id}: {e}. Falling back to CPU.")
                device = torch.device("cpu")
        else:
            logger.warning(f"[{stream_name_for_log}] CUDA available but no GPUs found. Using CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
    logger.info(f"[{stream_name_for_log}] Using device: {device}")

    model = initialize_yolo_model(YOLO_MODEL_PATH, device, stream_name_for_log)
    if not model:
        logger.error(f"[{stream_name_for_log}] Failed to initialize YOLO model. Exiting process for {rtmp_url}.")
        return

    box_annotator = sv.RoundBoxAnnotator(thickness=2)
    blob_service_client = initialize_azure_blob_client()

    output_key = f"{pname_base}-AI"
    output_url = f"{OUTPUT_HOST}/{output_key}"

    cap = connect_to_stream_with_retry(rtmp_url, stream_name_for_log, shutdown_event_obj)
    if not cap or shutdown_event_obj.is_set():
        logger.info(f"[{stream_name_for_log}] Initial connect failed or shutdown for {rtmp_url}. Exiting.")
        if cap: cap.release()
        return
   
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_cap = cap.get(cv2.CAP_PROP_FPS)
    cap.release() 

    width = width if width > 0 else 640
    height = height if height > 0 else 480
    fps = min(int(fps_cap if fps_cap > 0 else DEFAULT_FPS), DEFAULT_FPS)
    logger.info(f"[{stream_name_for_log}] Video properties for {rtmp_url}: {width}x{height} @ {fps} FPS (capture FPS: {fps_cap})")

    ffmpeg_proc = create_ffmpeg_process(width, height, fps, output_url, stream_name_for_log, shutdown_event_obj)
    if not ffmpeg_proc and not shutdown_event_obj.is_set():
        logger.error(f"[{stream_name_for_log}] Failed to start FFmpeg for {output_url}. Exiting process.")
        return

    frame_delay = 1.0 / fps if fps > 0 else (1.0 / DEFAULT_FPS)
    frame_skip_counter = 0
    frame_skip_interval = 2 

    last_ffmpeg_check_time = time.time()
    ffmpeg_check_interval = 10 # seconds

    main_processing_loop_active = True
    while main_processing_loop_active and not shutdown_event_obj.is_set():
        cap = connect_to_stream_with_retry(rtmp_url, stream_name_for_log, shutdown_event_obj)
        if not cap:
            if shutdown_event_obj.is_set(): break
            logger.warning(f"[{stream_name_for_log}] Reconnect to {rtmp_url} failed. Retrying in {RETRY_INTERVAL}s.")
            time.sleep(RETRY_INTERVAL)
            continue

        consecutive_failures = 0
        max_consecutive_failures = int(fps * 5) if fps > 0 else 50 # 5 seconds of failures

        frame_processing_loop_active = True
        while frame_processing_loop_active and not shutdown_event_obj.is_set():
            loop_start_time = time.time()
            
            if ffmpeg_proc and time.time() - last_ffmpeg_check_time > ffmpeg_check_interval:
                last_ffmpeg_check_time = time.time()
                if ffmpeg_proc.poll() is not None:
                    logger.warning(f"[{stream_name_for_log}] FFmpeg process for {output_url} died. Restarting.")
                    try: 
                        if ffmpeg_proc.stdin: ffmpeg_proc.stdin.close()
                    except: pass
                    ffmpeg_proc.wait()
                    ffmpeg_proc = create_ffmpeg_process(width, height, fps, output_url, stream_name_for_log, shutdown_event_obj)
                    if not ffmpeg_proc and not shutdown_event_obj.is_set():
                        logger.error(f"[{stream_name_for_log}] Failed to restart FFmpeg for {output_url}. Breaking inner loop.")
                        frame_processing_loop_active = False; continue
            
            if not ffmpeg_proc and not shutdown_event_obj.is_set():
                logger.error(f"[{stream_name_for_log}] FFmpeg not running for {output_url}. Breaking inner loop.")
                frame_processing_loop_active = False; continue

            ret, frame = cap.read()
           
            if not ret or frame is None:
                consecutive_failures += 1
                if consecutive_failures % (fps if fps > 0 else 10) == 0 : # Log every second of failure
                     logger.warning(f"[{stream_name_for_log}] Failed to read frame from {rtmp_url} ({consecutive_failures} consecutive).")
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"[{stream_name_for_log}] Max consecutive frame read failures for {rtmp_url}. Reconnecting.")
                    frame_processing_loop_active = False; continue 
                time.sleep(0.01) 
                continue
           
            consecutive_failures = 0
            frame_skip_counter += 1
            annotated_frame = frame 

            if frame_skip_counter % frame_skip_interval == 0:
                try:
                    results = model(frame, device=str(device), imgsz=640, conf=STATIC_THRESHOLD, iou=0.45, classes=[0], augment=False, verbose=False)
                    
                    count = 0
                    if results and results[0].boxes and len(results[0].boxes.xyxy) > 0 :
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        confidences = results[0].boxes.conf.cpu().numpy()
                        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                        count = len(boxes)
                        detections = sv.Detections(xyxy=boxes, confidence=confidences, class_id=class_ids)
                        annotated_frame = box_annotator.annotate(scene=frame.copy(), detections=detections)
                    
                    # Always draw count, even if zero
                    text_pos_x = max(10, width - 130) # Ensure text is within frame
                    cv2.putText(annotated_frame, f"People: {count}", (text_pos_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (3, 218, 25), 1, cv2.LINE_AA)
                
                    if count > MAX_PERSON_THRESHOLD:
                        alert_text_pos_x = max(10, width - 230)
                        cv2.putText(annotated_frame, "!!ALERT: MAX PERSONS !!!", (alert_text_pos_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1, cv2.LINE_AA)
                        if blob_service_client:
                            alert_thread = threading.Thread(
                                target=send_alert_to_api,
                                args=(pname_base, annotated_frame.copy(), count, blob_service_client,
                                      camera_last_alert_time_shared, shutdown_event_obj),
                                daemon=True )
                            alert_thread.name = f"Thread-{pname_base}-Alert"
                            alert_thread.start()
                except Exception as e:
                    logger.error(f"[{stream_name_for_log}] Processing error on {rtmp_url}: {e}", exc_info=True)
            
            if ffmpeg_proc and ffmpeg_proc.poll() is None:
                try:
                    ffmpeg_proc.stdin.write(annotated_frame.tobytes())
                except (IOError, BrokenPipeError) as e:
                    logger.warning(f"[{stream_name_for_log}] FFmpeg stdin write error for {output_url}: {e}. FFmpeg likely crashed.")
                    frame_processing_loop_active = False; continue 
                except Exception as e: # Catch other potential errors during write
                    logger.error(f"[{stream_name_for_log}] Unknown error writing to FFmpeg for {output_url}: {e}", exc_info=True)
                    frame_processing_loop_active = False; continue
            else:
                if not shutdown_event_obj.is_set():
                    logger.warning(f"[{stream_name_for_log}] FFmpeg process for {output_url} not available. Breaking inner loop.")
                frame_processing_loop_active = False; continue 

            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_delay - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # End of inner frame_processing_loop
        if cap: cap.release()
        if shutdown_event_obj.is_set():
            main_processing_loop_active = False; continue # Exit main loop

        if main_processing_loop_active: # If not shutting down, pause before retrying connection
             logger.info(f"[{stream_name_for_log}] Inner loop for {rtmp_url} exited. Retrying connection/setup in {RETRY_INTERVAL}s.")
             time.sleep(RETRY_INTERVAL)

    # End of outer main_processing_loop
    logger.info(f"[{stream_name_for_log}] Exiting process for {rtmp_url}.")
    if cap and cap.isOpened(): cap.release()
    if ffmpeg_proc and ffmpeg_proc.poll() is None:
        logger.info(f"[{stream_name_for_log}] Closing FFmpeg process for {output_url}.")
        try:
            if ffmpeg_proc.stdin: ffmpeg_proc.stdin.close()
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
        except Exception:
            if ffmpeg_proc.poll() is None: ffmpeg_proc.kill()
    
    if cleanup_thread.is_alive():
        logger.info(f"[{stream_name_for_log}] Waiting for memory cleanup thread to join.")
        cleanup_thread.join(timeout=5) # Wait for cleanup thread
    
    del model 
    if device and device.type == 'cuda':
        torch.cuda.empty_cache()
    gc.collect()
    logger.info(f"[{stream_name_for_log}] Resources released for {rtmp_url}.")


def main():
    global YOLO_MODEL_PATH, shutdown_event, camera_last_alert_time, active_processes

    parser = argparse.ArgumentParser(description="Multi-camera YOLOv8 + Supervision RTMP Processor")
    parser.add_argument('--yolo_model', type=str, default=YOLO_MODEL_PATH, help="Path to YOLOv8 .pt weights")
    parser.add_argument('-n', '--num_streams', type=int, default=None, help="Max number of streams to process from CSV (default: all)")
    parser.add_argument('-o', '--offset', type=int, default=0, help="Offset in CSV for selecting streams")
    parser.add_argument('--max_workers', type=int, default=MAX_CONCURRENT_PROCESSES_DEFAULT, help=f"Max concurrent processes (default: {MAX_CONCURRENT_PROCESSES_DEFAULT})")
    parser.add_argument('--batch_size', type=int, default=PROCESS_BATCH_SIZE_DEFAULT, help=f"Cameras per batch (default: {PROCESS_BATCH_SIZE_DEFAULT}) - unused in current process-per-camera model")
    args = parser.parse_args()

    YOLO_MODEL_PATH = args.yolo_model

    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    
    shutdown_event = ctx.Event()
    camera_last_alert_time = manager.dict()

    # Setup signal handlers for the main process
    # Use functools.partial to pass the mp.Event to the handler
    from functools import partial
    signal.signal(signal.SIGINT, partial(main_signal_handler, mp_shutdown_event=shutdown_event))
    signal.signal(signal.SIGTERM, partial(main_signal_handler, mp_shutdown_event=shutdown_event))

    csv_path = "camera-5.csv" 
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found: {csv_path}")
        return

    try:
        df = pd.read_csv(csv_path, header=None)
        all_urls = df[0].dropna().unique().tolist() # Get unique URLs
    except Exception as e:
        logger.error(f"Failed to read or parse CSV {csv_path}: {e}")
        return

    if not all_urls:
        logger.info("No camera URLs found in CSV. Exiting.")
        return
        
    start_index = args.offset
    end_index = len(all_urls)
    if args.num_streams is not None and args.num_streams > 0 :
        end_index = start_index + args.num_streams
    
    urls_to_process = all_urls[start_index:min(end_index, len(all_urls))]

    if not urls_to_process:
        logger.info(f"No URLs to process after applying offset/num_streams. Total unique URLs: {len(all_urls)}, Offset: {args.offset}, Num_streams: {args.num_streams}. Exiting.")
        return

    logger.info(f"Processing {len(urls_to_process)} camera streams. Max concurrent processes: {args.max_workers}.")

    procs_to_launch_objects = []
    for url in urls_to_process:
        p = ctx.Process(target=process_stream_with_retry, 
                        args=(url, shutdown_event, camera_last_alert_time))
        procs_to_launch_objects.append(p)

    launched_process_count = 0
    try:
        current_process_idx = 0
        while current_process_idx < len(procs_to_launch_objects) and not shutdown_event.is_set():
            with process_lock: # Protect active_processes list
                active_processes[:] = [p for p in active_processes if p.is_alive()]

            while len(active_processes) < args.max_workers and current_process_idx < len(procs_to_launch_objects) and not shutdown_event.is_set():
                p_obj = procs_to_launch_objects[current_process_idx]
                url_for_log = p_obj._args[0] if p_obj._args else "unknown_url" # For logging
                
                logger.info(f"Launching process for URL: {url_for_log} (Active: {len(active_processes)}, Max: {args.max_workers})")
                try:
                    p_obj.start()
                    launched_process_count += 1
                    with process_lock: # Ensure append is atomic with list modification
                        active_processes.append(p_obj)
                except Exception as e: # Catch errors during process start
                    logger.error(f"Failed to start process for {url_for_log}: {e}")
                
                current_process_idx += 1
                time.sleep(0.2) # Stagger process starts slightly to avoid thundering herd

            if shutdown_event.is_set():
                logger.info("Shutdown detected during process launching phase.")
                break
            
            if len(active_processes) >= args.max_workers and current_process_idx < len(procs_to_launch_objects):
                 logger.debug(f"Max workers ({args.max_workers}) reached. Waiting for a slot...")
            
            time.sleep(1) # Interval to check for free slots or shutdown

        logger.info(f"Process launching phase complete. {launched_process_count}/{len(procs_to_launch_objects)} processes were initiated.")
        
        logger.info("Main thread now waiting for all processes to complete or for shutdown signal.")
        while not shutdown_event.is_set():
            all_done = False
            with process_lock:
                active_processes[:] = [p for p in active_processes if p.is_alive()]
                # If all processes that were meant to be launched have been, and none are active
                if not active_processes and launched_process_count >= len(procs_to_launch_objects):
                    all_done = True
            
            if all_done:
                logger.info("All launched processes have completed their tasks.")
                break
            
            if shutdown_event.is_set(): # Check again after list modification
                break
            time.sleep(1)

    except KeyboardInterrupt: # Should be caught by signal handler setting shutdown_event
        logger.info("KeyboardInterrupt received in main. Shutdown should be in progress.")
    except Exception as e:
        logger.error(f"An unexpected error occurred in main: {e}", exc_info=True)
        if shutdown_event: shutdown_event.set() # Ensure shutdown on unexpected error
    finally:
        logger.info("Main function's finally block: ensuring shutdown and cleanup.")
        if shutdown_event and not shutdown_event.is_set():
            logger.info("Setting shutdown_event from main's finally block.")
            shutdown_event.set()
        
        # Wait a moment for processes to react to the shutdown event if they haven't already
        # This is a small grace period. cleanup_processes will handle forceful termination.
        if launched_process_count > 0: # Only if processes were started
            logger.info("Giving processes a moment to exit cleanly...")
            time.sleep(2) 

        cleanup_processes()
        
        if manager and hasattr(manager, 'shutdown'):
            try:
                logger.info("Shutting down multiprocessing manager.")
                manager.shutdown()
            except Exception as e:
                logger.error(f"Error shutting down manager: {e}")
        
        logger.info("Application shutdown sequence complete.")

if __name__ == "__main__":
    # Consider setting start method globally if issues arise, though ctx=mp.get_context("spawn") is preferred.
    # mp.set_start_method('spawn', force=True) 
    main()
