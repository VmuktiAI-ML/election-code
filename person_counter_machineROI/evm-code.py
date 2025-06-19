import cv2
import numpy as np
import subprocess
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import requests
import datetime
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
import queue  # For alert queue within each process
import threading  # For alert processing thread within each process
import time
import csv
from urllib.parse import urlparse
import multiprocessing

# --- Global Config ---
PERSON_MODEL_PATH = "yolov8n.pt"
ROI_MODEL_PATH = "evm.pt"
EVM_DETECT_CONFIDENCE = 0.5  # Added EVM detection threshold

ALERT_URL = "https://tn2023demo.vmukti.com/api/analytics"
CONTAINER_NAME = "nvrdatashinobi"
BLOB_FOLDER = "live-record/frimages"

AZURE_CONN_STR = (
    "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;"
    "QueueEndpoint=https://nvrdatashinobi.queue.windows.net/;"
    "FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;"
    "TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;"
    "SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&"
    "se=2025-07-31T13:32:09Z&st=2025-03-31T05:32:09Z&spr=https,http&"
    "sig=lxI3Z67F40w8c2M3i%2FAvx7dJQNo6LU%2Bx3TVE2XM0qws%3D"
)
try:
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)
except Exception as e:
    print(f"FATAL: Could not connect to Azure Blob Storage. Error: {e}")
    exit()

COOLDOWN_SECONDS = 20
IMG_COUNT_VALUE = 4
TWO_PERSON_ANALYTICS_ID = 28
PERSON_DETECT_CONFIDENCE = 0.4
ROI_RETRY_INTERVAL_SECONDS = 10
HEIGHT_RATIO = 0.3
WIDTH_RATIO = 0.2
INTERSECTION_THRESHOLD = 0.6
TWO_PERSON_ALERT_THRESHOLD_SECONDS = 5

def process_alerts(camera_id, alert_queue_specific, process_pid):
    while True:
        try:
            alert_data = alert_queue_specific.get(timeout=1)
            if alert_data is None:
                print(f"[{camera_id} PID {process_pid}] Alert thread shutting down.")
                break

            frame_copy, timestamp_original, payload_extras = alert_data
            time_offset = datetime.timedelta(hours=5, minutes=30)
            timestamp_adjusted = timestamp_original + time_offset

            filename = f"{camera_id}_alert_{timestamp_original.strftime('%Y%m%d%H%M%S%f')[:-3]}.png"
            local_path = os.path.join(os.getcwd(), filename)
            cv2.imwrite(local_path, frame_copy)

            blob_path = f"{BLOB_FOLDER}/{filename}"
            blob_client_instance = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)
            with open(local_path, "rb") as data:
                blob_client_instance.upload_blob(
                    data,
                    overwrite=True,
                    content_settings=ContentSettings(content_type='image/png')
                )
            img_url = f"https://{blob_service_client.account_name}.blob.core.windows.net/{CONTAINER_NAME}/{blob_path}"

            payload = {
                "cameradid": camera_id,
                "sendtime": timestamp_adjusted.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                "imgurl": img_url,
            }
            payload.update(payload_extras)

            response = requests.post(ALERT_URL, json=payload, timeout=10)
            if response.ok:
                print(f"[{camera_id} PID {process_pid}] ✅ API alert sent. Status: {response.status_code}")
            else:
                print(f"[{camera_id} PID {process_pid}] ❌ API alert failed. Status: {response.status_code} - {response.text}")
            os.remove(local_path)
            alert_queue_specific.task_done()

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[{camera_id} PID {process_pid}] ❌ Exception during alert processing: {e}")

def process_camera_feed(camera_config, process_idx, num_available_gpus):
    import cv2, numpy as np, subprocess, datetime, os, queue, threading, time
    from shapely.geometry import Polygon, box
    from ultralytics import YOLO

    rtmp_input = camera_config['input_url']
    camera_id = camera_config['camera_id']
    rtmp_output = camera_config['output_url']
    current_pid = os.getpid()

    device = 'cpu'
    if num_available_gpus > 0:
        device = f'cuda:{process_idx % num_available_gpus}'

    print(f"[{camera_id} PID {current_pid}] Loading models on device: {device}...")
    try:
        person_detection_model = YOLO(PERSON_MODEL_PATH).to(device)
        roi_detection_model = YOLO(ROI_MODEL_PATH).to(device)
        print(f"[{camera_id} PID {current_pid}] Models loaded.")
    except Exception as e:
        print(f"[{camera_id} PID {current_pid}] FATAL: Could not load models: {e}")
        return

    alert_queue = queue.Queue(maxsize=10)
    alert_thread = threading.Thread(target=process_alerts, args=(camera_id, alert_queue, current_pid), daemon=True)
    alert_thread.start()

    while True:
        cap = None
        ffmpeg_process = None
        width = height = 0
        fps = 25

        try:
            cap = cv2.VideoCapture(rtmp_input)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not cap.isOpened():
                time.sleep(5)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            stream_fps = cap.get(cv2.CAP_PROP_FPS)
            fps = int(stream_fps) if stream_fps and stream_fps > 0 else 25

            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
                '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-f', 'flv', rtmp_output
            ]
            ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

            ROI_POLYGON = None
            roi_shapely_poly = None
            last_roi_attempt_time = time.time() - ROI_RETRY_INTERVAL_SECONDS

            two_person_alert_frame_count = 0
            two_person_alert_required_frames = int(fps * TWO_PERSON_ALERT_THRESHOLD_SECONDS)
            two_person_alert_sent_recently = False
            two_person_cooldown_counter = 0
            two_person_cooldown_frames = int(fps * COOLDOWN_SECONDS)

            frame_count = 0
            start_time = time.time()
            last_fps_print = time.time()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()
                now = time.time()
                frame_count += 1

                # ROI detection with threshold
                if ROI_POLYGON is None and (now - last_roi_attempt_time >= ROI_RETRY_INTERVAL_SECONDS):
                    try:
                        roi_results = roi_detection_model(display_frame, conf=EVM_DETECT_CONFIDENCE, verbose=False)
                        boxes_roi = roi_results[0].boxes.data.cpu().numpy()
                        if len(boxes_roi) > 0:
                            x1, y1, x2, y2, _, _ = boxes_roi[0]
                            bw, bh = x2 - x1, y2 - y1
                            x1a = x1 - bw * (WIDTH_RATIO / 2)
                            x2a = x2 + bw * (WIDTH_RATIO / 2)
                            y1a = y1 - bh * (HEIGHT_RATIO / 2)
                            y2a = y2 + bh * (HEIGHT_RATIO / 2)
                            ROI_POLYGON = np.array([
                                [max(0, int(x1a)), max(0, int(y1a))],
                                [min(width, int(x2a)), max(0, int(y1a))],
                                [min(width, int(x2a)), min(height, int(y2a))],
                                [max(0, int(x1a)), min(height, int(y2a))]
                            ], dtype=np.int32)
                            roi_shapely_poly = Polygon(ROI_POLYGON)
                        else:
                            print(f"[{camera_id}] ❌ No ROI detected.")
                    except Exception as e:
                        print(f"[{camera_id}] ⚠️ ROI error: {e}")
                        ROI_POLYGON = None
                    last_roi_attempt_time = now

                if ROI_POLYGON is not None:
                    cv2.polylines(display_frame, [ROI_POLYGON], True, (0,255,255), 2)

                person_count_in_roi = 0
                total_person_count = 0
                try:
                    results = person_detection_model(display_frame, conf=PERSON_DETECT_CONFIDENCE, verbose=False)
                    for det in results[0].boxes.data.cpu().numpy():
                        if int(det[5]) != 0: continue
                        total_person_count += 1
                        x1p, y1p, x2p, y2p = map(int, det[:4])
                        poly = box(x1p, y1p, x2p, y2p)
                        in_roi = False
                        if ROI_POLYGON is not None and poly.intersects(roi_shapely_poly):
                            if (poly.intersection(roi_shapely_poly).area / poly.area) > INTERSECTION_THRESHOLD:
                                person_count_in_roi += 1
                                in_roi = True
                        color = (0,0,255) if in_roi else (0,255,0)
                        cv2.rectangle(display_frame, (x1p, y1p), (x2p, y2p), color, 2)
                except Exception as e:
                    print(f"[{camera_id}] ⚠️ Person detection error: {e}")

                cv2.putText(display_frame, f"Total Persons: {total_person_count}", (400,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

                if ROI_POLYGON is not None:
                    if person_count_in_roi >= 2:
                        two_person_alert_frame_count += 1
                    else:
                        two_person_alert_frame_count = 0

                    if two_person_alert_frame_count >= two_person_alert_required_frames:
                        cv2.putText(display_frame, "Multiple Persons Near EVM", (350,60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
                        if not two_person_alert_sent_recently and not alert_queue.full():
                            payload_extras = {
                                "an_id": TWO_PERSON_ANALYTICS_ID,
                                "ImgCount": IMG_COUNT_VALUE,
                                "totalcount": total_person_count
                            }
                            alert_queue.put_nowait((display_frame.copy(), datetime.datetime.now(), payload_extras))
                            two_person_alert_sent_recently = True
                            two_person_cooldown_counter = 0

                if two_person_alert_sent_recently:
                    two_person_cooldown_counter += 1
                    if two_person_cooldown_counter >= two_person_cooldown_frames:
                        two_person_alert_sent_recently = False

                if now - last_fps_print > 5:
                    fps_val = frame_count / (now - start_time) if (now - start_time) > 0 else 0
                    print(f"[{camera_id}] 📊 FPS: {fps_val:.1f}, Q: {alert_queue.qsize()}")
                    start_time = now
                    frame_count = 0
                    last_fps_print = now

                try:
                    ffmpeg_process.stdin.write(display_frame.tobytes())
                except BrokenPipeError:
                    break
                except Exception:
                    break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[{camera_id}] ❌ Outer loop error: {e}")
        finally:
            if cap: cap.release()
            if ffmpeg_process:
                try:
                    ffmpeg_process.stdin.close()
                    ffmpeg_process.terminate()
                    ffmpeg_process.wait(timeout=2)
                except:
                    ffmpeg_process.kill()
            time.sleep(1)

    # cleanup
    alert_queue.put_nowait(None)
    alert_thread.join(timeout=5)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    CSV_FILE = "camera-13.csv"
    OUTPUT_RTMP_BASE = "rtmp://aielection.vmukti.com:80/live-record/"

    num_gpus = 0
    try:
        import torch
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
    except:
        pass

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found.")
        exit()

    camera_configs = []
    with open(CSV_FILE, 'r') as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            url = row[0].strip()
            cam_id = os.path.basename(urlparse(url).path).strip()
            out_url = f"{OUTPUT_RTMP_BASE}{cam_id}-AI"
            camera_configs.append({"input_url": url, "camera_id": cam_id, "output_url": out_url})

    if not camera_configs:
        print("No valid streams. Exiting.")
        exit()

    processes = []
    for idx, cfg in enumerate(camera_configs):
        p = multiprocessing.Process(target=process_camera_feed, args=(cfg, idx, num_gpus), daemon=True)
        processes.append(p)
        p.start()
        time.sleep(1)

    try:
        while any(p.is_alive() for p in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        for p in processes:
            if p.is_alive():
                p.terminate()
        for p in processes:
            p.join(timeout=5)
    print("Main script exiting.")

