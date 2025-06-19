import os
import cv2
import csv
import time
import json
import threading
from sort import Sort
from ultralytics import YOLO
import numpy as np
import subprocess
import requests
from datetime import datetime, timedelta # Added timedelta
import io
from azure.storage.blob import BlobServiceClient 

# --- CONFIGURATION ---
CSV_FILES = ['door.csv']
MODEL_PATH = 'yolov8n.pt'
MASK_PATH = 'mask.png' # Optional mask
PERSON_CONF_THRESHOLD = 0.4

# API and Azure Configuration
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
ANALYTICS_ID = "17" # As per your logs, or your specific ID
API_SEND_ENTRY_COUNT_MILESTONE = 5

# State Persistence Configuration
STATE_FILE_PATH = 'camera_counts_state.json'
STATE_EXPIRY_HOURS = 24
STATE_FILE_LOCK = threading.Lock() # Lock for synchronizing access to the state file

# --- GLOBAL INITIALIZATIONS ---
try:
    model = YOLO(MODEL_PATH)
except Exception as e:
    print(f"FATAL: Could not load YOLO model from {MODEL_PATH}. Error: {e}")
    exit(1)

mask_image = cv2.imread(MASK_PATH) if os.path.exists(MASK_PATH) else None
if MASK_PATH and not os.path.exists(MASK_PATH) and MASK_PATH.strip() != "":
    print(f"INFO: Mask file specified but not found at '{MASK_PATH}'. Proceeding without mask.")
elif mask_image is not None:
    print(f"INFO: Mask image '{MASK_PATH}' loaded successfully.")


# --- STATE MANAGEMENT FUNCTIONS ---
def load_camera_state(camera_id):
    with STATE_FILE_LOCK:
        try:
            if os.path.exists(STATE_FILE_PATH):
                with open(STATE_FILE_PATH, 'r') as f:
                    all_states = json.load(f)
                
                camera_state = all_states.get(camera_id)
                if camera_state:
                    timestamp_str = camera_state.get("timestamp")
                    if timestamp_str:
                        saved_time = datetime.fromisoformat(timestamp_str)
                        if datetime.now() - saved_time < timedelta(hours=STATE_EXPIRY_HOURS):
                            print(f"[{camera_id}] Loaded valid state: {camera_state}")
                            return camera_state
                        else:
                            print(f"[{camera_id}] State expired for camera. Timestamp: {saved_time}")
                    else:
                        print(f"[{camera_id}] No timestamp in saved state.")
                else:
                    print(f"[{camera_id}] No previous state found in file.")
            else:
                print(f"[{camera_id}] State file not found. Starting fresh.")
        except (json.JSONDecodeError, IOError, Exception) as e:
            print(f"[{camera_id}] Error loading state file: {e}. Starting fresh.")
    return None # Return None if no valid state found or error

def save_camera_persistent_state(camera_id, resumable_entry_count, last_api_sent_milestone):
    with STATE_FILE_LOCK:
        all_states = {}
        try:
            if os.path.exists(STATE_FILE_PATH):
                with open(STATE_FILE_PATH, 'r') as f:
                    all_states = json.load(f)
        except (json.JSONDecodeError, IOError):
            print(f"Warning: Could not read or decode existing state file '{STATE_FILE_PATH}'. Will create a new one.")
            all_states = {} # Reset if file is corrupt

        all_states[camera_id] = {
            "resumable_entry_count": resumable_entry_count,
            "last_api_sent_for_milestone": last_api_sent_milestone,
            "timestamp": datetime.now().isoformat()
        }
        try:
            with open(STATE_FILE_PATH, 'w') as f:
                json.dump(all_states, f, indent=4)
            # print(f"[{camera_id}] State saved: REC={resumable_entry_count}, LASM={last_api_sent_milestone}")
        except IOError as e:
            print(f"[{camera_id}] CRITICAL: Could not write state file '{STATE_FILE_PATH}': {e}")


# --- HELPER FUNCTIONS (UNCHANGED FROM PREVIOUS) ---
def get_output_rtmp(input_url):
    last_part = input_url.split('/')[-1]
    return f'rtmp://aielection.vmukti.com:80/live-record/{last_part}-PROCESSED'

def get_region(centroid, line_points):
    (x1_l, y1_l), (x2_l, y2_l) = line_points
    cx, cy = centroid
    cross_product = (cx - x1_l) * (y2_l - y1_l) - (cy - y1_l) * (x2_l - x1_l)
    return 'A' if cross_product > 0 else 'B' if cross_product < 0 else None

def upload_frame_to_azure(image_bytes, camera_id, container_name, blob_folder, conn_str):
    if not conn_str:
        print(f"[{camera_id}] Azure connection string not provided. Skipping upload.")
        return None
    try:
        blob_service_client = BlobServiceClient.from_connection_string(conn_str)
        timestamp_str = datetime.now().strftime('%Y%m%d%H%M%S%f')
        safe_camera_id = "".join(c if c.isalnum() or c in ['-', '_'] else '_' for c in camera_id)
        blob_name = f"{blob_folder}/{safe_camera_id}/{timestamp_str}.jpg"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
        blob_client.upload_blob(image_bytes, blob_type="BlockBlob", overwrite=True)
        print(f"[{camera_id}] Successfully uploaded frame to Azure: {blob_client.url}")
        return blob_client.url
    except Exception as e:
        print(f"[{camera_id}] Error uploading frame to Azure: {e}")
        return None

def send_alert_api(payload, alert_url, camera_id):
    print(f"[{camera_id}] Attempting to send API alert. Payload: {json.dumps(payload, indent=2)}")
    retry_delay = 1
    while True:
        try:
            response = requests.post(alert_url, json=payload, timeout=15)
            response.raise_for_status()
            print(f"[{camera_id}] API alert sent successfully. Status: {response.status_code}, Response: {response.text}")
            return True
        except requests.exceptions.HTTPError as http_err:
            print(f"[{camera_id}] HTTP error sending API alert: {http_err}. Response: {http_err.response.text if http_err.response else 'No response'}. Retrying in {retry_delay}s...")
        except requests.exceptions.ConnectionError as conn_err:
            print(f"[{camera_id}] Connection error sending API alert: {conn_err}. Retrying in {retry_delay}s...")
        except requests.exceptions.Timeout as timeout_err:
            print(f"[{camera_id}] Timeout error sending API alert: {timeout_err}. Retrying in {retry_delay}s...")
        except requests.exceptions.RequestException as req_err:
            print(f"[{camera_id}] General error sending API alert: {req_err}. Retrying in {retry_delay}s...")
        except Exception as e:
            print(f"[{camera_id}] Unexpected error in send_alert_api: {e}. Retrying in {retry_delay}s...")
        time.sleep(retry_delay)

def handle_api_and_upload(frame_to_upload, camera_id, current_total_entry_count_for_payload, alert_url, 
                          container_name, blob_folder, conn_str, analytics_id_val):
    img_url_azure = None
    is_success, buffer_img = cv2.imencode('.jpg', frame_to_upload)
    
    if not is_success:
        print(f"[{camera_id}] Failed to encode frame to JPEG for upload.")
    else:
        frame_bytes_for_upload = buffer_img.tobytes()
        img_url_azure = upload_frame_to_azure(frame_bytes_for_upload, camera_id, container_name, blob_folder, conn_str)

    api_payload = {
        "cameradid": camera_id,
        "sendtime": datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
        "imgurl": img_url_azure if img_url_azure else "",
        "an_id": analytics_id_val,
        "totalcount": current_total_entry_count_for_payload, # This is the resumable_entry_count at milestone
        "ImgCount": 4
    }
    send_alert_api(api_payload, alert_url, camera_id)


# --- CAMERA WORKER ---
def camera_worker(rtmp_url, line_points):
    camera_id = rtmp_url.split('/')[-1]
    print(f"[{camera_id}] Worker started for {rtmp_url} with line: {line_points}")
    
    # Initialize counts from persistent state or start fresh
    resumable_entry_count = 0
    last_api_sent_for_milestone = 0
    
    loaded_state = load_camera_state(camera_id)
    if loaded_state:
        resumable_entry_count = loaded_state.get("resumable_entry_count", 0)
        last_api_sent_for_milestone = loaded_state.get("last_api_sent_for_milestone", 0)
        print(f"[{camera_id}] Resuming with REC={resumable_entry_count}, LASM={last_api_sent_for_milestone}")
    else:
        # Save initial state if starting fresh
        save_camera_persistent_state(camera_id, 0, 0)
        print(f"[{camera_id}] Starting fresh with REC=0, LASM=0")

    current_session_entry_ids = set() # Tracks unique object IDs for THIS session only

    while True: # Outer loop for reconnection
        cap = None
        pipe = None
        try:
            print(f"[{camera_id}] 🎥 Trying to open: {rtmp_url}")
            cap = cv2.VideoCapture(rtmp_url)
            if not cap.isOpened():
                print(f"[{camera_id}] ❌ Could not open stream: {rtmp_url}. Retrying in 3s...")
                time.sleep(1)
                continue

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            print(f"[{camera_id}] Stream opened: {width}x{height} @ {fps} FPS")

            output_url = get_output_rtmp(rtmp_url)
            ffmpeg_cmd = [
                'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                '-s', f'{width}x{height}', '-r', str(fps), '-i', '-',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-f', 'flv', output_url
            ]
            pipe = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            print(f"[{camera_id}] FFmpeg process started for output: {output_url}")

            tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)
            prev_region = {}
            # current_session_entry_ids is reset here if we want counts per stream session
            # but for resumable count, it should persist across reconnections within the same worker instance.
            # It's correctly initialized outside the reconnection loop.

            while True: # Frame processing loop
                success, frame = cap.read()
                if not success:
                    print(f"[{camera_id}] ⚠️ Lost connection (frame read failed) to: {rtmp_url}. Reconnecting...")
                    # current_session_entry_ids.clear() # Clear session IDs if tracker is re-initialized
                    # Actually, Sort tracker is re-initialized below, so this is implicitly handled.
                    break 

                if mask_image is not None:
                    try:
                        if mask_image.shape[0] != frame.shape[0] or mask_image.shape[1] != frame.shape[1]:
                            resized_mask = cv2.resize(mask_image, (frame.shape[1], frame.shape[0]))
                        else:
                            resized_mask = mask_image
                        processed_frame = cv2.bitwise_and(frame, resized_mask)
                    except cv2.error as e:
                        print(f"[{camera_id}] Error resizing or applying mask: {e}. Using original frame.")
                        processed_frame = frame.copy()
                else:
                    processed_frame = frame.copy()
                
                detections_for_tracker = np.empty((0, 5))
                results = model(processed_frame, stream=False, verbose=False, 
                                classes=[0], conf=PERSON_CONF_THRESHOLD)

                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf_val = float(box.conf[0])
                        detections_for_tracker = np.vstack((detections_for_tracker, [x1, y1, x2, y2, conf_val]))
                
                tracked_objects = tracker.update(detections_for_tracker)
                output_img = processed_frame.copy()
                current_region = {}
                new_entry_detected_this_frame = False

                for res in tracked_objects:
                    x1_trk, y1_trk, x2_trk, y2_trk, obj_id = map(int, res)
                    cx, cy = (x1_trk + x2_trk) // 2, (y1_trk + y2_trk) // 2
                    region = get_region((cx, cy), line_points)
                    current_region[obj_id] = region

                    if obj_id in prev_region and current_region[obj_id] and prev_region[obj_id]:
                        if prev_region[obj_id] == 'A' and current_region[obj_id] == 'B':
                            if obj_id not in current_session_entry_ids: # New unique entry for this session
                                current_session_entry_ids.add(obj_id)
                                resumable_entry_count += 1
                                new_entry_detected_this_frame = True
                                print(f"[{camera_id}] New entry ID {obj_id}. Resumable count: {resumable_entry_count}")
                    
                    cv2.rectangle(output_img, (x1_trk, y1_trk), (x2_trk, y2_trk), (0, 255, 0), 2)
                    # cv2.putText(output_img, f'ID:{obj_id}', (x1_trk, y1_trk - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.circle(output_img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

                prev_region = current_region.copy()

                if new_entry_detected_this_frame: # Save state if resumable_entry_count changed
                    save_camera_persistent_state(camera_id, resumable_entry_count, last_api_sent_for_milestone)

                cv2.line(output_img, tuple(line_points[0]), tuple(line_points[1]), (0, 255, 0), 3)
                cv2.putText(output_img, f'Entry: {resumable_entry_count}', (output_img.shape[1] - 200, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                try:
                    if pipe and pipe.stdin:
                        pipe.stdin.write(output_img.tobytes())
                except (BrokenPipeError, IOError) as e:
                    print(f"[{camera_id}] ⚠️ FFmpeg pipe error for {output_url}: {e}. Restarting FFmpeg.")
                    break 

                # API Call Logic based on resumable_entry_count milestones
                if resumable_entry_count > 0 and \
                   resumable_entry_count % API_SEND_ENTRY_COUNT_MILESTONE == 0 and \
                   resumable_entry_count > last_api_sent_for_milestone:
                    
                    print(f"[{camera_id}] Milestone {resumable_entry_count} reached. Last sent for: {last_api_sent_for_milestone}. Preparing API alert.")
                    frame_copy_for_upload = output_img.copy()
                    
                    # Update last_api_sent_for_milestone *before* spawning thread to prevent re-sends for this milestone
                    current_milestone_to_send = resumable_entry_count
                    last_api_sent_for_milestone = current_milestone_to_send 
                    save_camera_persistent_state(camera_id, resumable_entry_count, last_api_sent_for_milestone) # Save updated milestone
                    
                    api_upload_thread = threading.Thread(
                        target=handle_api_and_upload,
                        args=(
                            frame_copy_for_upload, camera_id, current_milestone_to_send, # Pass the milestone count
                            ALERT_URL, CONTAINER_NAME, BLOB_FOLDER, AZURE_CONN_STR, ANALYTICS_ID
                        ),
                        daemon=True
                    )
                    api_upload_thread.start()
            
            if cap: cap.release()
            if pipe:
                if pipe.stdin:
                    try: pipe.stdin.close()
                    except BrokenPipeError: pass
                pipe.terminate(); pipe.wait()
                print(f"[{camera_id}] FFmpeg process for {output_url} stopped.")
            
            print(f"[{camera_id}] Stream loop ended. Will attempt to reconnect/restart in 1s.")
            # When stream reconnects, tracker is new, so current_session_entry_ids should be cleared
            # to correctly count new unique IDs from the new tracker instance.
            current_session_entry_ids.clear()
            time.sleep(1)

        except Exception as e:
            print(f"[{camera_id}] ❌ Unexpected error in camera_worker for {rtmp_url}: {e}")
            import traceback; traceback.print_exc()
            if cap and cap.isOpened(): cap.release()
            if pipe and pipe.poll() is None:
                if pipe.stdin:
                    try: pipe.stdin.close()
                    except Exception: pass
                pipe.terminate(); pipe.wait()
            print(f"[{camera_id}] Restarting worker after unexpected error in 5s...")
            current_session_entry_ids.clear() # Also clear on major error before restart
            time.sleep(1)


# --- MAIN EXECUTION ---
def run_all_cameras():
    cameras_to_process = []
    for file_path in CSV_FILES:
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            continue
        try:
            with open(file_path, 'r') as f:
                for i, line_content in enumerate(f):
                    line_content = line_content.strip()
                    if not line_content: continue
                    parts = line_content.split(',', 1)
                    if len(parts) == 2:
                        rtmp_url_csv, line_json_str = parts[0].strip(), parts[1].strip()
                        if rtmp_url_csv.startswith("rtmp"):
                            try:
                                line_data = json.loads(line_json_str)
                                if 'line' in line_data and isinstance(line_data['line'], list) and len(line_data['line']) == 2:
                                    line_points_csv = [tuple(map(int, p)) for p in line_data['line']]
                                    cameras_to_process.append({'url': rtmp_url_csv, 'line': line_points_csv})
                                    print(f"✅ Scheduled: {rtmp_url_csv} with line {line_points_csv}")
                                else: print(f"⚠️ Invalid line format in JSON for row {i+1} in {file_path}. JSON: '{line_json_str}'")
                            except json.JSONDecodeError as e: print(f"⚠️ JSONDecodeError for line coordinates in row {i+1} in {file_path}. Error: {e}. JSON: '{line_json_str}'")
                            except Exception as e: print(f"⚠️ Error processing line data for row {i+1} in {file_path}: {e}. Line: '{line_content}'")
                        else: print(f"⚠️ Row {i+1} in {file_path} does not start with 'rtmp': '{rtmp_url_csv}'")
                    else: print(f"⚠️ Row {i+1} in {file_path} format error (URL,JSON_string). Content: '{line_content}'")
        except Exception as e: print(f"❌ Error reading/processing CSV {file_path}: {e}")

    if not cameras_to_process:
        print("No valid camera URLs and line coordinates found. Exiting.")
        return

    threads = []
    for camera_info in cameras_to_process:
        thread = threading.Thread(target=camera_worker, args=(camera_info['url'], camera_info['line']), daemon=True)
        threads.append(thread); thread.start()
        time.sleep(1) 

    try:
        while True:
            if not any(t.is_alive() for t in threads) and threads:
                print("All camera threads have finished or died. Exiting main program.")
                break
            time.sleep(10)
    except KeyboardInterrupt: print("\n🔴 KeyboardInterrupt received. Exiting main program...")

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH): print(f"FATAL: YOLO model not found at {MODEL_PATH}."); exit(1)
    if not CSV_FILES: print("FATAL: No CSV files specified."); exit(1)
    if not any(os.path.exists(f) for f in CSV_FILES): print(f"FATAL: None of specified CSV files found: {CSV_FILES}"); exit(1)

    print("Starting camera processing system...")
    print(f"State file: {STATE_FILE_PATH}, State expiry: {STATE_EXPIRY_HOURS} hours")
    print(f"Person detection confidence: {PERSON_CONF_THRESHOLD}")
    print(f"API alerts at entry count multiples of: {API_SEND_ENTRY_COUNT_MILESTONE}")
    print(f"API Analytics ID (an_id): {ANALYTICS_ID}")
    if mask_image is None and MASK_PATH and MASK_PATH.strip() != "": print(f"Warning: Mask '{MASK_PATH}' specified but not loaded.")
    elif mask_image is None: print("INFO: No mask image. Processing full frames.")

    run_all_cameras()
    print("Main program has finished.")
