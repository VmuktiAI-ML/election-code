import cv2
import numpy as np
import subprocess
from shapely.geometry import Polygon, box
from ultralytics import YOLO
import requests
import datetime
import os
from azure.storage.blob import BlobServiceClient, ContentSettings
import threading
import queue
import time
import csv
from urllib.parse import urlparse # For parsing RTMP URLs

# --- Global Config ---
# Model paths remain global, loaded once
PERSON_MODEL_PATH = "yolov8n.pt"
ROI_MODEL_PATH = "best.pt"

# API Alert Config remains global
ALERT_URL = "https://tn2023demo.vmukti.com/api/analytics"
CONTAINER_NAME = "nvrdatashinobi"
BLOB_FOLDER = "live-record/frimages"

# Azure Blob Storage Connection remains global
AZURE_CONN_STR = (
    "BlobEndpoint=https://nvrdatashinobi.blob.core.windows.net/;"
    "QueueEndpoint=https://nvrdatashinobi.queue.windows.net/;"
    "FileEndpoint=https://nvrdatashinobi.file.core.windows.net/;"
    "TableEndpoint=https://nvrdatashinobi.table.core.windows.net/;"
    "SharedAccessSignature=sv=2024-11-04&ss=bfqt&srt=sco&sp=rwdlacupiytfx&"
    "se=2025-07-31T13:32:09Z&st=2025-03-31T05:32:09Z&spr=https,http&"
    "sig=lxI3Z67F40w8c2M3i%2FAvx7dJQNo6LU%2Bx3TVE2XM0qws%3D"
)
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STR)

# Parameters remain global (or can be passed per camera if needed)
height_ratio = 0.3
width_ratio = 0.2
person_detect_confidence = 0.5
intersection_threshold = 0.6
alert_threshold_seconds = 10
COOLDOWN_SECONDS = 20 # Cooldown period for alerts
ROI_RETRY_INTERVAL_SECONDS = 10 # How often to try detecting ROI if not found

# --- Load Models (Global, loaded once, shared across threads) ---
print("Loading YOLO models...")
person_model = YOLO(PERSON_MODEL_PATH)
roi_model = YOLO(ROI_MODEL_PATH)
print("Models loaded successfully.")

# --- Alert Processing Thread Function (modified to take a specific queue) ---
def process_alerts(camera_id, alert_queue_specific):
    """Background thread to handle alert processing for a specific camera"""
    while True:
        try:
            alert_data = alert_queue_specific.get(timeout=1)
            if alert_data is None:  # Shutdown signal
                print(f"[{camera_id}] Alert thread shutting down.")
                break
                
            frame_copy, total_person_count, timestamp = alert_data # Renamed `person_count` to `total_person_count` for clarity
            
            try:
                # Use a unique filename for each camera + timestamp
                filename = f"{camera_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')[:-3]}.png"
                local_path = os.path.join(os.getcwd(), filename)

                # Save image
                cv2.imwrite(local_path, frame_copy)

                # Upload to Azure Blob Storage
                blob_path = f"{BLOB_FOLDER}/{filename}"
                blob_client = blob_service_client.get_blob_client(container=CONTAINER_NAME, blob=blob_path)

                with open(local_path, "rb") as data:
                    blob_client.upload_blob(
                        data,
                        overwrite=True,
                        content_settings=ContentSettings(content_type='image/png')
                    )
                img_url = f"https://nvrdatashinobi.blob.core.windows.net/{CONTAINER_NAME}/{blob_path}"

                # Prepare and send API payload
                payload = {
                    "cameradid": camera_id,
                    "sendtime": timestamp.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                    "imgurl": img_url,
                    "an_id": 28, # Analytics ID for "Two Person Alert"
                    "ImgCount": 4, # Placeholder? Adjust as needed
                    "totalcount": total_person_count # This now explicitly uses the total count passed
                }

                response = requests.post(ALERT_URL, json=payload, timeout=10)
                if response.ok:
                    print(f"[{camera_id}] ✅ API sent. Status: {response.status_code}")
                else:
                    print(f"[{camera_id}] ❌ API failed. Status: {response.status_code} - {response.text}")

                # Clean up local file
                try:
                    os.remove(local_path)
                except OSError as e:
                    print(f"[{camera_id}] ⚠️ Could not remove local file {local_path}: {e}")

            except requests.exceptions.RequestException as e:
                print(f"[{camera_id}] ❌ Network error sending API alert: {e}")
            except Exception as e:
                print(f"[{camera_id}] ❌ Exception during alert processing: {e}")
            
            finally:
                alert_queue_specific.task_done()
                
        except queue.Empty:
            continue # No alerts in queue, keep waiting
        except Exception as e:
            print(f"[{camera_id}] ❌ Alert thread critical exception: {e}")

# --- Main Camera Processing Function ---
def process_camera_feed(camera_config):
    rtmp_input = camera_config['input_url']
    camera_id = camera_config['camera_id']
    rtmp_output = camera_config['output_url']
    
    print(f"[{camera_id}] Starting processing for input: {rtmp_input}")

    # Initialize per-camera alert system
    alert_queue = queue.Queue(maxsize=5)
    alert_thread = threading.Thread(target=process_alerts, args=(camera_id, alert_queue), daemon=True)
    alert_thread.start()

    # Outer loop for stream connection and full reconnection retry
    while True:
        cap = None
        ffmpeg_process = None
        width, height, fps = 0, 0, 20 # Default FPS values

        try:
            print(f"[{camera_id}] Attempting to open input stream...")
            cap = cv2.VideoCapture(rtmp_input)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Reduce buffer size for lower latency

            if not cap.isOpened():
                print(f"[{camera_id}] ❌ Could not open input stream. Retrying in 1 second...")
                time.sleep(1)
                continue # Retry outer loop

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20 # Fallback to 20 FPS

            if width == 0 or height == 0:
                print(f"[{camera_id}] ❌ Invalid frame dimensions ({width}x{height}). Retrying in 1 second...")
                cap.release()
                time.sleep(1)
                continue # Retry outer loop

            print(f"[{camera_id}] ✅ Input stream opened: {width}x{height} @ {fps} FPS")

            # --- FFmpeg Output Setup ---
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo', '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24', '-s', f'{width}x{height}', '-r', str(fps),
                '-i', '-', '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                '-preset', 'ultrafast', '-tune', 'zerolatency',
                '-f', 'flv', rtmp_output
            ]
            
            print(f"[{camera_id}] Starting FFmpeg process for output: {rtmp_output}")
            try:
                ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
                # Note: For robust error handling, consider reading stderr from ffmpeg in a separate thread.
            except FileNotFoundError:
                print(f"[{camera_id}] ❌ FFmpeg not found. Please ensure FFmpeg is installed and in your PATH.")
                time.sleep(5) # Long wait, as this is a setup issue
                continue # Retry outer loop
            except Exception as e:
                print(f"[{camera_id}] ❌ Failed to start FFmpeg: {e}. Retrying connection...")
                cap.release() # Release cap before retrying ffmpeg
                time.sleep(5)
                continue # Retry outer loop
            
            print(f"[{camera_id}] ✅ FFmpeg process started.")

            # --- Dynamic ROI management variables ---
            ROI_POLYGON = None
            roi_shapely_poly = None
            last_roi_attempt_time = time.time() - ROI_RETRY_INTERVAL_SECONDS # Force immediate first attempt
            
            # --- Main Frame Processing Loop ---
            alert_frame_count = 0
            alert_required_frames = int(fps * alert_threshold_seconds)
            alert_sent_recently = False
            cooldown_counter = 0
            cooldown_frames = int(fps * COOLDOWN_SECONDS)

            frame_count = 0
            start_time = time.time()
            last_fps_print = time.time()

            # Inner loop for continuous frame processing
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"[{camera_id}] ⚠️ Frame read failed or stream ended. Reconnecting stream...")
                    break # Break from inner loop to trigger outer reconnection logic

                # --- Always write frame to FFmpeg ---
                try:
                    display_frame = frame.copy() 
                except Exception as e:
                    print(f"[{camera_id}] ❌ Error copying frame: {e}. Reconnecting...")
                    break # Reconnect if frame copy fails unexpectedly

                frame_count += 1
                
                # --- Dynamic ROI Detection (attempts if not set) ---
                if ROI_POLYGON is None:
                    current_time = time.time()
                    if current_time - last_roi_attempt_time >= ROI_RETRY_INTERVAL_SECONDS:
                        print(f"[{camera_id}] Attempting ROI detection...")
                        roi_results = roi_model(display_frame) # Use current frame for ROI detection
                        boxes = roi_results[0].boxes.data.cpu().numpy()
                        
                        if len(boxes) > 0:
                            x1, y1, x2, y2, conf, cls = boxes[0] # Take the first detected ROI
                            box_w, box_h = x2 - x1, y2 - y1
                            
                            # Apply aspect ratio adjustments
                            x1_adj = x1 - box_w * (width_ratio / 2)
                            x2_adj = x2 + box_w * (width_ratio / 2)
                            y1_adj = y1 - box_h * (height_ratio / 2)
                            y2_adj = y2 + box_h * (height_ratio / 2)
                            
                            # Clamp to frame boundaries
                            x1_adj, y1_adj, x2_adj, y2_adj = map(int, [
                                max(0, x1_adj), max(0, y1_adj), 
                                min(width, x2_adj), min(height, y2_adj)
                            ])
                            
                            ROI_POLYGON = np.array([[x1_adj, y1_adj], [x2_adj, y1_adj], [x2_adj, y2_adj], [x1_adj, y2_adj]], dtype=np.int32)
                            roi_shapely_poly = Polygon(ROI_POLYGON)
                            print(f"[{camera_id}] ✅ ROI polygon detected: {ROI_POLYGON.tolist()}")
                        else:
                            print(f"[{camera_id}] ❌ No ROI object detected. Will retry in {ROI_RETRY_INTERVAL_SECONDS} seconds.")
                        last_roi_attempt_time = current_time # Reset timer regardless of detection success

                else:
                    # Draw ROI polygon if it's detected
                    cv2.polylines(display_frame, [ROI_POLYGON], isClosed=True, color=(0, 255, 255), thickness=2)


                # --- Always perform person detection ---
                results = person_model(display_frame, conf=person_detect_confidence)
                detections = results[0].boxes.data.cpu().numpy()
                
                person_count_roi = 0 
                total_person_count_in_frame = 0 # Initialize for current frame

                for det in detections:
                    x1, y1, x2, y2, conf, cls = det
                    if int(cls) != 0:  # Only process person class (class 0 in COCO)
                        continue

                    # Increment total person count regardless of ROI
                    total_person_count_in_frame += 1 

                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    person_poly = box(x1, y1, x2, y2)
                    
                    is_in_roi = False
                    # Check ROI intersection only if ROI is defined
                    if ROI_POLYGON is not None and person_poly.intersects(roi_shapely_poly):
                        inter_area = person_poly.intersection(roi_shapely_poly).area
                        person_area = person_poly.area
                        # Avoid division by zero if person_area is 0 (very small or invalid box)
                        ixa = inter_area / person_area if person_area > 0 else 0

                        if ixa > intersection_threshold:
                            person_count_roi += 1
                            is_in_roi = True

                    # Draw bounding box based on ROI status
                    if is_in_roi:
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red for ROI
                        # percent_text = f"Int: {ixa * 100:.1f}%"
                        # cv2.putText(display_frame, percent_text, (x1, y1 - 10),                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        # If not in ROI, or if ROI is not detected at all, draw green
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green for non-ROI/no-ROI

                # --- Display total person count on the frame ---
                cv2.putText(display_frame, 
                            f"Total Persons: {total_person_count_in_frame}", 
                            (450, 30), # Top-left corner
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5,      # Font scale
                            (255, 0, 0), # Green color (BGR)
                            1,        # Thickness
                            cv2.LINE_AA)

                # --- Alert logic (only if ROI is active) ---
                if ROI_POLYGON is not None: # Only run alert logic if ROI is set
                    if person_count_roi >= 2: # Alert condition is still based on persons in ROI
                        alert_frame_count += 1
                    else:
                        alert_frame_count = 0

                    if alert_frame_count >= alert_required_frames:
                        # Display alert message
                        cv2.putText(display_frame, f"Multiple Persons Detected Near EVM", (350, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                        # Send alert asynchronously only if queue is not full and cooldown has passed
                        if not alert_sent_recently and not alert_queue.full():
                            try:
                                timestamp = datetime.datetime.now()
                                frame_copy = display_frame.copy() # Make a copy for the alert thread
                                # Pass total_person_count_in_frame for the API payload
                                alert_queue.put_nowait((frame_copy, total_person_count_in_frame, timestamp)) 
                                alert_sent_recently = True
                                cooldown_counter = 0
                                print(f"[{camera_id}] 🚨 Alert queued - {person_count_roi} persons in ROI, {total_person_count_in_frame} total in frame.") # Updated log
                            except queue.Full:
                                print(f"[{camera_id}] ⚠️ Alert queue full, skipping alert for now.")

                    # Cooldown management for alerts
                    if alert_sent_recently:
                        cooldown_counter += 1
                        if cooldown_counter >= cooldown_frames:
                            alert_sent_recently = False
                else:
                    # Reset alert state if ROI is not active
                    alert_frame_count = 0
                    alert_sent_recently = False
                    cooldown_counter = 0


                # Performance monitoring
                current_time = time.time()
                if current_time - last_fps_print > 5:  # Print FPS every 5 seconds
                    elapsed = current_time - start_time
                    current_fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"[{camera_id}] 📊 Processing FPS: {current_fps:.1f}, Queue size: {alert_queue.qsize()}")
                    last_fps_print = current_time

                # Write processed frame to FFmpeg
                try:
                    ffmpeg_process.stdin.write(display_frame.tobytes())
                except Exception as e:
                    print(f"[{camera_id}] ❌ FFmpeg write error: {e}. Reconnecting stream...")
                    break # Break from inner loop to trigger full reconnect

        except KeyboardInterrupt:
            print(f"[{camera_id}] 🛑 Stopping processing due to KeyboardInterrupt...")
            break # Break from outer loop to exit thread
        except Exception as e:
            print(f"[{camera_id}] ❌ Unhandled error in main processing loop: {e}")
            break # Break from inner loop to try re-establishing connection
        
        finally:
            # This finally block runs when the inner while loop breaks or an exception occurs within it.
            # It ensures that video capture and ffmpeg process are released before re-attempting connection.
            print(f"[{camera_id}] 🧹 Cleaning up resources for reconnection...")
            if cap:
                cap.release()
                print(f"[{camera_id}] VideoCapture released.")
            if ffmpeg_process:
                try:
                    ffmpeg_process.stdin.close()
                    print(f"[{camera_id}] FFmpeg stdin closed.")
                    if ffmpeg_process.poll() is None: # None means process is still running
                        ffmpeg_process.wait(timeout=2)
                    if ffmpeg_process.poll() is None:
                        print(f"[{camera_id}] FFmpeg process still running, terminating...")
                        ffmpeg_process.terminate()
                except Exception as e:
                    print(f"[{camera_id}] Error during FFmpeg cleanup for reconnection: {e}")
            
            # Small delay before trying to reconnect the stream in the outer loop
            time.sleep(1) 

    # This part runs only when the outermost `while True` loop is broken (e.g., by KeyboardInterrupt)
    print(f"[{camera_id}] 🧹 Final cleanup for thread exit...")
    # Signal alert thread to stop and wait for it
    try:
        alert_queue.put_nowait(None) # Send shutdown signal
        alert_thread.join(timeout=5) # Wait for alert thread to finish
        if alert_thread.is_alive():
            print(f"[{camera_id}] Alert thread did not terminate gracefully.")
    except Exception as e:
        print(f"[{camera_id}] Error during alert thread cleanup: {e}")
        
    print(f"[{camera_id}] ✅ Thread for {camera_id} exited.")


# --- Main Script Execution ---
if __name__ == "__main__":
    CSV_FILE = "streams.csv"
    OUTPUT_RTMP_BASE = "rtmp://aielection.vmukti.com:80/live-record/"
    threads = []

    if not os.path.exists(CSV_FILE):
        print(f"Error: {CSV_FILE} not found. Please create it with camera RTMP URLs.")
        exit()

    try:
        with open(CSV_FILE, 'r') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                if not row:
                    continue # Skip empty rows
                input_url = row[0].strip()
                if not input_url:
                    continue

                # Parse the camera ID from the input RTMP URL
                parsed_url = urlparse(input_url)
                # Get the last segment of the path, which is usually the camera ID
                camera_id = os.path.basename(parsed_url.path).strip()
                
                if not camera_id:
                    print(f"Warning: Could not extract camera ID from {input_url}. Skipping.")
                    continue

                # Construct the dynamic output RTMP URL
                output_url = f"{OUTPUT_RTMP_BASE}{camera_id}-Box_Alert_AI"

                camera_config = {
                    "input_url": input_url,
                    "camera_id": camera_id,
                    "output_url": output_url
                }
                
                print(f"Setting up thread for Camera ID: {camera_id}, Input: {input_url}, Output: {output_url}")
                
                # Start a new thread for each camera feed
                thread = threading.Thread(target=process_camera_feed, args=(camera_config,), daemon=True)
                threads.append(thread)
                thread.start()
                
                # Small delay to prevent overwhelming system if many cameras
                time.sleep(0.1) 

        print(f"\nStarted {len(threads)} camera processing threads. Press Ctrl+C to stop all.\n")

        # Keep the main thread alive so daemon threads can run
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\nReceived KeyboardInterrupt in main thread. All daemon threads will attempt to clean up and exit.")
        # Daemon threads automatically terminate when the main thread exits.
        # Their `finally` blocks will be called.
        
    except Exception as e:
        print(f"An unexpected error occurred in the main script: {e}")

    finally:
        print("Main script exiting.")