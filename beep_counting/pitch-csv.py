import subprocess
import numpy as np
import librosa
import datetime
from collections import deque
import time
import pyodbc
import csv
import threading
from queue import Queue

# --- Configuration ---
CHUNK_DURATION_MS = 100
SAMPLE_RATE = 44100
threshold_low = 3100
threshold_high = 4000
detection_duration = 3.0
detection_window = 4.0
max_history_duration = 180
peak_threshold = 3300
min_peaks = 5
cooldown_seconds = 10.0

# --- Database Config ---
DB_SERVER = "98.70.48.159"
DB_NAME = "westbengal2025"
DB_USERNAME = "vmukti"
DB_PASSWORD = "bhargav@123456"

# --- Max concurrent streams ---
MAX_WORKERS = 10  # Adjust depending on your CPU/memory


def log_beep_to_db(camera_id, video_datetime, video_time, bitrate=1, flag='Update'):
    try:
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={DB_SERVER};DATABASE={DB_NAME};UID={DB_USERNAME};PWD={DB_PASSWORD}"
        )
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC SaveBeepfromCamera ?, ?, ?, ?, ?",
                camera_id, video_datetime, video_time, bitrate, flag
            )
            conn.commit()
        print(f"[DB] Logged beep to DB for CameraID={camera_id} at {video_time}")
    except Exception as e:
        print(f"[ERROR] DB Logging Failed for CameraID={camera_id}: {e}")


def process_stream(video_url):
    camera_id = video_url.split('/')[-1]
    print(f"[INFO][{camera_id}] Starting stream processing: {video_url}")

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", video_url,
        "-vn",
        "-f", "s16le",
        "-acodec", "pcm_s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-"
    ]

    pitch_history = deque()
    time_history = deque()
    logical_time = 0.0
    count = 0
    last_detection_time = -cooldown_seconds

    try:
        process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print(f"[ERROR][{camera_id}] Failed to start ffmpeg process: {e}")
        return

    try:
        while True:
            bytes_needed = int(SAMPLE_RATE * 2 * CHUNK_DURATION_MS / 1000)  # 2 bytes per sample (int16)
            raw = b""
            read_attempts = 0
            max_attempts = 20

            while len(raw) < bytes_needed and read_attempts < max_attempts:
                chunk = process.stdout.read(bytes_needed - len(raw))
                if not chunk:
                    read_attempts += 1
                    print(f"[WARNING][{camera_id}] No audio data received. Retrying...")
                    time.sleep(0.05)
                else:
                    raw += chunk

            if len(raw) < bytes_needed:
                print(f"[ERROR][{camera_id}] Skipping chunk due to insufficient audio.")
                logical_time += CHUNK_DURATION_MS / 1000.0
                continue

            audio_chunk = np.frombuffer(raw, dtype=np.int16)
            if len(audio_chunk) == 0:
                print(f"[WARNING][{camera_id}] Empty audio buffer. Skipping.")
                logical_time += CHUNK_DURATION_MS / 1000.0
                continue

            y_float = audio_chunk.astype(np.float32) / 32768.0

            pitches, magnitudes = librosa.piptrack(y=y_float, sr=SAMPLE_RATE)
            pitch_values = [
                pitches[magnitudes[:, i].argmax(), i]
                for i in range(pitches.shape[1])
                if pitches[magnitudes[:, i].argmax(), i] > 0
            ]
            avg_pitch = np.mean(pitch_values) if pitch_values else 0

            pitch_history.append(avg_pitch)
            time_history.append(logical_time)

            while time_history and (logical_time - time_history[0]) > max_history_duration:
                pitch_history.popleft()
                time_history.popleft()

            window_pitches = [p for t, p in zip(time_history, pitch_history) if logical_time - t <= detection_window]
            in_range_count = sum(threshold_low <= p <= threshold_high for p in window_pitches)
            peak_count = sum(p > peak_threshold for p in window_pitches)
            duration_in_range = in_range_count * (CHUNK_DURATION_MS / 1000.0)

            timestamp = str(datetime.timedelta(seconds=int(logical_time)))
            realtime_now = datetime.datetime.now()
            print(f"[INFO][{camera_id}] Time: {timestamp} | Pitch: {avg_pitch:.2f} Hz")

            if (duration_in_range >= detection_duration or peak_count >= min_peaks) and \
                    (logical_time - last_detection_time > cooldown_seconds):
                count += 1
                print(f"[DETECTED][{camera_id}] Beep #{count} at {timestamp} (Real Time: {realtime_now})")
                last_detection_time = logical_time

                log_beep_to_db(camera_id, realtime_now.strftime("%Y-%m-%d %H:%M:%S"), timestamp, bitrate=1, flag='Update')

            logical_time += CHUNK_DURATION_MS / 1000.0

    except KeyboardInterrupt:
        print(f"\n[INFO][{camera_id}] Stopped by user.")
    except Exception as e:
        print(f"[ERROR][{camera_id}] Exception: {e}")
    finally:
        process.kill()
        print(f"[INFO][{camera_id}] Stream processing ended.")


def worker(queue):
    while True:
        video_url = queue.get()
        if video_url is None:
            break
        process_stream(video_url)
        queue.task_done()


def main():
    # Read CSV file of URLs (one per line, first column)
    input_csv = "rtmp_urls.csv"
    urls = []
    with open(input_csv, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if row and row[0].strip():
                urls.append(row[0].strip())

    print(f"[INFO] Loaded {len(urls)} RTMP URLs from {input_csv}")

    # Create a queue and threads pool
    q = Queue()
    threads = []
    for _ in range(min(MAX_WORKERS, len(urls))):
        t = threading.Thread(target=worker, args=(q,), daemon=True)
        t.start()
        threads.append(t)

    # Enqueue all URLs
    for url in urls:
        q.put(url)

    # Wait for all URLs to finish
    try:
        q.join()
    except KeyboardInterrupt:
        print("[INFO] Keyboard interrupt detected, shutting down...")

    # Stop threads
    for _ in threads:
        q.put(None)
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()