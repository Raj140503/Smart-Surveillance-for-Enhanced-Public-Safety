import csv
import os

LOG_FILE = "fall_log.csv"

def init_log():
    if not os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Video_Path", "Confidence", "Resolution"])
            print("Log file created.")
        except Exception as e:
            print("Error creating log file:", e)

def log_fall(timestamp, video_path, confidence, resolution):
    try:
        with open(LOG_FILE, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, video_path, confidence, resolution])
        print(f"Logged fall event at {timestamp} with video: {video_path}")
    except Exception as e:
        print("Error logging fall event:", e)
