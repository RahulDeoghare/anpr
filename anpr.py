import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import sys
import json
from datetime import datetime
import re
import boto3
from botocore.client import Config



# Path to your folder and model
image_folder_1 = "/home/ubantu/vms/data/screenshots"  # First folder
image_folder_2 = "/home/ubantu/vms/data/anpr_ss/"    # Second folder
model_path = "truck.pt"        # Path to your YOLO model
output_folder = "/home/ubantu/anpr/output"  # Folder to save annotated images
os.makedirs(output_folder, exist_ok=True)

# Local folder for storing date-based JSONs
local_json_folder = "/home/ubantu/anpr/json_local"
os.makedirs(local_json_folder, exist_ok=True)

# DigitalOcean Spaces credentials
DO_SPACES_KEY = 'DO801UYGLUGLVCDQFYNM'
DO_SPACES_SECRET = 'fBDdr0Cp5NmbkSkD0jeRgE+oIaOZcOdSfzOautQGnL4'
DO_SPACES_REGION = 'blr1'
DO_SPACES_ENDPOINT = 'https://blr1.digitaloceanspaces.com'
DO_SPACES_BUCKET = 'vigilscreenshots'
DO_SPACES_FOLDER = 'anpr_json'

# Setup boto3 client for DigitalOcean Spaces
session = boto3.session.Session()
s3_client = session.client('s3',
    region_name=DO_SPACES_REGION,
    endpoint_url=DO_SPACES_ENDPOINT,
    aws_access_key_id=DO_SPACES_KEY,
    aws_secret_access_key=DO_SPACES_SECRET,
    config=Config(signature_version='s3v4')
)

# Load YOLO model
model = YOLO(model_path)

# Initialize PaddleOCR (English, CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
processed_images = set()

month_dict = {'Jan': '01', 'Feb': '02', 'Mar': '03', 'Apr': '04', 'May': '05', 'Jun': '06',
              'Jul': '07', 'Aug': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dec': '12'}

# Store results by date

# Store results for the current date only
current_date = None
current_date_results = []

print(f"Watching folders: {image_folder_1} and {image_folder_2} for new images...")
sys.stdout.flush()
while True:

    # Define allowed prefixes for each folder
    allowed_prefixes_1 = ("exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d", "exit_192.168.1.104", "exit_uuid_53b3850d-e0ef-4668-9fb5-12c980aac83d")
    allowed_prefixes_2 = ("anpr_uuid_53b3850d-e0ef-4668-9fb5-12c980aac83d",)

    # Collect images from both folders
    image_files_1 = [f for f in glob.glob(os.path.join(image_folder_1, "*"))
                     if f.lower().endswith(image_extensions)
                     and os.path.basename(f).startswith(allowed_prefixes_1)]
    image_files_2 = [f for f in glob.glob(os.path.join(image_folder_2, "*"))
                     if f.lower().endswith(image_extensions)
                     and os.path.basename(f).startswith(allowed_prefixes_2)]

    # Merge and filter new images
    all_image_files = image_files_1 + image_files_2
    new_images = [f for f in all_image_files if f not in processed_images]
    if not new_images:
        print("[INFO] Waiting for new images...")
        sys.stdout.flush()
        time.sleep(2)
        continue

    # Sort new_images by timestamp from filename
    def get_datetime_from_filename(path):
        base = os.path.basename(path)
        match = re.search(r'_(\d{2})_(\w{3})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.', base)
        if match:
            day, mon, year, h, m, s = match.groups()
            mon_num = month_dict.get(mon, '01')
            dt_str = f"{year}-{mon_num}-{day} {h}:{m}:{s}"
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
        return datetime.min

    new_images.sort(key=get_datetime_from_filename)

    for image_path in new_images:
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image:", image_path)
            processed_images.add(image_path)
            continue
        results = model.predict(image, verbose=False)[0]
        output_image = image.copy()
        print(f"\nProcessing: {os.path.basename(image_path)}")
        found_plate = False
        # Extract time and date from filename
        base_name = os.path.basename(image_path)
        date_match = re.search(r'_(\d{2})_(\w{3})_(\d{4})_(\d{2})_(\d{2})_(\d{2})\.', base_name)
        if date_match:
            day = date_match.group(1)
            month_str = date_match.group(2)
            year = date_match.group(3)
            hour = date_match.group(4)
            minute = date_match.group(5)
            second = date_match.group(6)
            month = month_dict.get(month_str, '01')  # default to 01 if not found
            date_str = f"{day}-{month}-{year}"
            time_str = f"{hour}:{minute}:{second}"
            # Compute timestamp_seconds from parsed date and time
            dt = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M:%S")
            timestamp_seconds = dt.timestamp()
        else:
            now = datetime.now()
            date_str = now.strftime("%d-%m-%Y")
            time_str = now.strftime("%H:%M:%S")
            timestamp_seconds = time.time()

        # If we move to a new date, flush the previous date's results
        if current_date is not None and date_str != current_date and current_date_results:
            out_json_path = os.path.join(output_folder, f"{current_date}.json")
            local_json_path = os.path.join(local_json_folder, f"{current_date}.json")
            with open(out_json_path, 'w') as outjf:
                json.dump(current_date_results, outjf, indent=2)
            with open(local_json_path, 'w') as localjf:
                json.dump(current_date_results, localjf, indent=2)
            # Upload to DigitalOcean Spaces
            s3_key = f"{DO_SPACES_FOLDER}/{current_date}.json"
            try:
                s3_client.upload_file(out_json_path, DO_SPACES_BUCKET, s3_key)
                print(f"[INFO] Uploaded {out_json_path} to s3://{DO_SPACES_BUCKET}/{s3_key}")
            except Exception as e:
                print(f"[ERROR] Failed to upload {out_json_path} to DigitalOcean Spaces: {e}")
            current_date_results = []

        current_date = date_str

        if results.boxes and len(results.boxes) > 0:
            all_plates = []
            all_confs = []
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                if plate_img.shape[0] < 5 or plate_img.shape[1] < 5:
                    continue
                ocr_result = ocr.ocr(plate_img)
                detected_texts = []
                best_conf = 0.0
                if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        detected_texts.append(text)
                        if conf > best_conf:
                            best_conf = conf
                        print(f"  Plate {i+1} Text: {text} | Confidence: {conf:.2f}")
                    # Print concatenated result in a single line (no separator)
                    concat_text = ''.join(detected_texts)
                    if concat_text:
                        print(f"  Plate {i+1} Single Line: {concat_text}")
                        all_plates.append(concat_text)
                        all_confs.append(best_conf)
                else:
                    print(f"  Plate {i+1} Text: No text detected")
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = ', '.join(detected_texts) if detected_texts else 'No Text'
                cv2.putText(output_image, label, (x1, y1-10 if y1-10 > 10 else y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                found_plate = True
            # After processing all plates, save one JSON entry per image
            if all_plates:
                plate_text = ' '.join(all_plates)
                confidence = max(all_confs)
                status = "confirmed"
            else:
                plate_text = ""
                confidence = 0.0
                status = "to be reviewed"
            json_data = {
                "sequence": 1,
                "plate": plate_text,
                "frame": 1,
                "chunk_file": "unknown",
                "timestamp_seconds": timestamp_seconds,
                "timestamp_formatted": time_str,
                "confidence": float(confidence),
                "group_size": len(all_plates) if all_plates else 1,
                "status": status
            }
            current_date_results.append(json_data)
        if not found_plate:
            print("  No plates detected by YOLO.")
            # Save JSON result for no plate detected
            json_data = {
                "sequence": 1,
                "plate": "",
                "frame": 1,
                "chunk_file": "unknown",
                "timestamp_seconds": timestamp_seconds,
                "timestamp_formatted": time_str,
                "confidence": 0.0,
                "group_size": 1,
                "status": "to be reviewed"
            }
            current_date_results.append(json_data)
        # Save the output image
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, output_image)
        processed_images.add(image_path)

# At the end, flush any remaining results for the last date
if current_date is not None and current_date_results:
    out_json_path = os.path.join(output_folder, f"{current_date}.json")
    local_json_path = os.path.join(local_json_folder, f"{current_date}.json")
    with open(out_json_path, 'w') as outjf:
        json.dump(current_date_results, outjf, indent=2)
    with open(local_json_path, 'w') as localjf:
        json.dump(current_date_results, localjf, indent=2)
    s3_key = f"{DO_SPACES_FOLDER}/{current_date}.json"
    try:
        s3_client.upload_file(out_json_path, DO_SPACES_BUCKET, s3_key)
        print(f"[INFO] Uploaded {out_json_path} to s3://{DO_SPACES_BUCKET}/{s3_key}")
    except Exception as e:
        print(f"[ERROR] Failed to upload {out_json_path} to DigitalOcean Spaces: {e}")
