import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import sys
import re
from datetime import datetime
import logging

# Configure logging to save to a file and print to console
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
fh = logging.FileHandler('anpr_logs.txt')
fh.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

# Console handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)



# Path to your folder and model
image_folder = "/home/ubantu/vms/data/anpr_ss/"    # Folder to monitor
model_path = "truck.pt"        # Path to your YOLO model
output_folder = "/home/ubantu/anpr/output"  # Folder to save annotated images
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Initialize PaddleOCR (English, CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
processed_images = set()

logger.info(f"Watching folder: {image_folder} for new images...")

# Process all existing images first in systematic order
logger.info("[INFO] Processing all existing images in folder...")
# Define allowed prefixes
allowed_prefixes = ("anpr_uuid_53b3850d-e0ef-4668-9fb5-12c980aac83d",)

# Collect all images from the folder
all_image_files = [f for f in glob.glob(os.path.join(image_folder, "*"))
                   if f.lower().endswith(image_extensions)
                   and os.path.basename(f).startswith(allowed_prefixes)]

# Sort all existing images by filename
all_image_files.sort(key=lambda x: os.path.basename(x))

for image_path in all_image_files:
    if image_path in processed_images:
        continue
        image = cv2.imread(image_path)
        if image is None:
            logger.info("Could not read image: %s", image_path)
            processed_images.add(image_path)
            continue
        results = model.predict(image, verbose=False)[0]
        output_image = image.copy()
        logger.info(f"\nProcessing: {os.path.basename(image_path)}")
        found_plate = False

        if results.boxes and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                if plate_img.shape[0] < 5 or plate_img.shape[1] < 5:
                    continue
                ocr_result = ocr.ocr(plate_img)
                detected_texts = []
                if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        detected_texts.append(text)
                        logger.info(f"  Plate {i+1} Text: {text} | Confidence: {conf:.2f}")
                    # Print concatenated result in a single line (no separator)
                    concat_text = ''.join(detected_texts)
                    if concat_text:
                        logger.info(f"  Plate {i+1} Single Line: {concat_text}")
                else:
                    logger.info(f"  Plate {i+1} Text: No text detected")
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = ', '.join(detected_texts) if detected_texts else 'No Text'
                cv2.putText(output_image, label, (x1, y1-10 if y1-10 > 10 else y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                found_plate = True
        if not found_plate:
            logger.info("  No plates detected by YOLO.")
        # Save the output image
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, output_image)
        processed_images.add(image_path)

logger.info("[INFO] Finished processing all existing images. Now watching for new images...")

while True:

    # Define allowed prefixes
    allowed_prefixes = ("anpr_uuid_53b3850d-e0ef-4668-9fb5-12c980aac83d",)

    # Collect images from the folder
    image_files = [f for f in glob.glob(os.path.join(image_folder, "*"))
                   if f.lower().endswith(image_extensions)
                   and os.path.basename(f).startswith(allowed_prefixes)]

    # Filter new images
    new_images = [f for f in image_files if f not in processed_images]
    if not new_images:
        logger.info("[INFO] Waiting for new images...")
        time.sleep(2)
        continue

    # Sort new_images by filename (which includes timestamp, so alphabetical sort will be chronological)
    new_images.sort(key=lambda x: os.path.basename(x))

    for image_path in new_images:
        image = cv2.imread(image_path)
        if image is None:
            logger.info("Could not read image: %s", image_path)
            processed_images.add(image_path)
            continue
        results = model.predict(image, verbose=False)[0]
        output_image = image.copy()
        logger.info(f"\nProcessing: {os.path.basename(image_path)}")
        found_plate = False

        if results.boxes and len(results.boxes) > 0:
            for i, box in enumerate(results.boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plate_img = image[y1:y2, x1:x2]
                if plate_img.shape[0] < 5 or plate_img.shape[1] < 5:
                    continue
                ocr_result = ocr.ocr(plate_img)
                detected_texts = []
                if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                    for line in ocr_result[0]:
                        text = line[1][0]
                        conf = line[1][1]
                        detected_texts.append(text)
                        logger.info(f"  Plate {i+1} Text: {text} | Confidence: {conf:.2f}")
                    # Print concatenated result in a single line (no separator)
                    concat_text = ''.join(detected_texts)
                    if concat_text:
                        logger.info(f"  Plate {i+1} Single Line: {concat_text}")
                else:
                    logger.info(f"  Plate {i+1} Text: No text detected")
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = ', '.join(detected_texts) if detected_texts else 'No Text'
                cv2.putText(output_image, label, (x1, y1-10 if y1-10 > 10 else y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                found_plate = True
        if not found_plate:
            logger.info("  No plates detected by YOLO.")
        # Save the output image
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, output_image)
        processed_images.add(image_path)

# At the end, no flushing needed
