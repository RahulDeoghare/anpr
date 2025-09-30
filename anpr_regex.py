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

# List of common non-plate words that OCR might pick up
NON_PLATE_WORDS = [
    'CARRIER', 'GOODS', 'TRANSPORT', 'LOGISTICS', 'CARGO',
    'TRUCK', 'BUS', 'CAR', 'VEHICLE', 'MOTOR', 'AUTO',
    'INDIA', 'BHARAT', 'COMPANY', 'LTD', 'PVT', 'LIMITED'
]

def is_non_plate_word(text):
    """Check if the text is a common non-plate word"""
    return text.upper() in NON_PLATE_WORDS

def clean_plate_text_basic(text):
    """
    Basic cleaning: removes special characters but keeps the structure
    """
    # Convert to uppercase
    text = text.upper()
    
    # Remove leading/trailing special characters like -, ., *, ", etc.
    text = re.sub(r'^[^\w]+', '', text)  # Remove non-word chars from start
    text = re.sub(r'[^\w]+$', '', text)  # Remove non-word chars from end
    
    # Remove special characters except dots and hyphens (which may be part of format)
    text = re.sub(r'[^A-Z0-9.\-\s]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def clean_plate_text_strict(text):
    """
    Strict cleaning: keeps only alphanumeric characters
    """
    text = text.upper()
    # Remove all non-alphanumeric characters
    text = re.sub(r'[^A-Z0-9]', '', text)
    return text

def validate_indian_plate(text):
    """
    Validate if the text matches Indian license plate patterns
    Common patterns:
    - MH12AB1234 (State + District + Series + Number)
    - MH12A1234 (Older format)
    - 48AG455 (Some commercial vehicles)
    """
    # Remove dots and spaces for validation
    clean_text = re.sub(r'[.\s\-]', '', text)
    
    # Pattern 1: Standard Indian format (e.g., MH15FV9318)
    pattern1 = r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{3,4}$'
    
    # Pattern 2: Older format (e.g., MH15H3594)
    pattern2 = r'^[A-Z]{2}\d{1,2}[A-Z]\d{3,4}$'
    
    # Pattern 3: Commercial vehicle format (e.g., 48AG455)
    pattern3 = r'^\d{2}[A-Z]{1,2}\d{3,4}$'
    
    # Pattern 4: Format with just state and numbers (e.g., MH156563)
    pattern4 = r'^[A-Z]{2}\d{5,6}$'
    
    if (re.match(pattern1, clean_text) or 
        re.match(pattern2, clean_text) or 
        re.match(pattern3, clean_text) or
        re.match(pattern4, clean_text)):
        return True
    return False

def fix_common_ocr_errors(text):
    """
    Fix common OCR misreadings based on patterns observed
    """
    # Common substitutions based on your logs
    substitutions = {
        'WR': 'MH',  # WR often misread for MH
        'HH': 'MH',  # HH at start often means MH
        'KH': 'MH',  # KH at start often means MH
        'CH': 'MH',  # CH at start often means MH
        'SM': 'MH',  # SM at start often means MH
    }
    
    # Apply substitutions only at the beginning of the text
    for wrong, correct in substitutions.items():
        if text.startswith(wrong):
            text = correct + text[len(wrong):]
    
    return text

def process_plate_text(raw_texts):
    """
    Process multiple text detections from a single plate
    """
    # Clean each text
    cleaned_texts = []
    for text in raw_texts:
        # Skip if it's a known non-plate word
        if is_non_plate_word(text):
            continue
            
        cleaned = clean_plate_text_basic(text)
        if cleaned:
            cleaned_texts.append(cleaned)
    
    # Concatenate all cleaned texts
    combined = ''.join(cleaned_texts)
    
    # Apply strict cleaning
    strict_cleaned = clean_plate_text_strict(combined)
    
    # Fix common OCR errors
    fixed_text = fix_common_ocr_errors(strict_cleaned)
    
    # Validate the plate format
    is_valid = validate_indian_plate(fixed_text)
    
    return {
        'raw': ' '.join(raw_texts),
        'cleaned': combined,
        'strict': strict_cleaned,
        'fixed': fixed_text,
        'valid': is_valid
    }

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
            raw_texts = []
            confidences = []
            
            if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                for line in ocr_result[0]:
                    raw_text = line[1][0]
                    conf = line[1][1]
                    raw_texts.append(raw_text)
                    confidences.append(conf)
                
                # Process all detected texts
                processed = process_plate_text(raw_texts)
                
                # Log the results
                logger.info(f"  Plate {i+1} Raw: {processed['raw']}")
                logger.info(f"  Plate {i+1} Cleaned: {processed['cleaned']}")
                logger.info(f"  Plate {i+1} Strict: {processed['strict']}")
                logger.info(f"  Plate {i+1} Fixed: {processed['fixed']}")
                logger.info(f"  Plate {i+1} Valid Format: {processed['valid']}")
                logger.info(f"  Plate {i+1} Avg Confidence: {sum(confidences)/len(confidences):.2f}")
                
                # Use the fixed text for display if valid, otherwise use strict
                display_text = processed['fixed'] if processed['valid'] else processed['strict']
                
                # Color code the rectangle based on validity
                color = (0, 255, 0) if processed['valid'] else (0, 165, 255)  # Green if valid, orange if not
                
            else:
                logger.info(f"  Plate {i+1} Text: No text detected")
                display_text = 'No Text'
                color = (0, 0, 255)  # Red for no text
            
            # Draw rectangle on output image
            cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
            
            # Add text label
            cv2.putText(output_image, display_text, (x1, y1-10 if y1-10 > 10 else y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            found_plate = True
    
    if not found_plate:
        logger.info("  No plates detected by YOLO.")
    
    # Save the output image
    out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
    out_path = os.path.join(output_folder, out_name)
    cv2.imwrite(out_path, output_image)
    processed_images.add(image_path)

logger.info("[INFO] Finished processing all existing images. Now watching for new images...")

# Continuous monitoring loop
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

    # Sort new_images by filename
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
                raw_texts = []
                confidences = []
                
                if ocr_result and len(ocr_result) > 0 and ocr_result[0]:
                    for line in ocr_result[0]:
                        raw_text = line[1][0]
                        conf = line[1][1]
                        raw_texts.append(raw_text)
                        confidences.append(conf)
                    
                    # Process all detected texts
                    processed = process_plate_text(raw_texts)
                    
                    # Log the results
                    logger.info(f"  Plate {i+1} Raw: {processed['raw']}")
                    logger.info(f"  Plate {i+1} Cleaned: {processed['cleaned']}")
                    logger.info(f"  Plate {i+1} Strict: {processed['strict']}")
                    logger.info(f"  Plate {i+1} Fixed: {processed['fixed']}")
                    logger.info(f"  Plate {i+1} Valid Format: {processed['valid']}")
                    logger.info(f"  Plate {i+1} Avg Confidence: {sum(confidences)/len(confidences):.2f}")
                    
                    # Use the fixed text for display if valid, otherwise use strict
                    display_text = processed['fixed'] if processed['valid'] else processed['strict']
                    
                    # Color code the rectangle based on validity
                    color = (0, 255, 0) if processed['valid'] else (0, 165, 255)  # Green if valid, orange if not
                    
                else:
                    logger.info(f"  Plate {i+1} Text: No text detected")
                    display_text = 'No Text'
                    color = (0, 0, 255)  # Red for no text
                
                # Draw rectangle on output image
                cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)
                
                # Add text label
                cv2.putText(output_image, display_text, (x1, y1-10 if y1-10 > 10 else y1+20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                found_plate = True
        
        if not found_plate:
            logger.info("  No plates detected by YOLO.")
        
        # Save the output image
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, output_image)
        processed_images.add(image_path)

# At the end, no flushing needed