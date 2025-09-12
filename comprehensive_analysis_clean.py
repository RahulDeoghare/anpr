import cv2
import torch
import numpy as np
from datetime import datetime
import os
import re
import json
import glob
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Configuration
INPUT_FOLDER = "/home/ubantu/vms/data/screenshots"
MODEL_PATH = "truck.pt"
OUTPUT_FOLDER = "json_results"

def extract_datetime_from_filename(filename):
    """
    Extract datetime from filename pattern like:
    exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d_10_Sep_2025_22_26_50
    Returns: 10_Sep_2025_22_26_50
    """
    # Pattern to match the datetime part after the UUID
    pattern = r'exit_uuid:[a-f0-9-]+_(\d{1,2}_[A-Za-z]{3}_\d{4}_\d{2}_\d{2}_\d{2})'
    match = re.search(pattern, filename)
    if match:
        return match.group(1)
    return None

def get_matching_images(folder_path):
    """
    Get all images that start with 'exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d'
    """
    if not os.path.exists(folder_path):
        log_message(f"Input folder not found: {folder_path}", "ERROR")
        return []
    
    # Pattern to match files starting with the specific UUID
    pattern = os.path.join(folder_path, "exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d*")
    matching_files = glob.glob(pattern)
    
    # Filter for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    image_files = []
    
    for file_path in matching_files:
        if any(file_path.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file_path)
    
    return sorted(image_files)

def log_message(message, log_type="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if log_type == "OCR":
        print(f"\033[92m[{timestamp}] [OCR] {message}\033[0m")  # Green
    elif log_type == "ERROR":
        print(f"\033[91m[{timestamp}] [ERROR] {message}\033[0m")  # Red
    elif log_type == "SUCCESS":
        print(f"\033[96m[{timestamp}] [SUCCESS] {message}\033[0m")  # Cyan
    else:
        print(f"[{timestamp}] [{log_type}] {message}")

def is_valid_indian_plate(plate_text):
    """
    Enhanced validation for Indian number plates with better error tolerance.
    """
    if not plate_text or len(plate_text) < 8:
        return False
    
    # Remove spaces and convert to uppercase
    clean_plate = plate_text.replace(" ", "").upper()
    
    # Allow slight length variations (9-11 characters for flexibility)
    if len(clean_plate) < 9 or len(clean_plate) > 11:
        return False
    
    # Enhanced patterns with more flexibility
    # Pattern 1: Standard format - 2 letters + 2 digits + 2 letters + 4 digits
    pattern1 = r'^[A-Z]{2}\d{2}[A-Z]{1,2}\d{3,4}$'
    
    # Pattern 2: BH Series format - 2 digits + BH + 4 digits + 2 letters
    pattern2 = r'^\d{2}BH\d{4}[A-Z]{2}$'
    
    # Pattern 3: New format variations
    pattern3 = r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$'  # Exact 10 char
    
    if re.match(pattern1, clean_plate) or re.match(pattern2, clean_plate) or re.match(pattern3, clean_plate):
        # Extended state codes list
        state_codes = [
            'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK',
            'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG',
            'TR', 'UP', 'UT', 'WB', 'LA', 'PY'
        ]
        
        # Check first two characters for state code or BH series
        if clean_plate.startswith(tuple(state_codes)) or 'BH' in clean_plate[:4]:
            return True
    
    return False

def format_indian_plate(plate_text):
    """
    Format the plate text in standard Indian format with spaces.
    """
    clean_plate = plate_text.replace(" ", "").upper()
    
    if len(clean_plate) == 10:
        if clean_plate[2:4] == 'BH':
            # BH series: 22 BH 1234 AB
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:8]} {clean_plate[8:]}"
        else:
            # Standard: MH 15 FV 8808
            return f"{clean_plate[:2]} {clean_plate[2:4]} {clean_plate[4:6]} {clean_plate[6:]}"
    
    return plate_text

def process_plate_image(plate_img):
    """
    Process the extracted plate image for better OCR results
    """
    if plate_img.shape[0] == 0 or plate_img.shape[1] == 0:
        return None
    
    # Scale up for better OCR
    scale_factor = max(3.0, 100.0 / min(plate_img.shape[:2]))
    new_width = int(plate_img.shape[1] * scale_factor)
    new_height = int(plate_img.shape[0] * scale_factor)
    plate_resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # Convert to grayscale and enhance
    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Simple sharpening
    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
    processed_img = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
    
    return processed_img

def extract_plate_text_with_ocr(ocr, processed_img):
    """
    Extract and process text from plate image using advanced OCR logic
    """
    try:
        ocr_result = ocr.ocr(processed_img)
        if not ocr_result or len(ocr_result) == 0:
            return None, []
        
        result = ocr_result[0]  # Get first result
        
        best_text = ""
        best_confidence = 0
        all_detected_texts = []
        all_confidences = []
        
        # Handle different PaddleOCR output formats
        if isinstance(result, list):
            # Legacy format: list of [bbox, (text, confidence)]
            for item in result:
                if len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
                    text, confidence = item[1]
                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                    if cleaned_text and len(cleaned_text) >= 2:  # Accept even small fragments
                        all_detected_texts.append(cleaned_text)
                        all_confidences.append(confidence)
                    
                    # Check individual text for complete plate
                    if confidence > best_confidence and len(cleaned_text) >= 8 and is_valid_indian_plate(cleaned_text):
                        best_text = cleaned_text
                        best_confidence = confidence
        
        elif isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
            # New dictionary format
            texts = result['rec_texts']
            scores = result['rec_scores']
            
            if texts and scores:
                # Collect all text fragments
                for text, confidence in zip(texts, scores):
                    cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                    if cleaned_text and len(cleaned_text) >= 2:  # Accept even small fragments
                        all_detected_texts.append(cleaned_text)
                        all_confidences.append(confidence)
                    
                    # Check individual text for complete plate
                    if confidence > best_confidence and len(cleaned_text) >= 8 and is_valid_indian_plate(cleaned_text):
                        best_text = cleaned_text
                        best_confidence = confidence
        
        # If no complete valid plate found, try combining fragments
        if not best_text and all_detected_texts:
            # Try different combinations of detected text fragments
            combinations_to_try = []
            
            # Simple concatenation of all fragments in detection order
            merged_all = ''.join(all_detected_texts)
            if len(merged_all) >= 8 and len(merged_all) <= 12:
                avg_conf = sum(all_confidences) / len(all_confidences)
                combinations_to_try.append((merged_all, avg_conf, "sequential_merge"))
            
            # Try pairs of fragments in both orders
            for i in range(len(all_detected_texts)):
                for j in range(len(all_detected_texts)):
                    if i != j:  # Allow any pair combination
                        combo = all_detected_texts[i] + all_detected_texts[j]
                        if 8 <= len(combo) <= 12:  # Valid Indian plate length range
                            avg_conf = (all_confidences[i] + all_confidences[j]) / 2
                            combinations_to_try.append((combo, avg_conf, f"pair_{i}+{j}"))
            
            # Test each combination for valid Indian plate format
            for combo_text, combo_conf, method in combinations_to_try:
                if is_valid_indian_plate(combo_text):
                    best_text = combo_text
                    best_confidence = combo_conf
                    break
        
        # Return the best plate text found and all fragments for debugging
        if best_text and is_valid_indian_plate(best_text):
            return {
                "text": best_text,
                "formatted_text": format_indian_plate(best_text),
                "confidence": best_confidence,
                "method": "fragment_combination" if len(all_detected_texts) > 1 else "direct_ocr",
                "all_fragments": all_detected_texts
            }, all_detected_texts
        else:
            return None, all_detected_texts
            
    except Exception as e:
        print(f"   ❌ OCR processing failed: {e}")
        return None, []

def process_single_image(image_path, model, ocr, output_folder, sequence_number):
    """Process a single image and return results in the specified format"""
    filename = os.path.basename(image_path)
    datetime_str = extract_datetime_from_filename(filename)
    
    if not datetime_str:
        log_message(f"Could not extract datetime from filename: {filename}", "ERROR")
        return None
    
    log_message(f"Processing: {filename}")
    log_message(f"Extracted datetime: {datetime_str}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        log_message(f"Could not read image: {image_path}", "ERROR")
        return None
    
    # Create debug folder for this image
    debug_folder = os.path.join(output_folder, f"debug_{datetime_str}")
    os.makedirs(debug_folder, exist_ok=True)
    
    # Save original image to debug folder
    cv2.imwrite(os.path.join(debug_folder, "00_original_image.jpg"), image)
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Test different confidence thresholds
    confidence_thresholds = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    
    best_results = None
    best_threshold = 0
    
    log_message(f"Testing YOLO detection with different confidence thresholds...")
    
    for conf_thresh in confidence_thresholds:
        try:
            # Run YOLO detection
            results = model.predict(
                image, 
                verbose=False, 
                device=device, 
                half=False, 
                imgsz=640,
                conf=conf_thresh
            )[0]
            
            if results.boxes and len(results.boxes) > 0:
                # Keep track of best results (most detections)
                if best_results is None or len(results.boxes) > len(best_results.boxes):
                    best_results = results
                    best_threshold = conf_thresh
                    
                # Create annotated image
                annotated_image = image.copy()
                
                for i, box in enumerate(results.boxes):
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Draw rectangle and confidence
                    cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_image, f"{conf:.3f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save annotated image
                output_path = os.path.join(debug_folder, f"detections_conf_{conf_thresh:.2f}.jpg")
                cv2.imwrite(output_path, annotated_image)
                    
        except Exception as e:
            log_message(f"Error with confidence {conf_thresh}: {e}", "ERROR")
    
    # Process OCR on best detections
    best_plate = None
    best_confidence = 0
    
    if best_results and best_results.boxes:
        log_message(f"Found {len(best_results.boxes)} detections with confidence {best_threshold}")
        
        for i, box in enumerate(best_results.boxes):
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Extract region with padding
            expansion = 20
            x1_pad = max(0, x1 - expansion)
            y1_pad = max(0, y1 - expansion)
            x2_pad = min(image.shape[1], x2 + expansion)
            y2_pad = min(image.shape[0], y2 + expansion)
            
            extracted_region = image[y1_pad:y2_pad, x1_pad:x2_pad]
            
            if extracted_region.shape[0] < 5 or extracted_region.shape[1] < 5:
                continue
            
            # Save extracted region
            region_path = os.path.join(debug_folder, f"extracted_region_{i+1}.jpg")
            cv2.imwrite(region_path, extracted_region)
            
            # Process the plate image
            processed_img = process_plate_image(extracted_region)
            
            if processed_img is not None:
                # Save processed version
                processed_path = os.path.join(debug_folder, f"processed_region_{i+1}.jpg")
                cv2.imwrite(processed_path, processed_img)
                
                # Extract text using OCR
                plate_result, all_fragments = extract_plate_text_with_ocr(ocr, processed_img)
                
                if plate_result and plate_result['confidence'] > best_confidence:
                    log_message(f"Valid Indian plate detected: {plate_result['formatted_text']}", "SUCCESS")
                    
                    best_plate = plate_result['text'].replace(" ", "")  # Remove spaces for clean format
                    best_confidence = plate_result['confidence']
                    
                    # Create success image
                    final_annotated = extracted_region.copy()
                    cv2.putText(final_annotated, plate_result['formatted_text'], (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    cv2.putText(final_annotated, f"Conf: {plate_result['confidence']:.3f}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    success_path = os.path.join(debug_folder, f"SUCCESS_plate_{i+1}_{plate_result['text']}.jpg")
                    cv2.imwrite(success_path, final_annotated)
    
    # Prepare result data in the requested format
    if best_plate:
        result_data = {
            "sequence": sequence_number,
            "plate": best_plate,
            "frame": sequence_number,  # Using sequence as frame number
            "chunk_file": "unknown",
            "timestamp_seconds": sequence_number * 0.5333333333333333,  # Approximate 30fps timing
            "timestamp_formatted": f"{int((sequence_number * 0.5333333333333333) // 60):02d}:{int((sequence_number * 0.5333333333333333) % 60):02d}",
            "confidence": best_confidence,
            "group_size": 1,
            "status": "confirmed"
        }
        log_message(f"Plate detected: {best_plate} (confidence: {best_confidence:.3f})", "SUCCESS")
    else:
        result_data = {
            "sequence": sequence_number,
            "plate": "",
            "frame": sequence_number,
            "chunk_file": "unknown",
            "timestamp_seconds": sequence_number * 0.5333333333333333,
            "timestamp_formatted": f"{int((sequence_number * 0.5333333333333333) // 60):02d}:{int((sequence_number * 0.5333333333333333) % 60):02d}",
            "confidence": 0,
            "group_size": 1,
            "status": "please review this once"
        }
        log_message("No valid plate detected", "ERROR")
    
    return result_data

def main():
    """Main function to process all matching images and save results as JSON files"""
    print("="*90)
    print("🔍 BATCH PROCESSING: YOLO + OCR ANALYSIS FOR EXIT UUID IMAGES")
    print("="*90)
    
    # Create output folder
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Get all matching images
    log_message(f"Searching for images in: {INPUT_FOLDER}")
    image_files = get_matching_images(INPUT_FOLDER)
    
    if not image_files:
        log_message("No matching images found!", "ERROR")
        log_message(f"Looking for files starting with: exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d", "INFO")
        return
    
    log_message(f"Found {len(image_files)} matching image(s)", "SUCCESS")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        log_message(f"Model file not found: {MODEL_PATH}", "ERROR")
        return
    
    # Initialize device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_message(f"Using device: {device}", "SUCCESS")
    
    # Load YOLO model
    log_message(f"Loading YOLO model: {MODEL_PATH}")
    try:
        model = YOLO(MODEL_PATH)
        model.to(device)
        log_message("YOLO model loaded successfully", "SUCCESS")
    except Exception as e:
        log_message(f"Failed to load YOLO model: {e}", "ERROR")
        return
    
    # Initialize OCR
    log_message("Initializing PaddleOCR...")
    try:
        # Force CPU usage to avoid CUDNN issues
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=False, show_log=False)
        log_message("PaddleOCR initialized successfully with GPU: False (forced CPU to avoid CUDNN issues)", "SUCCESS")
    except Exception as e:
        log_message(f"Failed to initialize OCR: {e}", "ERROR")
        return
    
    # Process each image
    total_plates_found = 0
    successful_files = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"\n{'='*90}")
        print(f"📸 PROCESSING IMAGE {i}/{len(image_files)}")
        print(f"{'='*90}")
        
        filename = os.path.basename(image_path)
        datetime_str = extract_datetime_from_filename(filename)
        
        if not datetime_str:
            log_message(f"Skipping {filename} - could not extract datetime", "ERROR")
            continue
        
        # Process the image
        try:
            result_data = process_single_image(image_path, model, ocr, OUTPUT_FOLDER, i)
        except Exception as e:
            log_message(f"Error processing {filename}: {e}", "ERROR")
            continue
        
        if result_data is None:
            log_message(f"Failed to process {filename}", "ERROR")
            continue
        
        # Save results as JSON
        json_filename = f"{datetime_str}.json"
        json_path = os.path.join(OUTPUT_FOLDER, json_filename)
        
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            log_message(f"Results saved to: {json_filename}", "SUCCESS")
            successful_files += 1
            
            if result_data['plate']:
                total_plates_found += 1
            
            # Print summary for this image
            print(f"\n📊 SUMMARY FOR {filename}:")
            print(f"   📅 Datetime: {datetime_str}")
            print(f"   🏷️  Plate: {result_data['plate'] if result_data['plate'] else 'None detected'}")
            print(f"   🎯 Confidence: {result_data['confidence']:.3f}")
            print(f"   ✅ Status: {result_data['status']}")
            
        except Exception as e:
            log_message(f"Failed to save JSON for {filename}: {e}", "ERROR")
    
    # Final summary
    print(f"\n{'='*90}")
    print("📊 BATCH PROCESSING COMPLETE")
    print(f"{'='*90}")
    print(f"📁 Input folder: {INPUT_FOLDER}")
    print(f"📁 Output folder: {OUTPUT_FOLDER}")
    print(f"📸 Total images processed: {successful_files}/{len(image_files)}")
    print(f"🏷️  Total plates detected: {total_plates_found}")
    print(f"📄 JSON files created: {successful_files}")
    
    if successful_files > 0:
        print(f"\n✅ SUCCESS: Results saved as JSON files in '{OUTPUT_FOLDER}/' folder")
        print(f"💡 Each JSON file contains:")
        print(f"   • sequence: Image sequence number")
        print(f"   • plate: Detected license plate (empty if none)")
        print(f"   • frame: Frame number")
        print(f"   • confidence: OCR confidence score")
        print(f"   • status: 'confirmed' or 'please review this once'")
    else:
        print(f"\n❌ No images were successfully processed")
    
    print(f"\n🔧 Debug images and intermediate results saved in debug folders")

if __name__ == "__main__":
    main()
