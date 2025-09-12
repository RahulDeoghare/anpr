import cv2
import torch
import numpy as np
import boto3
from datetime import datetime
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deep_sort_realtime.deepsort_tracker import DeepSort
import os
import json
import time
import glob
from pathlib import Path
import sys

# Configure CUDA environment to handle cuDNN issues
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# PaddlePaddle specific environment variables to handle cuDNN version mismatch
os.environ['FLAGS_use_cudnn'] = 'true'  # Enable cuDNN but with compatibility layer
os.environ['FLAGS_conv_workspace_size_limit'] = '512'  # Limit memory usage
os.environ['FLAGS_cudnn_deterministic'] = 'false'  # Allow non-deterministic for compatibility
os.environ['FLAGS_enable_cublas'] = 'true'  # Keep cuBLAS for matrix operations

# Add cuDNN compatibility library path
venv_lib_path = os.path.join(os.path.dirname(__file__), 'venv_gpu', 'lib')
if 'LD_LIBRARY_PATH' in os.environ:
    os.environ['LD_LIBRARY_PATH'] = f"{venv_lib_path}:{os.environ['LD_LIBRARY_PATH']}"
else:
    os.environ['LD_LIBRARY_PATH'] = venv_lib_path

# DigitalOcean Spaces credentials
DO_SPACES_KEY = 'DO801UYGLUGLVCDQFYNM'
DO_SPACES_SECRET = 'fBDdr0Cp5NmbkSkD0jeRgE+oIaOZcOdSfzOautQGnL4'
DO_SPACES_REGION = 'blr1'
DO_SPACES_ENDPOINT = 'https://blr1.digitaloceanspaces.com'
DO_SPACES_BUCKET = 'vigilscreenshots'
DO_SPACES_FOLDER = 'testing'

# Global variable to store OCR logs
ocr_logs = []
ocr_session_start_time = datetime.now()
show_only_ocr = True  # Set to True to show only OCR-related messages

def log_ocr_message(message, log_type="INFO"):
    """
    Log OCR-related messages with timestamp
    """
    global ocr_logs
    timestamp = datetime.now()
    log_entry = {
        "timestamp": timestamp.isoformat(),
        "relative_time": (timestamp - ocr_session_start_time).total_seconds(),
        "type": log_type,
        "message": message
    }
    ocr_logs.append(log_entry)
    
    # Print to terminal with color coding
    if log_type == "OCR":
        print(f"\033[92m[OCR] {message}\033[0m")  # Green for OCR
    elif log_type == "OCR_ERROR":
        print(f"\033[91m[OCR ERROR] {message}\033[0m")  # Red for errors
    elif log_type == "OCR_SKIP":
        print(f"\033[93m[OCR SKIP] {message}\033[0m")  # Yellow for skips
    else:
        print(f"[{log_type}] {message}")

def print_message(message, force_print=False):
    """
    Print message only if show_only_ocr is False or force_print is True
    """
    global show_only_ocr
    if not show_only_ocr or force_print:
        print(message)

def save_ocr_logs_to_json():
    """
    Save OCR logs to JSON file
    """
    global ocr_logs
    
    if not ocr_logs:
        return
    
    # Create OCR logs directory
    ocr_logs_dir = "ocr_logs"
    os.makedirs(ocr_logs_dir, exist_ok=True)
    
    # Generate filename with timestamp
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    ocr_log_filename = f"ocr_terminal_logs_{now}.json"
    ocr_log_path = os.path.join(ocr_logs_dir, ocr_log_filename)
    
    # Prepare comprehensive log data
    log_data = {
        "session_info": {
            "start_time": ocr_session_start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_duration_seconds": (datetime.now() - ocr_session_start_time).total_seconds(),
            "total_ocr_messages": len(ocr_logs),
            "ocr_success_count": len([log for log in ocr_logs if log["type"] == "OCR"]),
            "ocr_error_count": len([log for log in ocr_logs if log["type"] == "OCR_ERROR"]),
            "ocr_skip_count": len([log for log in ocr_logs if log["type"] == "OCR_SKIP"])
        },
        "ocr_terminal_logs": ocr_logs
    }
    
    # Save to file
    with open(ocr_log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\n\033[96m📋 OCR Terminal Logs saved to: {ocr_log_path}\033[0m")
    
    # Also save in local_results directory
    local_save_dir = "local_results"
    os.makedirs(local_save_dir, exist_ok=True)
    local_ocr_log_path = os.path.join(local_save_dir, ocr_log_filename)
    with open(local_ocr_log_path, "w") as f:
        json.dump(log_data, f, indent=4)
    
    print(f"📋 OCR Terminal Logs also saved to: {local_ocr_log_path}")
    
    # Upload to DigitalOcean Spaces
    try:
        upload_json_to_spaces(ocr_log_path, ocr_log_filename)
        print(f"📋 OCR Terminal Logs uploaded to DigitalOcean Spaces")
    except Exception as e:
        print(f"[UPLOAD ERROR] Could not upload OCR logs to DigitalOcean Spaces: {e}")
    
    return ocr_log_path

def print_ocr_summary():
    """
    Print a summary of OCR processing
    """
    global ocr_logs
    
    if not ocr_logs:
        print("\n📋 No OCR processing logs found.")
        return
    
    print(f"\n\033[96m{'='*80}")
    print(f"📋 OCR PROCESSING TERMINAL SUMMARY")
    print(f"{'='*80}\033[0m")
    
    ocr_success = [log for log in ocr_logs if log["type"] == "OCR"]
    ocr_errors = [log for log in ocr_logs if log["type"] == "OCR_ERROR"]
    ocr_skips = [log for log in ocr_logs if log["type"] == "OCR_SKIP"]
    
    print(f"🕒 Session Duration: {(datetime.now() - ocr_session_start_time).total_seconds():.1f} seconds")
    print(f"✅ Successful OCR Operations: {len(ocr_success)}")
    print(f"❌ OCR Errors: {len(ocr_errors)}")
    print(f"⏭️  OCR Skips: {len(ocr_skips)}")
    print(f"📊 Total OCR Messages: {len(ocr_logs)}")
    
    if ocr_success:
        print(f"\n\033[92m🎯 SUCCESSFUL OCR DETECTIONS:\033[0m")
        for log in ocr_success:
            timestamp = datetime.fromisoformat(log["timestamp"])
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"   [{time_str}] {log['message']}")
    
    if ocr_errors:
        print(f"\n\033[91m❌ OCR ERRORS:\033[0m")
        for log in ocr_errors:
            timestamp = datetime.fromisoformat(log["timestamp"])
            time_str = timestamp.strftime("%H:%M:%S")
            print(f"   [{time_str}] {log['message']}")
    
    print(f"\033[96m{'='*80}\033[0m")

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

# Load YOLOv8 model (ANPR2.pt - specialized for license plate detection)
# Enable GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# Initialize CUDA properly 
if device == 'cuda':
    try:
        # Initialize CUDA context properly
        torch.cuda.init()
        torch.cuda.empty_cache()  # Clear any existing cache
        # Set memory allocation strategy
        torch.cuda.set_per_process_memory_fraction(0.7)  # Use 70% of GPU memory max
        print(f"[INFO] CUDA initialized successfully. GPU: {torch.cuda.get_device_name()}")
        print(f"[INFO] CUDA Memory: {torch.cuda.get_device_properties(0).total_memory // 1024**3} GB")
    except Exception as e:
        print(f"[WARNING] CUDA initialization issue: {e}")

model = YOLO("truck.pt")
# Try to move model to GPU with cuDNN error handling
if device == 'cuda':
    try:
        model.to(device)
        # Test the model with a dummy prediction to catch cuDNN issues early
        test_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict(test_frame, verbose=False, device=device, half=False, imgsz=640)
        print(f"[INFO] YOLO model successfully initialized on GPU")
    except Exception as e:
        if "cuDNN" in str(e):
            print(f"[WARNING] cuDNN error during YOLO model initialization: {e}")
            print(f"[INFO] Keeping YOLO model on GPU but will fallback to CPU during predictions if needed")
            model.to(device)  # Keep on GPU, handle errors during prediction
        else:
            print(f"[WARNING] GPU initialization failed for YOLO: {e}")
            print(f"[INFO] Falling back to CPU for YOLO model")
            device = 'cpu'
            model.to(device)
else:
    model.to(device)

# Global OCR instance and fallback management
ocr_instance = None
ocr_gpu_mode = False
ocr_gpu_failed = False

def initialize_paddleocr():
    """
    Initialize PaddleOCR with GPU mode first, fallback to CPU if needed
    """
    global ocr_instance, ocr_gpu_mode, ocr_gpu_failed
    
    print("[INFO] Initializing PaddleOCR...")
    
    # Try GPU mode first with cuDNN compatibility layer
    print("[INFO] Attempting to initialize PaddleOCR with GPU mode (cuDNN compatibility layer)...")
    try:
        # Initialize with GPU using cuDNN compatibility
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=True, show_log=False)
        
        # Test OCR with a simple image to ensure it works
        test_image = np.ones((50, 200, 3), dtype=np.uint8) * 255  # White test image
        test_result = ocr.ocr(test_image)
        
        print(f"[INFO] 🎉 PaddleOCR initialized successfully with GPU: True (cuDNN compatibility)")
        ocr_gpu_mode = True
        ocr_gpu_failed = False
        return ocr
    except Exception as e:
        print(f"[WARNING] GPU mode with cuDNN compatibility failed: {e}")
        print("[INFO] Falling back to CPU mode for stability...")
    
    # Fallback to CPU mode if GPU fails
    try:
        ocr = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=False, show_log=False)
        print(f"[INFO] PaddleOCR initialized successfully with GPU: False (CPU fallback)")
        ocr_gpu_mode = False
        ocr_gpu_failed = True
        return ocr
    except Exception as e:
        print(f"[WARNING] Failed to initialize PaddleOCR with CPU mode: {e}")
    
    # If standard configurations fail, try minimal configuration
    try:
        ocr = PaddleOCR(lang='en', use_gpu=False, show_log=False)
        print(f"[INFO] PaddleOCR initialized with minimal configuration (CPU mode)")
        ocr_gpu_mode = False
        ocr_gpu_failed = True
        return ocr
    except Exception as e:
        print(f"[ERROR] Failed to initialize PaddleOCR with minimal configuration: {e}")
        return None
        return None

def process_ocr_result_enhanced(ocr_result):
    """
    Enhanced OCR result processing with multi-line concatenation and Indian plate format matching
    """
    if not ocr_result or len(ocr_result) == 0:
        return None, 0
    
    result = ocr_result[0]
    all_texts = []
    all_confidences = []
    
    # Extract all text and confidence pairs
    if isinstance(result, list):
        for item in result:
            if len(item) == 2 and isinstance(item[1], tuple) and len(item[1]) == 2:
                text, confidence = item[1]
                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                if len(cleaned_text) >= 2:  # Minimum 2 characters
                    all_texts.append(cleaned_text)
                    all_confidences.append(confidence)
    elif isinstance(result, dict) and 'rec_texts' in result and 'rec_scores' in result:
        texts = result['rec_texts']
        scores = result['rec_scores']
        if texts and scores:
            for text, confidence in zip(texts, scores):
                cleaned_text = ''.join(c for c in text if c.isalnum()).upper()
                if len(cleaned_text) >= 2:  # Minimum 2 characters
                    all_texts.append(cleaned_text)
                    all_confidences.append(confidence)
    
    if not all_texts:
        return None, 0
    
    # Try different concatenation strategies
    candidates = []
    
    # Strategy 1: Direct concatenation of all texts (most common for Indian plates)
    if len(all_texts) > 1:
        merged_text = ''.join(all_texts)
        avg_confidence = sum(all_confidences) / len(all_confidences)
        candidates.append((merged_text, avg_confidence, "multi_line_concat", 100))  # High priority
    
    # Strategy 2: Individual texts (single line detection)
    for i, (text, conf) in enumerate(zip(all_texts, all_confidences)):
        candidates.append((text, conf, f"single_line_{i}", 50))  # Medium priority
    
    # Strategy 3: Smart Indian plate format reconstruction
    # Look for state codes and try to build proper format
    state_codes = ['AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CT', 'DN', 'DD', 'DL', 'GA', 'GJ', 'HR', 'HP', 'JH', 'JK',
                   'KA', 'KL', 'LD', 'MP', 'MH', 'MN', 'ML', 'MZ', 'NL', 'OR', 'PY', 'PB', 'RJ', 'SK', 'TN', 'TG',
                   'TR', 'UP', 'UT', 'WB', 'LA', 'PY']
    
    # Strategy 3a: Find state code and build from there
    if len(all_texts) >= 2:
        for i, text in enumerate(all_texts):
            # Check if this text starts with or is a state code
            for state_code in state_codes:
                if text.startswith(state_code) or text == state_code:
                    # Found state code, try to build complete plate
                    remaining_texts = [all_texts[j] for j in range(len(all_texts)) if j != i]
                    for remaining in remaining_texts:
                        # Try state code first
                        reconstructed = text + remaining
                        avg_conf = (all_confidences[i] + all_confidences[remaining_texts.index(remaining) if remaining in remaining_texts else 0]) / 2
                        candidates.append((reconstructed, avg_conf, f"smart_state_first_{state_code}", 120))  # Highest priority
                        
                        # Also try remaining first (less likely but possible)
                        reconstructed_rev = remaining + text
                        candidates.append((reconstructed_rev, avg_conf, f"smart_state_second_{state_code}", 80))
    
    # Strategy 4: Try all two-line combinations with priority for likely formats
    if len(all_texts) >= 2:
        for i in range(len(all_texts)):
            for j in range(i + 1, len(all_texts)):
                combined = all_texts[i] + all_texts[j]
                avg_conf = (all_confidences[i] + all_confidences[j]) / 2
                
                # Higher priority if first text looks like state code + digits
                priority = 70
                if len(all_texts[i]) >= 3 and all_texts[i][:2] in state_codes:
                    priority = 110
                
                candidates.append((combined, avg_conf, f"two_line_{i}_{j}", priority))
                
                # Reverse order with lower priority
                combined_rev = all_texts[j] + all_texts[i]
                rev_priority = 60
                if len(all_texts[j]) >= 3 and all_texts[j][:2] in state_codes:
                    rev_priority = 90
                    
                candidates.append((combined_rev, avg_conf, f"two_line_rev_{j}_{i}", rev_priority))
    
    # Evaluate all candidates
    best_text = None
    best_confidence = 0
    best_score = 0
    best_method = None
    
    for text, confidence, method, base_priority in candidates:
        # Skip if too short
        if len(text) < 8:
            continue
            
        # Bonus for valid Indian plate format (major bonus)
        format_bonus = 0
        if is_valid_indian_plate(text):
            format_bonus = 0.25  # 25% bonus for valid format
        
        # Bonus for proper length (10 characters is ideal)
        length_bonus = 0
        if len(text) == 10:
            length_bonus = 0.08  # 8% bonus for perfect length
        elif 9 <= len(text) <= 11:
            length_bonus = 0.05  # 5% bonus for good length
        
        # Penalty for length issues
        length_penalty = 0
        if len(text) < 9:
            length_penalty = 0.15  # 15% penalty for too short
        elif len(text) > 11:
            length_penalty = 0.10  # 10% penalty for too long
        
        # Priority bonus (convert base priority to confidence bonus)
        priority_bonus = base_priority / 1000.0  # Convert 100 -> 0.1, etc.
        
        # Calculate final score
        adjusted_confidence = confidence + format_bonus + length_bonus + priority_bonus - length_penalty
        
        # Combine with base priority for final ranking
        final_score = adjusted_confidence * 1000 + base_priority
        
        if final_score > best_score:
            best_text = text
            best_confidence = confidence  # Keep original confidence for logging
            best_score = final_score
            best_method = method
    
    return best_text, best_confidence

def safe_ocr_operation(image):
    """
    Perform OCR operation with error handling (CPU mode)
    """
    global ocr_instance, ocr_gpu_mode, ocr_gpu_failed
    
    try:
        # Try OCR operation
        result = ocr_instance.ocr(image)
        return result, None
    
    except Exception as e:
        error_msg = str(e)
        print(f"[WARNING] OCR operation failed: {error_msg[:200]}...")
        
        # Try to reinitialize OCR if it failed
        try:
            print("[INFO] Attempting to reinitialize PaddleOCR...")
            ocr_instance = PaddleOCR(use_textline_orientation=True, lang='en', use_gpu=False, show_log=False)
            print("[INFO] Successfully reinitialized PaddleOCR, retrying OCR...")
            
            # Retry the OCR operation
            result = ocr_instance.ocr(image)
            return result, None
            
        except Exception as fallback_error:
            return None, f"OCR reinitialize failed: {fallback_error}"

try:
    ocr_instance = initialize_paddleocr()
    if ocr_instance is None:
        print(f"[ERROR] Could not initialize PaddleOCR with any configuration")
        exit(1)
except Exception as e:
    print(f"[ERROR] Failed to initialize PaddleOCR: {e}")
    exit(1)

# Initialize DeepSORT with GPU support
print("[INFO] Initializing DeepSort tracker...")
try:
    # Try GPU mode first for DeepSort
    tracker = DeepSort(max_age=30, embedder="mobilenet", embedder_gpu=True, embedder_model_name="osnet_x0_25")
    print("[INFO] DeepSort initialized successfully with GPU embedder")
except Exception as e:
    print(f"[WARNING] Failed to initialize DeepSort with GPU embedder: {e}")
    print("[INFO] Trying DeepSort with CPU embedder as fallback...")
    try:
        tracker = DeepSort(max_age=30, embedder="mobilenet", embedder_gpu=False)
        print("[INFO] DeepSort initialized with CPU embedder")
    except Exception as e2:
        print(f"[ERROR] Failed to initialize DeepSort: {e2}")
        exit(1)

# Create debug folder for plate images
debug_folder = "debug_plates"
os.makedirs(debug_folder, exist_ok=True)


# Video chunks directory
CHUNKS_DIR = "/home/ubantu/recordings/192.168.1.104/"

# RTSP URL for testing (fallback)
RTSP_URL = "rtsp://admin:newkt$0123@192.168.1.104:554/cam/realmonitor?channel=1&subtype=0"

# List of videos to process in order (leave empty to use chunks from folder)
videos = []  # If you want to use video files, add them here

def get_chunk_files(chunks_dir, start_from_chunk=0):
    """
    Get all chunk files in the directory, sorted by chunk number, starting from a specific chunk
    """
    if not os.path.exists(chunks_dir):
        print(f"[ERROR] Chunks directory does not exist: {chunks_dir}")
        return []
    
    # Find all chunk files with pattern chunk_XXX.mp4
    chunk_pattern = os.path.join(chunks_dir, "chunk_*.mp4")
    chunk_files = glob.glob(chunk_pattern)
    
    # Sort by chunk number
    def extract_chunk_number(filename):
        basename = os.path.basename(filename)
        # Extract number from chunk_XXX.mp4
        try:
            number_str = basename.split('_')[1].split('.')[0]
            return int(number_str)
        except:
            return 0
    
    chunk_files.sort(key=extract_chunk_number)
    
    # Filter to start from the specified chunk number
    filtered_chunks = []
    for chunk_file in chunk_files:
        chunk_number = extract_chunk_number(chunk_file)
        if chunk_number >= start_from_chunk:
            filtered_chunks.append(chunk_file)
    
    print(f"[INFO] Found {len(filtered_chunks)} chunks starting from chunk_{start_from_chunk:03d}.mp4")
    return filtered_chunks

def monitor_and_process_chunks(chunks_dir, check_interval=5):
    """
    Monitor the chunks directory and process new chunks as they appear
    """
    processed_chunks = set()
    all_results = {}
    chunk_counter = 0
    
    # Real-time finalized plate saving
    local_save_dir = "local_results"
    os.makedirs(local_save_dir, exist_ok=True)
    json_filename = f"chunks_realtime.json"
    local_json_path = os.path.join(local_save_dir, json_filename)
    finalized_plates = {}
    
    print(f"[INFO] Starting chunk monitoring in: {chunks_dir}")
    print(f"[INFO] Checking for new chunks every {check_interval} seconds...")
    print(f"\n\033[95m{'='*70}")
    print(f"🔍 CHUNK MONITORING STARTED")
    print(f"📁 Directory: {chunks_dir}")
    print(f"⏰ Check interval: {check_interval} seconds")
    print(f"🚀 Starting from: chunk_000.mp4")
    print(f"{'='*70}\033[0m\n")
    
    while True:
        try:
            # Get current chunk files starting from chunk 0
            current_chunks = get_chunk_files(chunks_dir, start_from_chunk=0)
            
            # Find new chunks to process
            new_chunks = [chunk for chunk in current_chunks if chunk not in processed_chunks]
            
            if new_chunks:
                print(f"[INFO] Found {len(new_chunks)} new chunks to process")
                
                for chunk_path in new_chunks:
                    chunk_name = os.path.basename(chunk_path)
                    print(f"\n\033[94m{'='*60}")
                    print(f"🎬 NOW PROCESSING: {chunk_name}")
                    print(f"{'='*60}\033[0m")
                    print_message(f"[INFO] Processing chunk: {chunk_name}", force_print=True)
                    
                    # Process the chunk
                    chunk_results = process_video_chunk(chunk_path, chunk_counter, all_results)
                    
                    # Merge results
                    all_results.update(chunk_results)
                    
                    # Update finalized plates with valid Indian plates
                    for track_id, info in chunk_results.items():
                        if (info.get("best_plate_text") and 
                            is_valid_indian_plate(info["best_plate_text"]) and
                            info.get("best_confidence", 0) > 0.6):
                            
                            finalized_plates[f"chunk_{chunk_counter}_{track_id}"] = {
                                "plate_text": info["best_plate_text"],
                                "formatted_plate_text": info.get("formatted_plate_text", format_indian_plate(info["best_plate_text"])),
                                "frame_detected": info["frame_detected"],
                                "chunk_file": chunk_name,
                                "bbox": info["bbox"],
                                "confidence": info["best_confidence"]
                            }
                    
                    # Save real-time results
                    with open(local_json_path, "w") as f:
                        json.dump(finalized_plates, f, indent=4)
                    
                    processed_chunks.add(chunk_path)
                    chunk_counter += 1
                    
                    print(f"\n\033[92m✅ COMPLETED: {chunk_name}")
                    print(f"📊 Chunk Results: {len([k for k, v in chunk_results.items() if v.get('best_plate_text')])} valid plates detected")
                    print(f"🔢 Total finalized plates so far: {len(finalized_plates)}\033[0m")
                    print_message(f"[INFO] Completed processing {chunk_name}. Total finalized plates: {len(finalized_plates)}", force_print=True)
                    
                    # Print current OCR stats
                    ocr_success_count = len([log for log in ocr_logs if log["type"] == "OCR"])
                    ocr_error_count = len([log for log in ocr_logs if log["type"] == "OCR_ERROR"]) 
                    print(f"\033[96m[OCR STATS] Session: {ocr_success_count} successful, {ocr_error_count} errors, {len(ocr_logs)} total OCR operations\033[0m")
            
            else:
                print(f"\033[90m⏳ No new chunks found. Waiting {check_interval} seconds... (Processed: {len(processed_chunks)} chunks)\033[0m")
                print_message(f"[INFO] No new chunks found. Waiting {check_interval} seconds...")
            
            # Wait before checking again
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n[INFO] Stopping chunk monitoring...")
            # Save OCR logs before stopping
            print_ocr_summary()
            save_ocr_logs_to_json()
            break
        except Exception as e:
            print(f"[ERROR] Error in chunk monitoring: {e}")
            time.sleep(check_interval)
    
    return all_results, finalized_plates

def process_video_chunk(video_path, chunk_id, global_results):
    """
    Process a single video chunk
    """
    chunk_name = os.path.basename(video_path)
    print(f"\n\033[93m🎯 STARTING PROCESSING: {chunk_name}")
    print(f"📁 Chunk ID: {chunk_id}")
    print(f"📄 Full path: {video_path}\033[0m")
    
    # Store results for this chunk
    chunk_results = {}
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {video_path}")
        return chunk_results
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 50 == 0:
            print(f"\033[96m📹 {chunk_name}: Frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)\033[0m")
            print_message(f"[PROGRESS] Chunk {chunk_id}: Processing frame {frame_count}/{total_frames}...")

        # YOLO model prediction with cuDNN error handling
        try:
            results = model.predict(frame, verbose=False, device=device, half=False, imgsz=640)[0]
        except RuntimeError as e:
            if "cuDNN" in str(e):
                print_message(f"[WARNING] cuDNN error encountered, falling back to CPU for this frame: {e}")
                try:
                    # Move model to CPU temporarily for this prediction
                    model.to('cpu')
                    results = model.predict(frame, verbose=False, device='cpu', half=False, imgsz=640)[0]
                    # Try to move back to GPU for next frame
                    if device == 'cuda':
                        model.to(device)
                except Exception as cpu_e:
                    print_message(f"[ERROR] Even CPU prediction failed: {cpu_e}")
                    continue
            else:
                print_message(f"[ERROR] YOLO prediction failed: {e}")
                continue
        except Exception as e:
            print_message(f"[ERROR] Unexpected error in YOLO prediction: {e}")
            continue
            
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = f"chunk_{chunk_id}_{track.track_id}"
            l, t, w, h = track.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)
            if x2 - x1 < 30 or y2 - y1 < 15:
                continue
            expansion = 20
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(frame.shape[1], x2 + expansion)
            y2_expanded = min(frame.shape[0], y2 + expansion)

            if track_id not in chunk_results:
                chunk_results[track_id] = {
                    "all_detections": [],
                    "best_plate_text": "",
                    "formatted_plate_text": "",
                    "best_confidence": 0,
                    "frame_detected": frame_count,
                    "bbox": [x1_expanded, y1_expanded, x2_expanded, y2_expanded],
                    "method": "enhanced_processing",
                    "last_ocr_frame": 0,
                    "ocr_attempts": 0,
                    "max_ocr_attempts": 3,  # Reduced from 5 to 3
                    "chunk_file": os.path.basename(video_path),
                    "last_plate_detected": "",  # Track last detected plate
                    "stable_plate_count": 0,    # Count consecutive same detections
                    "is_finalized": False       # Mark when plate is confirmed
                }

            # Enhanced OCR triggering logic
            should_ocr = (
                not chunk_results[track_id]["is_finalized"] and  # Don't OCR if already finalized
                chunk_results[track_id]["ocr_attempts"] < chunk_results[track_id]["max_ocr_attempts"] and
                len(chunk_results[track_id]["all_detections"]) < 2 and  # Reduced from 3 to 2
                (frame_count - chunk_results[track_id]["last_ocr_frame"]) > 15 and  # Increased from 10 to 15 frames
                (chunk_results[track_id]["best_confidence"] < 0.90 or not is_valid_indian_plate(chunk_results[track_id]["best_plate_text"]))  # Increased threshold
            )

            if should_ocr:
                chunk_results[track_id]["last_ocr_frame"] = frame_count
                chunk_results[track_id]["ocr_attempts"] += 1
                plate_img = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                
                # Save original plate image for debugging (colored)
                debug_path_original = os.path.join(debug_folder, f"plate_original_{track_id}_{frame_count}.jpg")
                cv2.imwrite(debug_path_original, plate_img)
                
                if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                    # Scale up the image for better OCR
                    scale_factor = max(3.0, 100.0 / min(plate_img.shape[:2]))
                    new_width = int(plate_img.shape[1] * scale_factor)
                    new_height = int(plate_img.shape[0] * scale_factor)
                    plate_resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Save resized colored image
                    debug_path_resized = os.path.join(debug_folder, f"plate_resized_{track_id}_{frame_count}.jpg")
                    cv2.imwrite(debug_path_resized, plate_resized)
                    
                    # Create enhanced version for OCR processing
                    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
                    
                    # For OCR, use the enhanced grayscale image directly (PaddleOCR can handle grayscale)
                    processed_img = sharpened
                    
                    # Save the processed grayscale image for comparison
                    debug_path_processed = os.path.join(debug_folder, f"plate_processed_{track_id}_{frame_count}.jpg")
                    cv2.imwrite(debug_path_processed, processed_img)
                else:
                    processed_img = plate_img

                try:
                    ocr_result, error_msg = safe_ocr_operation(processed_img)
                    if error_msg:
                        log_ocr_message(f"ID {track_id}: {error_msg}", "OCR_ERROR")
                        continue
                        
                    # Use enhanced multi-line OCR processing
                    best_text, best_confidence = process_ocr_result_enhanced(ocr_result)
                    
                    if best_text and best_text.strip():
                        current_plate = best_text.strip()
                        
                        # Check for stable plate detection
                        if current_plate == chunk_results[track_id]["last_plate_detected"]:
                            chunk_results[track_id]["stable_plate_count"] += 1
                            if chunk_results[track_id]["stable_plate_count"] >= 2:  # 2 consecutive same detections
                                chunk_results[track_id]["is_finalized"] = True
                                log_ocr_message(f"ID {track_id}: Plate finalized after {chunk_results[track_id]['stable_plate_count']} stable detections: {current_plate}", "OCR_FINALIZED")
                        else:
                            chunk_results[track_id]["stable_plate_count"] = 1
                            chunk_results[track_id]["last_plate_detected"] = current_plate
                        
                        # Process the detection
                        if is_valid_indian_plate(best_text):
                            formatted_plate = format_indian_plate(best_text)
                            log_ocr_message(f"ID {track_id}: '{formatted_plate}' (conf: {best_confidence:.3f}, stable: {chunk_results[track_id]['stable_plate_count']}) - VALID INDIAN PLATE", "OCR")
                            current_detection = {
                                "text": best_text,
                                "formatted_text": formatted_plate,
                                "confidence": best_confidence,
                                "frame": frame_count
                            }
                            chunk_results[track_id]["all_detections"].append(current_detection)
                            if best_confidence > chunk_results[track_id]["best_confidence"]:
                                chunk_results[track_id]["best_plate_text"] = best_text
                                chunk_results[track_id]["formatted_plate_text"] = formatted_plate
                                chunk_results[track_id]["best_confidence"] = best_confidence
                                chunk_results[track_id]["frame_detected"] = frame_count
                                chunk_results[track_id]["bbox"] = [x1_expanded, y1_expanded, x2_expanded, y2_expanded]
                            
                            # Early termination for high confidence valid plates
                            if (best_confidence > 0.92 and chunk_results[track_id]["stable_plate_count"] >= 1):
                                chunk_results[track_id]["is_finalized"] = True
                                log_ocr_message(f"ID {track_id}: High confidence plate detected, finalizing: {best_text} (conf: {best_confidence:.3f})", "OCR_FINALIZED")
                        else:
                            log_ocr_message(f"ID {track_id}: '{current_plate}' (conf: {best_confidence:.3f}, stable: {chunk_results[track_id]['stable_plate_count']}) - INVALID FORMAT", "OCR")
                    else:
                        log_ocr_message(f"ID {track_id}: No text detected in OCR result", "OCR_EMPTY")
                except Exception as e:
                    log_ocr_message(f"ID {track_id}: {e}", "OCR_ERROR")
            else:
                if chunk_results[track_id]["best_plate_text"] and is_valid_indian_plate(chunk_results[track_id]["best_plate_text"]):
                    formatted_text = chunk_results[track_id].get("formatted_plate_text", format_indian_plate(chunk_results[track_id]["best_plate_text"]))
                    log_ocr_message(f"Skipping OCR for ID {track_id} - already have valid Indian plate: {formatted_text}", "OCR_SKIP")

    cap.release()
    cv2.destroyAllWindows()
    
    valid_plates_in_chunk = len([k for k, v in chunk_results.items() if v.get('best_plate_text') and is_valid_indian_plate(v['best_plate_text'])])
    print(f"\n\033[92m✅ CHUNK PROCESSING COMPLETE: {os.path.basename(video_path)}")
    print(f"📊 Valid plates found in this chunk: {valid_plates_in_chunk}")
    print(f"🎬 Total frames processed: {frame_count}\033[0m")
    
    return chunk_results

def process_video(video_path):
    # Store results
    results_dict = {}

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = f"output_{os.path.basename(video_path)}"
    out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_count = 0

    # Real-time finalized plate saving
    local_save_dir = "local_results"
    os.makedirs(local_save_dir, exist_ok=True)
    json_filename = f"rtsp_realtime.json"
    local_json_path = os.path.join(local_save_dir, json_filename)
    finalized_plates = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 100 == 0:
            print(f"[PROGRESS] Processing frame {frame_count} of {video_path}...")

        # YOLO model prediction with cuDNN error handling
        try:
            results = model.predict(frame, verbose=False, device=device, half=False, imgsz=640)[0]
        except RuntimeError as e:
            if "cuDNN" in str(e):
                print_message(f"[WARNING] cuDNN error encountered, falling back to CPU for this frame: {e}")
                try:
                    # Move model to CPU temporarily for this prediction
                    model.to('cpu')
                    results = model.predict(frame, verbose=False, device='cpu', half=False, imgsz=640)[0]
                    # Try to move back to GPU for next frame
                    if device == 'cuda':
                        model.to(device)
                except Exception as cpu_e:
                    print_message(f"[ERROR] Even CPU prediction failed: {cpu_e}")
                    continue
            else:
                print_message(f"[ERROR] YOLO prediction failed: {e}")
                continue
        except Exception as e:
            print_message(f"[ERROR] Unexpected error in YOLO prediction: {e}")
            continue
        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 0))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, w, h = track.to_ltwh()
            x1, y1, x2, y2 = int(l), int(t), int(l + w), int(t + h)
            if x2 - x1 < 30 or y2 - y1 < 15:
                continue
            expansion = 20
            x1_expanded = max(0, x1 - expansion)
            y1_expanded = max(0, y1 - expansion)
            x2_expanded = min(frame.shape[1], x2 + expansion)
            y2_expanded = min(frame.shape[0], y2 + expansion)

            if track_id not in results_dict:
                results_dict[track_id] = {
                    "all_detections": [],
                    "best_plate_text": "",
                    "formatted_plate_text": "",
                    "best_confidence": 0,
                    "frame_detected": frame_count,
                    "bbox": [x1_expanded, y1_expanded, x2_expanded, y2_expanded],
                    "method": "enhanced_processing",
                    "last_ocr_frame": 0,
                    "ocr_attempts": 0,
                    "max_ocr_attempts": 5
                }

            should_ocr = (
                results_dict[track_id]["ocr_attempts"] < results_dict[track_id]["max_ocr_attempts"] and
                len(results_dict[track_id]["all_detections"]) < 3 and
                (frame_count - results_dict[track_id]["last_ocr_frame"]) > 10 and
                (results_dict[track_id]["best_confidence"] < 0.85 or not is_valid_indian_plate(results_dict[track_id]["best_plate_text"]))
            )

            if should_ocr:
                results_dict[track_id]["last_ocr_frame"] = frame_count
                results_dict[track_id]["ocr_attempts"] += 1
                plate_img = frame[y1_expanded:y2_expanded, x1_expanded:x2_expanded]
                
                # Save original plate image for debugging (colored)
                debug_path_original = os.path.join(debug_folder, f"plate_original_{track_id}_{frame_count}.jpg")
                cv2.imwrite(debug_path_original, plate_img)
                
                if plate_img.shape[0] > 0 and plate_img.shape[1] > 0:
                    # Scale up the image for better OCR
                    scale_factor = max(3.0, 100.0 / min(plate_img.shape[:2]))
                    new_width = int(plate_img.shape[1] * scale_factor)
                    new_height = int(plate_img.shape[0] * scale_factor)
                    plate_resized = cv2.resize(plate_img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
                    
                    # Save resized colored image
                    debug_path_resized = os.path.join(debug_folder, f"plate_resized_{track_id}_{frame_count}.jpg")
                    cv2.imwrite(debug_path_resized, plate_resized)
                    
                    # Create enhanced version for OCR processing
                    gray = cv2.cvtColor(plate_resized, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                    enhanced = clahe.apply(gray)
                    kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
                    sharpened = cv2.filter2D(enhanced, -1, kernel_sharp)
                    
                    # For OCR, use the enhanced grayscale image directly (PaddleOCR can handle grayscale)
                    processed_img = sharpened
                    
                    # Save the processed grayscale image for comparison
                    debug_path_processed = os.path.join(debug_folder, f"plate_processed_{track_id}_{frame_count}.jpg")
                    cv2.imwrite(debug_path_processed, processed_img)
                else:
                    processed_img = plate_img

                try:
                    ocr_result, error_msg = safe_ocr_operation(processed_img)
                    if error_msg:
                        log_ocr_message(f"ID {track_id}: {error_msg}", "OCR_ERROR")
                        continue
                        
                    # Use enhanced multi-line OCR processing
                    best_text, best_confidence = process_ocr_result_enhanced(ocr_result)
                    
                    if best_text and is_valid_indian_plate(best_text):
                        formatted_plate = format_indian_plate(best_text)
                        log_ocr_message(f"ID {track_id}: '{formatted_plate}' (conf: {best_confidence:.3f}) - FINALIZED INDIAN PLATE", "OCR")
                        current_detection = {
                            "text": best_text,
                            "formatted_text": formatted_plate,
                            "confidence": best_confidence,
                            "frame": frame_count
                        }
                        results_dict[track_id]["all_detections"].append(current_detection)
                        if best_confidence > results_dict[track_id]["best_confidence"]:
                            results_dict[track_id]["best_plate_text"] = best_text
                            results_dict[track_id]["formatted_plate_text"] = formatted_plate
                            results_dict[track_id]["best_confidence"] = best_confidence
                            results_dict[track_id]["frame_detected"] = frame_count
                            results_dict[track_id]["bbox"] = [x1_expanded, y1_expanded, x2_expanded, y2_expanded]
                            finalized_plates[track_id] = {
                                "plate_text": best_text,
                                "formatted_plate_text": formatted_plate,
                                "frame_detected": frame_count,
                                "bbox": [x1_expanded, y1_expanded, x2_expanded, y2_expanded],
                                "confidence": best_confidence
                            }
                            # Real-time write to JSON (only finalized plates)
                            with open(local_json_path, "w") as f:
                                json.dump(finalized_plates, f, indent=4)
                        # No else: do not write if nothing detected
                except Exception as e:
                    log_ocr_message(f"ID {track_id}: {e}", "OCR_ERROR")
            else:
                if results_dict[track_id]["best_plate_text"] and is_valid_indian_plate(results_dict[track_id]["best_plate_text"]):
                    formatted_text = results_dict[track_id].get("formatted_plate_text", format_indian_plate(results_dict[track_id]["best_plate_text"]))
                    log_ocr_message(f"Skipping OCR for ID {track_id} - already have valid Indian plate: {formatted_text}", "OCR_SKIP")

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return finalized_plates

def calculate_plate_similarity(plate1, plate2):
    """Enhanced similarity calculation for license plates"""
    # Remove spaces and convert to uppercase
    clean1 = plate1.replace(" ", "").upper()
    clean2 = plate2.replace(" ", "").upper()
    
    # If plates are identical, return 100% similarity
    if clean1 == clean2:
        return 1.0
    
    # Handle partial matches and OCR errors
    min_len = min(len(clean1), len(clean2))
    max_len = max(len(clean1), len(clean2))
    
    if max_len == 0:
        return 0.0
    
    # Check if one plate is a substring of another (common OCR issue)
    if clean1 in clean2 or clean2 in clean1:
        return 0.9  # High similarity for substring matches
    
    # Character-by-character similarity
    matches = 0
    for i in range(min_len):
        if clean1[i] == clean2[i]:
            matches += 1
        # Allow for common OCR confusions
        elif (clean1[i], clean2[i]) in [('0', 'O'), ('O', '0'), ('1', 'I'), ('I', '1'), ('5', 'S'), ('S', '5')]:
            matches += 0.8
    
    # Penalize length differences
    length_penalty = min_len / max_len
    
    # Calculate similarity score with OCR error tolerance
    similarity = (matches / max_len) * length_penalty
    
    return similarity

def group_similar_plates(results_dict, similarity_threshold=0.75, min_confidence=0.6, min_group_size=1):
    """Enhanced grouping with confidence filtering and better similarity matching"""
    
    # First, get only valid Indian plates with minimum confidence
    valid_results = {}
    for tid, info in results_dict.items():
        if (info.get("best_plate_text") and 
            is_valid_indian_plate(info["best_plate_text"]) and 
            info.get("best_confidence", 0) >= min_confidence):
            valid_results[tid] = info
    
    print(f"[INFO] Filtered to {len(valid_results)} high-confidence detections (min conf: {min_confidence})")
    
    if not valid_results:
        return {}
    
    # Group similar plates with enhanced similarity
    groups = []
    processed_tids = set()
    
    for tid1, info1 in valid_results.items():
        if tid1 in processed_tids:
            continue
            
        current_group = [(tid1, info1)]
        processed_tids.add(tid1)
        
        for tid2, info2 in valid_results.items():
            if tid2 in processed_tids:
                continue
                
            # Calculate similarity between plates
            similarity = calculate_plate_similarity(info1["best_plate_text"], info2["best_plate_text"])
            
            if similarity >= similarity_threshold:
                current_group.append((tid2, info2))
                processed_tids.add(tid2)
        
        # Only keep groups with minimum size (to filter out single false positives)
        if len(current_group) >= min_group_size:
            groups.append(current_group)
    
    print(f"[INFO] Found {len(groups)} groups with minimum {min_group_size} detections each")
    
    # Select best detection from each group
    final_results = {}
    
    for group_idx, group in enumerate(groups):
        # Find the best detection in this group
        best_tid = None
        best_info = None
        best_score = -1
        
        for tid, info in group:
            # Enhanced composite scoring
            confidence_score = info["best_confidence"]
            attempt_bonus = min(len(info["all_detections"]) * 0.05, 0.15)  # Reduced bonus
            group_consistency_bonus = min(len(group) * 0.02, 0.1)  # Bonus for larger groups
            
            # Composite scoring with group consistency
            composite_score = confidence_score + attempt_bonus + group_consistency_bonus
            
            if composite_score > best_score:
                best_score = composite_score
                best_tid = tid
                best_info = info
        
        if best_tid and best_info:
            # Use a new unique ID for the grouped result
            unique_id = f"Vehicle_{group_idx + 1}"
            
            status = "confirmed" if best_info["best_confidence"] >= 0.8 else "needs_review"
            
            final_results[unique_id] = {
                "plate_text": best_info["best_plate_text"],
                "formatted_plate_text": best_info.get("formatted_plate_text", format_indian_plate(best_info["best_plate_text"])),
                "frame_detected": best_info["frame_detected"],
                "bbox": best_info["bbox"],
                "confidence": best_info["best_confidence"],
                "method": best_info.get("method", "enhanced_processing"),
                "total_attempts": len(best_info["all_detections"]),
                "group_size": len(group),
                "original_ids": [tid for tid, _ in group],
                "composite_score": best_score,
                "avg_group_confidence": sum(info["best_confidence"] for _, info in group) / len(group),
                "status": status
            }
    
    return final_results

def upload_json_to_spaces(local_json_path, remote_filename):
    remote_path = f"{DO_SPACES_FOLDER}/{remote_filename}"
    session = boto3.session.Session()
    client = session.client('s3',
        region_name=DO_SPACES_REGION,
        endpoint_url=DO_SPACES_ENDPOINT,
        aws_access_key_id=DO_SPACES_KEY,
        aws_secret_access_key=DO_SPACES_SECRET
    )
    client.upload_file(local_json_path, DO_SPACES_BUCKET, remote_path)
    print(f"[UPLOAD] Uploaded {local_json_path} to {remote_path} in DigitalOcean Spaces.")


# Main logic: Process chunks from directory
if not videos:
    print(f"\n\033[96m{'='*80}")
    print(f"🔍 OCR TERMINAL OUTPUT MODE")
    print(f"{'='*80}\033[0m")
    print(f"📋 Only OCR-related messages will be displayed in the terminal")
    print(f"✅ OCR Success: \033[92mGreen\033[0m")
    print(f"❌ OCR Errors: \033[91mRed\033[0m") 
    print(f"⏭️  OCR Skips: \033[93mYellow\033[0m")
    print(f"📊 OCR Stats: \033[96mCyan\033[0m")
    print(f"💾 All OCR logs will be saved to JSON files")
    print(f"\033[96m{'='*80}\033[0m\n")
    
    print_message(f"[INFO] Starting processing for video chunks in: {CHUNKS_DIR}", force_print=True)
    
    # Check if chunks directory exists
    if not os.path.exists(CHUNKS_DIR):
        print(f"[ERROR] Chunks directory does not exist: {CHUNKS_DIR}")
        print(f"[INFO] Falling back to RTSP stream: {RTSP_URL}")
        results_dict = process_video(RTSP_URL)
        
        # Process RTSP results and save OCR logs
        print_ocr_summary()
        save_ocr_logs_to_json()
        
    else:
        # Monitor and process chunks continuously
        all_results, finalized_plates = monitor_and_process_chunks(CHUNKS_DIR, check_interval=5)
        
        print("\n[INFO] Grouping similar license plates with enhanced algorithm...")
        final_results = group_similar_plates(
            all_results,
            similarity_threshold=0.75,
            min_confidence=0.6,
            min_group_size=1
        )

        # Sort by frame_detected (timestamp order) instead of confidence
        sorted_results = dict(sorted(final_results.items(),
                                  key=lambda x: x[1]["frame_detected"]))

        # Save enhanced results with date/time in filename (local and DigitalOcean)
        now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        json_filename = f"chunks_{now}.json"
        local_save_dir = "local_results"
        os.makedirs(local_save_dir, exist_ok=True)
        local_json_path = os.path.join(local_save_dir, json_filename)
        with open(json_filename, "w") as f:
            json.dump(sorted_results, f, indent=4)
        with open(local_json_path, "w") as f:
            json.dump(sorted_results, f, indent=4)

        print(f"\n💾 Results saved to '{json_filename}' (current dir) and '{local_json_path}' (local_results dir)")

        # Upload to DigitalOcean Spaces
        try:
            upload_json_to_spaces(json_filename, json_filename)
            upload_json_to_spaces(local_json_path, json_filename)  # Also upload local copy
        except Exception as e:
            print(f"[UPLOAD ERROR] Could not upload to DigitalOcean Spaces: {e}")

        print("\n" + "="*80)
        print("🚗 VEHICLES FROM VIDEO CHUNKS IN CHRONOLOGICAL ORDER")
        print("="*80)

        for idx, (vehicle_id, info) in enumerate(sorted_results.items(), 1):
            # Calculate timestamp from frame number (assuming 30 FPS)
            fps = 30  # Adjust this based on your video's actual FPS
            timestamp_seconds = info['frame_detected'] / fps
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            chunk_info = info.get('chunk_file', 'unknown')
            print(f"\033[92m✓ Vehicle {idx} (chronological order):\033[0m")
            print(f"   📋 Plate: \033[1m{info['formatted_plate_text']}\033[0m")
            print(f"   🎯 Confidence: {info['confidence']:.3f}")
            print(f"   📊 Avg Group Confidence: {info['avg_group_confidence']:.3f}")
            print(f"   🔍 Total Attempts: {info['total_attempts']}")
            print(f"   👥 Group Size: {info['group_size']} detections")
            print(f"   📈 Composite Score: {info['composite_score']:.3f}")
            print(f"   🎬 Frame: {info['frame_detected']}")
            print(f"   📁 Chunk: {chunk_info}")
            print(f"   ⏰ Timestamp: {minutes:02d}:{seconds:02d}")
            print(f"   🔗 Original IDs: {info['original_ids']}")
            print(f"   📝 Status: {info['status']}")
            print()

        # Enhanced summary
        print(f"\033[94m{'='*80}")
        print(f"📊 ENHANCED FINAL SUMMARY (From Video Chunks):")
        print(f"   🚗 High-confidence unique vehicles: {len(sorted_results)}")
        print(f"   📋 Total valid detections: {len([tid for tid, info in all_results.items() if info['best_plate_text'] and is_valid_indian_plate(info['best_plate_text'])])}")
        print(f"   🎯 High-confidence detections: {len([tid for tid, info in all_results.items() if info['best_plate_text'] and is_valid_indian_plate(info['best_plate_text']) and info['best_confidence'] >= 0.8])}")
        print(f"   📈 Grouping efficiency: {len(sorted_results)} vehicles from {len([tid for tid, info in all_results.items() if info['best_plate_text'] and is_valid_indian_plate(info['best_plate_text'])])} total detections")
        print(f"   🎖️ Average confidence: {sum(info['confidence'] for info in sorted_results.values()) / len(sorted_results):.3f}" if sorted_results else "   🎖️ Average confidence: 0.000")
        print(f"   📊 Average group size: {sum(info['group_size'] for info in sorted_results.values()) / len(sorted_results):.1f}" if sorted_results else "   📊 Average group size: 0.0")
        print(f"{'='*80}\033[0m")

        # Show vehicles in order of appearance
        print(f"\n\033[93m🕒 VEHICLES IN ORDER OF APPEARANCE (From Chunks):\033[0m")
        for idx, (vehicle_id, info) in enumerate(sorted_results.items(), 1):
            fps = 30  # Adjust based on your video's FPS
            timestamp_seconds = info['frame_detected'] / fps
            minutes = int(timestamp_seconds // 60)
            seconds = int(timestamp_seconds % 60)
            chunk_info = info.get('chunk_file', 'unknown')
            print(f"   {idx}. {info['formatted_plate_text']} - appeared at {minutes:02d}:{seconds:02d} (frame {info['frame_detected']}) in {chunk_info} - Status: {info['status']}")

        # Optional: Create a detailed timeline JSON
        timeline_data = []
        for idx, (vehicle_id, info) in enumerate(sorted_results.items(), 1):
            fps = 30
            timestamp_seconds = info['frame_detected'] / fps
            timeline_data.append({
                "sequence": idx,
                "plate": info['formatted_plate_text'],
                "frame": info['frame_detected'],
                "chunk_file": info.get('chunk_file', 'unknown'),
                "timestamp_seconds": timestamp_seconds,
                "timestamp_formatted": f"{int(timestamp_seconds // 60):02d}:{int(timestamp_seconds % 60):02d}",
                "confidence": info['confidence'],
                "group_size": info['group_size'],
                "status": info['status']
            })

        timeline_filename = f"chunks_timeline_{now}.json"
        local_timeline_path = os.path.join(local_save_dir, timeline_filename)
        with open(timeline_filename, "w") as f:
            json.dump(timeline_data, f, indent=4)
        with open(local_timeline_path, "w") as f:
            json.dump(timeline_data, f, indent=4)

        print(f"\n\033[96m💾 Timeline data saved to '{timeline_filename}' (current dir) and '{local_timeline_path}' (local_results dir)\033[0m")

        # Upload timeline to DigitalOcean Spaces
        try:
            upload_json_to_spaces(timeline_filename, timeline_filename)
            upload_json_to_spaces(local_timeline_path, timeline_filename)  # Also upload local copy
        except Exception as e:
            print(f"[UPLOAD ERROR] Could not upload timeline to DigitalOcean Spaces: {e}")

        print("\n[INFO] Video chunks processing completed.")
        
        # Print OCR summary and save logs
        print_ocr_summary()
        save_ocr_logs_to_json()
        
        # Continue monitoring for new chunks
        print("\n[INFO] Continuing to monitor for new chunks...")
        print("[INFO] Press Ctrl+C to stop monitoring...")
else:
    pass  # No video files provided; RTSP stream is used. Add video file logic here if needed.