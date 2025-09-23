import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
import time
import sys

# Path to your folder and model
image_folder = "/home/ubantu/anpr/19thSept"  # Change this to your folder
model_path = "truck.pt"        # Path to your YOLO model
output_folder = "/home/ubantu/anpr/19thSept_Output"  # Folder to save annotated images
os.makedirs(output_folder, exist_ok=True)

# Load YOLO model
model = YOLO(model_path)

# Initialize PaddleOCR (English, CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif")
processed_images = set()

print(f"Watching folder: {image_folder} for new images...")
sys.stdout.flush()
while True:
    image_files = [f for f in glob.glob(os.path.join(image_folder, "*")) if f.lower().endswith(image_extensions)]
    new_images = [f for f in image_files if f not in processed_images]
    if not new_images:
        print("[INFO] Waiting for new images...")
        sys.stdout.flush()
        time.sleep(2)
        continue
    for image_path in sorted(new_images):
        image = cv2.imread(image_path)
        if image is None:
            print("Could not read image:", image_path)
            processed_images.add(image_path)
            continue
        results = model.predict(image, verbose=False)[0]
        output_image = image.copy()
        print(f"\nProcessing: {os.path.basename(image_path)}")
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
                        print(f"  Plate {i+1} Text: {text} | Confidence: {conf:.2f}")
                    # Print concatenated result in a single line (no separator)
                    concat_text = ''.join(detected_texts)
                    if concat_text:
                        print(f"  Plate {i+1} Single Line: {concat_text}")
                else:
                    print(f"  Plate {i+1} Text: No text detected")
                cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = ', '.join(detected_texts) if detected_texts else 'No Text'
                cv2.putText(output_image, label, (x1, y1-10 if y1-10 > 10 else y1+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                found_plate = True
        if not found_plate:
            print("  No plates detected by YOLO.")
        # Save the output image
        out_name = os.path.splitext(os.path.basename(image_path))[0] + "_output.jpg"
        out_path = os.path.join(output_folder, out_name)
        cv2.imwrite(out_path, output_image)
        processed_images.add(image_path)
    time.sleep(2)
