import cv2
from paddleocr import PaddleOCR
import numpy as np

# Path to your image
image_path = "input/exit_uuid:53b3850d-e0ef-4668-9fb5-12c980aac83d_14_Sep_2025_22_26_50.jpg"  # Change this to your image file

# Initialize PaddleOCR (English, CPU)
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=False)

# Read image
image = cv2.imread(image_path)
if image is None:
    print("Could not read image:", image_path)
    exit(1)

# Run OCR
result = ocr.ocr(image)

# Draw results
for line in result[0]:
    box = line[0]
    text = line[1][0]
    conf = line[1][1]
    pts = [tuple(map(int, pt)) for pt in box]
    cv2.polylines(image, [np.array(pts)], isClosed=True, color=(0,255,0), thickness=2)
    cv2.putText(image, text, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

# Show image
cv2.imshow("OCR Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()