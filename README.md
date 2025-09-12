# ANPR (Automatic Number Plate Recognition) System

A comprehensive ANPR system using YOLO for vehicle detection and PaddleOCR for license plate text recognition.

## Features

- Vehicle detection using custom YOLO model
- Indian license plate recognition
- Batch processing of images
- JSON output for results
- DigitalOcean Spaces integration for cloud storage
- Comprehensive logging and debugging

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- Dependencies listed in requirements.txt

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd anpr
```

2. Create and activate virtual environment:
```bash
python -m venv venv_gpu
source venv_gpu/bin/activate  # Linux/Mac
# or
venv_gpu\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Processing

Run the comprehensive analysis script:
```bash
python comprehensive_analysis.py
```

### Configuration

Edit the configuration variables in `comprehensive_analysis.py`:
- `INPUT_FOLDER`: Path to input images
- `MODEL_PATH`: Path to YOLO model file
- `OUTPUT_FOLDER`: Path for JSON results
- DigitalOcean Spaces credentials

## File Structure

- `comprehensive_analysis.py` - Main processing script
- `run.py` - Alternative execution script
- `truck.pt` - YOLO model file (not included in repo)
- `yolov8n.pt` - Base YOLO model (not included in repo)
- `json_results/` - Output JSON files
- `debug_plates/` - Debug images and intermediate results

## Output Format

The system generates JSON files with the following structure:
```json
[
  {
    "sequence": 1,
    "plate": "MH 15 FV 8808",
    "frame": 1,
    "chunk_file": "unknown",
    "timestamp_seconds": 0.5333333333333333,
    "timestamp_formatted": "22:26:50",
    "confidence": 0.95,
    "group_size": 1,
    "status": "confirmed"
  }
]
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
