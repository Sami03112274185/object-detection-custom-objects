# Object Detection and Localization of Custom Objects in Videos

**Authors:** Sami Ghaffar, Abdul Raheem, Wasi Hussain.  
**Institution:** Bahria University, Karachi Campus (B.S. Computer Science)  
**Course:** Final Year Project (ESC 498 / ESC 499)

## Overview
This repository contains the implementation and documentation for my Bachelor's thesis on detecting and localizing custom-trained objects in continuous video streams. The pipeline extracts video frames, runs deep learning object detection inference, and draws real-time bounding boxes with class labels and confidence scores.

## Key Features
* **Custom Dataset Preparation:** Annotated custom object classes with bounding-box coordinates for supervised training.
* **Video Frame Processing:** Real-time frame extraction and inference using OpenCV and deep learning object detection models.
* **Localization & Tracking:** Generates bounding-box coordinates `(x, y, w, h)` and confidence thresholds across consecutive frames.

## Tech Stack
* **Language:** Python 3.x
* **Computer Vision & Deep Learning:** OpenCV (`cv2`), NumPy, TensorFlow / PyTorch / YOLO

## Repository Structure
* `detect_video.py` — Main inference script for running object detection on video files or webcam feeds.
* `requirements.txt` — Python dependencies required to run the pipeline.

## Usage
```bash
pip install -r requirements.txt
python detect_video.py --video sample_input.mp4 --confidence 0.5
```
