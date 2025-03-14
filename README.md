# Hackathon-Light-weighted-posture-guidance-system
## Problem Statement
Poor seated posture leads to chronic musculoskeletal disorders. This system provides real-time visual feedback to users to maintain ergonomic posture during extended sitting sessions.

## Solution Overview
- Uses **YOLOv8** for person detection
- Implements **MediaPipe Pose** for body landmark tracking
- Calculates posture angles using **OpenCV**
- Trained with **TensorFlow/Keras** on custom posture dataset
- Provides real-time feedback via webcam

## Key Technologies
```python
Python 3.9 | TensorFlow 2.15 | PyTorch 2.0 | OpenCV 4.8 | MediaPipe 0.10 | Ultralytics-YOLO 8.0
