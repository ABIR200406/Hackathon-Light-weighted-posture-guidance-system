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
```
## Setup & Installation

### Step 1: Install Dependencies
Ensure you have Python and TensorFlow installed. Open your terminal and run the following commands:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Step 2: Dataset Preparation
Place your training data in the specified directory structure:
```
C:\Users\KIIT\Minor\train
```
Ensure your data is organized into subfolders corresponding to the class labels.

### Step 3: Running the Script
Save the provided code in a file, e.g., `posture_model.py`, and run the script:

```bash
python posture_model.py
```
This will train the model and save the output as `posture_model_v2.keras`.

---

## Usage Instructions

### Step 1: Load the Model
To use the saved model for inference, open a Python script or Jupyter Notebook and load the model:

```python
import tensorflow as tf
model = tf.keras.models.load_model('posture_model_v2.keras')
```

### Step 2: Preprocess Input Image
Ensure the image is resized to `(224, 224)` and normalized:

```python
from tensorflow.keras.preprocessing import image
import numpy as np

img = image.load_img('path_to_image.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0
```

### Step 3: Make Predictions

```python
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
print("Predicted Class:", class_index)
```

### Step 4: Evaluate Model

You can evaluate the model on a validation set or new data:

```python
model.evaluate(validation_dataset)
```

## Step 5: Visualize Results
Check the confusion matrix and classification report by running the original script to generate the plots and report.

### Step 6: Troubleshooting
- Ensure the dataset is correctly structured.
- Verify that TensorFlow and other dependencies are installed.
- Adjust the batch size or learning rate if you encounter performance issues.



https://github.com/user-attachments/assets/e5c8df97-43db-44ad-928b-fb5cced8c9aa



