# How to Train YOLOv8 on Google Colab with Custom Road Damage Datasets

This guide explains how to fine-tune YOLOv8 on GPU accelerators in **Google Colab** using top open-source road infrastructure datasets (Roboflow Pothole Dataset, RDD2022, or Kaggle), and download the resulting `best.pt` weights to drop directly into your system.

---

## Step 1: Open Google Colab with GPU Accelerated Runtime
1. Go to [Google Colab](https://colab.research.google.com/).
2. Click **New Notebook**.
3. In the top menu, click **Runtime** > **Change runtime type** > Select **T4 GPU**.

---

## Step 2: Copy-Paste the Training Script into Colab

Run the following code cells in your Colab notebook:

### Cell 1: Install Dependencies
```python
# Install Ultralytics YOLOv8 and Roboflow dataset downloader
!pip install ultralytics roboflow
```

### Cell 2: Download High-Quality Road Damage Dataset
You can download a pre-labeled Pothole & Road Damage dataset directly via Roboflow API (Free):

```python
from roboflow import Roboflow

# Free public Roboflow Road Damage & Potholes dataset
rf = Roboflow(api_key="PUBLIC_ROBOFLOW_KEY") # You can get a free key at roboflow.com or use public download links
project = rf.workspace("vessels-uqued").project("pothole-detection-system")
version = project.version(1)
dataset = version.download("yolov8")

print("Dataset downloaded to:", dataset.location)
```

> **Alternative via Kaggle/Direct Download**:
> If using Kaggle's Pothole Dataset:
> ```python
> !pip install kaggle
> !kaggle datasets download -d keremberke/pothole-detection-bounding-box-dataset --unzip
> ```

### Cell 3: Train YOLOv8 Model on GPU
```python
from ultralytics import YOLO

# Load pre-trained YOLOv8 nano model
model = YOLO('yolov8n.pt')  # Or 'yolov8s.pt' / 'yolov8m.pt' for higher precision

# Start GPU fine-tuning
results = model.train(
    data=f"{dataset.location}/data.yaml",  # Path to dataset YAML
    epochs=50,                             # 30-50 epochs recommended for high precision
    imgsz=640,                             # High resolution 640x640
    batch=16,                              # Batch size
    name='road_damage_yolov8_model',
    device=0                               # Use GPU 0
)

print("Training finished! Model saved to: runs/detect/road_damage_yolov8_model/weights/best.pt")
```

### Cell 4: Evaluate Model Precision & Bounding Boxes
```python
# Validate trained weights on test set
metrics = model.val()
print("mAP50-95 Score:", metrics.box.map)
print("mAP50 Score:", metrics.box.map50)
```

### Cell 5: Download Trained `best.pt` Weights
```python
from google.colab import files

# Download trained PyTorch model weights to your PC
files.download('runs/detect/road_damage_yolov8_model/weights/best.pt')
```

---

## Step 3: Integrate Trained Model into Your Backend

Once `best.pt` downloads to your computer:
1. Rename `best.pt` to `yolov8n.pt` (or keep as `best.pt`).
2. Copy the file into your local project directory at:
   ```text
   traffic-safety-python/backend/app/models/yolov8n.pt
   ```
3. Restart your backend server. The AI detection pipeline in `detector.py` will automatically load your fine-tuned weights!
