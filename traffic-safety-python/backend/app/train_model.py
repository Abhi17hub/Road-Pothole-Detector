"""
Automated YOLOv8 Model Training / Fine-tuning Script.
Allows training YOLOv8 on custom Pothole and Road Damage datasets (e.g. Roboflow / Kaggle YOLO format).
"""
import os
import sys
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_model")

def train_yolo(
    data_yaml_path: str = "data.yaml",
    epochs: int = 25,
    img_size: int = 640,
    batch_size: int = 16,
    base_weights: str = "yolov8n.pt"
):
    try:
        from ultralytics import YOLO
        logger.info(f"Loading base model {base_weights} for fine-tuning...")
        model = YOLO(base_weights)
        
        if not os.path.exists(data_yaml_path):
            logger.error(f"Dataset config '{data_yaml_path}' not found.")
            print(f"\n[!] Please place your dataset YAML file at: {data_yaml_path}")
            print("Dataset structure should include train/val images and labels in YOLO format.")
            return False
            
        logger.info(f"Starting training for {epochs} epochs on {data_yaml_path}...")
        results = model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            project="runs/detect",
            name="road_damage_model",
            exist_ok=True
        )
        
        logger.info("Training complete! Best weights saved to runs/detect/road_damage_model/weights/best.pt")
        return True
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return False

if __name__ == "__main__":
    yaml_file = sys.argv[1] if len(sys.argv) > 1 else "data.yaml"
    train_yolo(data_yaml_path=yaml_file)
