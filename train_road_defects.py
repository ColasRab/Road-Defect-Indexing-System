from ultralytics import YOLOv10
import os

# Ensure the dataset directories exist
dataset_dir = os.path.join('datasets', 'road_defects')
os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'annotations'), exist_ok=True)

# Initialize a new YOLOv10 model from scratch using COCO format
model = YOLOv10('yolov10m.yaml')  # Using medium size model architecture

# Training configuration for COCO format
training_args = {
    'data': os.path.abspath('road_defects.yaml'),  # use absolute path
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'device': 'cpu',
    'workers': 4,
    'patience': 50,
    'save': True,
    'cache': False,
    'amp': False,
    'rect': False,
    'resume': False,
    'optimizer': 'SGD',
    'verbose': True,
    'seed': 0,
    'deterministic': True,
    'project': 'runs/train',
    'name': 'road_defects',
    'exist_ok': True
}

def verify_paths():
    base_dir = os.path.join('datasets', 'road_defects')
    required_paths = [
        os.path.join(base_dir, 'images', 'train'),
        os.path.join(base_dir, 'images', 'val'),
        os.path.join(base_dir, 'annotations')
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Missing required directory: {path}")
            return False
    return True

# Add this before training
if not verify_paths():
    raise ValueError("Required directories are missing. Please check your dataset structure.")

# Train the model
try:
    results = model.train(**training_args)
except Exception as e:
    print(f"Error during training: {str(e)}")
    print("Please ensure your dataset is properly organized and annotations are in COCO format")
    raise

# Validate the model after training
val_args = {
    'data': os.path.abspath('road_defects.yaml'),
    'batch': 8,
    'imgsz': 640,
    'conf': 0.001,
    'iou': 0.6,
    'max_det': 300,
    'half': False,
    'device': 'cpu'
}
results = model.val(**val_args)

# Save the trained model
model.save('road_defects_yolov10.pt') 