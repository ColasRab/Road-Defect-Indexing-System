from ultralytics import YOLOv10
import os
import shutil
import random

dataset_dir = os.path.join('datasets', 'road_defects')
os.makedirs(os.path.join(dataset_dir, 'images', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'images', 'val'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels', 'train'), exist_ok=True)
os.makedirs(os.path.join(dataset_dir, 'labels', 'val'), exist_ok=True)

def split_train_val(train_ratio=0.8):
    """Split the dataset into train and validation sets"""
    train_label_dir = os.path.join(dataset_dir, 'labels', 'train')
    val_label_dir = os.path.join(dataset_dir, 'labels', 'val')
    train_img_dir = os.path.join(dataset_dir, 'images', 'train')
    val_img_dir = os.path.join(dataset_dir, 'images', 'val')

    label_files = [f for f in os.listdir(train_label_dir) if f.endswith('.txt')]

    num_val = int(len(label_files) * (1 - train_ratio))
    val_files = random.sample(label_files, num_val)

    for label_file in val_files:
        src_label = os.path.join(train_label_dir, label_file)
        dst_label = os.path.join(val_label_dir, label_file)
        shutil.move(src_label, dst_label)

        img_name = label_file.replace('.txt', '.jpg')
        src_img = os.path.join(train_img_dir, img_name)
        dst_img = os.path.join(val_img_dir, img_name)
        if os.path.exists(src_img):
            shutil.move(src_img, dst_img)
        else:
            for ext in ['.jpeg', '.png']:
                img_name = label_file.replace('.txt', ext)
                src_img = os.path.join(train_img_dir, img_name)
                dst_img = os.path.join(val_img_dir, img_name)
                if os.path.exists(src_img):
                    shutil.move(src_img, dst_img)
                    break

split_train_val()

# Creating blank YOLOv10 model
model = YOLOv10('yolov10s.pt')

# Training config
training_args = {
    'data': os.path.abspath('road_defects.yaml'),
    'epochs': 100,
    'imgsz': 640,
    'batch': 8,
    'device': 0, 
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
        os.path.join(base_dir, 'labels', 'train'),
        os.path.join(base_dir, 'labels', 'val')
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"Missing required directory: {path}")
            return False
    return True

if not verify_paths():
    raise ValueError("Required directories are missing. Please check your dataset structure.")

# Training the model
try:
    results = model.train(**training_args)
except Exception as e:
    print(f"Error during training: {str(e)}")
    print("Please ensure your dataset is properly organized")
    raise

# Validation
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

# Saving trained model
model.save('road_defect_detection.pt')