import torch
import cv2
import numpy as np
from torchvision import transforms
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# Define the custom YOLOv10DetectionModel class
class YOLOv10DetectionModel(DetectionModel):
    def __init__(self):
        super().__init__()

# Add the custom model class to PyTorch's safe globals
torch.serialization.add_safe_globals(['ultralytics.nn.tasks.YOLOv10DetectionModel'])

class RoadExtractor:
    def __init__(self, image_path, scale_factor=0.5):
        self.image_path = image_path
        self.scale_factor = scale_factor
        self.image = None

    def extract_road_region(self, image, num_segments=100, k_clusters=3):
        # Placeholder for road extraction logic
        mask = np.ones(image.shape[:2], dtype=np.uint8)
        return image, mask, None

class Road_Yolov10_Model(torch.nn.Module):
    def __init__(self, model_path, scale_factor=0.5, num_segments=100, k_clusters=3):
        super(Road_Yolov10_Model, self).__init__()
        
        try:
            # Load the checkpoint with weights_only=False since we trust the source
            checkpoint = torch.load(model_path, weights_only=False)
            
            # Extract the model state from the checkpoint
            if isinstance(checkpoint.get('model', None), dict):
                # If model is a state dict
                self.model = YOLOv10DetectionModel()
                self.model.load_state_dict(checkpoint['model'])
            else:
                # If model is the full model
                self.model = checkpoint['model']
            
            print(f"Successfully loaded YOLOv10 model from {model_path}")
        except Exception as e:
            raise Exception(f"Failed to load YOLOv10 model: {str(e)}")
        
        # Initialize road extractor
        self.road_extractor = RoadExtractor("placeholder.jpg", scale_factor=scale_factor)
        self.num_segments = num_segments
        self.k_clusters = k_clusters
        self.to_tensor = transforms.ToTensor()

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            # Convert tensor to numpy image
            x_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
            x_np = (x_np * 255).astype(np.uint8)
        else:
            # Handle case where input is already a numpy array
            x_np = x

        # Ensure BGR format for OpenCV
        if x_np.shape[-1] == 3:  
            x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

        # Extract road region
        self.road_extractor.image = x_np
        road_region, road_mask, _ = self.road_extractor.extract_road_region(
            self.road_extractor.image, 
            num_segments=self.num_segments, 
            k_clusters=self.k_clusters
        )

        # Convert road region back to tensor format
        road_region_rgb = cv2.cvtColor(road_region, cv2.COLOR_BGR2RGB)
        input_tensor = self.to_tensor(road_region_rgb).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            results = self.model(input_tensor)
        return results

def initialize_model(model_path):
    """
    Initialize the Road_Yolov10_Model with error handling
    """
    try:
        road_yolo_model = Road_Yolov10_Model(model_path)
        print("✅ Road_Yolov10_Model initialized successfully!")
        return road_yolo_model
    except Exception as e:
        print(f"❌ Model initialization failed: {str(e)}")
        return None

# Usage example
if __name__ == "__main__":
    model_path = "C:/Users/bentl/Desktop/Road-Defect-Indexing-System/runs/train/road_defects/weights/best.pt"
    model = initialize_model(model_path)
    model.eval()
      
