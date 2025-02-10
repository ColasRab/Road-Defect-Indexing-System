import torch
import cv2
import numpy as np
from torchvision import transforms
from road_extractor import RoadExtractor

"""
Custom Model Wrapper for YOLOv10 with Road Extraction
"""

class Road_Yolov10_Model(torch.nn.Module):
    def __init__(self, model_path, scale_factor=0.5, num_segments=100, k_clusters=3):
        super(Road_Yolov10_Model, self).__init__()
        self.model = model_path
        self.road_extractor = RoadExtractor("placeholder.jpg", scale_factor=scale_factor)
        self.num_segments = num_segments
        self.k_clusters = k_clusters
        self.to_tensor = transforms.ToTensor()

    def forward(self, x):
        x_np = x[0].permute(1, 2, 0).detach().cpu().numpy()
        x_np = (x_np * 255).astype(np.uint8)
        x_np = cv2.cvtColor(x_np, cv2.COLOR_RGB2BGR)

        self.road_extractor.image = x_np

        road_region, road_mask, _ = self.road_extractor.extract_road_region(
            self.road_extractor.image, 
            num_segments = self.num_segments, 
            k_clusters = self.k_clusters
        )

        road_region_rgb = cv2.cvtColor(road_region, cv2.COLOR_BGR2RGB)
        road_tensor = self.to_tensor(road_region_rgb).unsqueeze(0)

        detections = self.model(road_tensor)
        return detections
    
def load_model(model_path):
    model = torch.load(model_path)
    model.eval()
    return model