import cv2
import os
from road_extractor import RoadExtractor

def preprocess_images(image_dir):
    """Preprocess images using RoadExtractor"""
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        road_extractor = RoadExtractor(img_path, scale_factor=0.5)
        road_region, road_mask, segments = road_extractor.extract_road_region(road_extractor.image)
        cv2.imwrite(img_path, road_region)

dataset_dir = os.path.join('datasets', 'road_defects')

# Preprocess train and validation images
preprocess_images(os.path.join(dataset_dir, 'images', 'train'))
preprocess_images(os.path.join(dataset_dir, 'images', 'val'))