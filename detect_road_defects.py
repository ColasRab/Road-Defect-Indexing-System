from ultralytics import YOLOv10
import cv2
import numpy as np
import os
from datetime import datetime

# Load the trained model
model = YOLOv10('road_defects_yolov10.pt')

def save_mask(mask, original_image, save_path, filename):
    """Save the mask as a binary image"""
    # Resize mask to match original image size
    mask = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    
    # Create binary mask (255 for defect, 0 for background)
    binary_mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)
    
    # Save the binary mask
    mask_filename = f"{filename}_mask.png"
    cv2.imwrite(os.path.join(save_path, mask_filename), binary_mask)
    return mask_filename

# Create output directory for masks
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('runs', 'detect', f'masks_{timestamp}')
os.makedirs(output_dir, exist_ok=True)

# Run inference on images with mask output
results = model.predict(
    source='path/to/test/images',  # directory of test images
    conf=0.25,  # confidence threshold
    iou=0.45,   # NMS IoU threshold
    save=False,  # don't save regular predictions
    save_txt=False,
    save_conf=False,
    project=output_dir,
    name='',
    exist_ok=True,
    retina_masks=True,  # use high-quality segmentation masks
    boxes=False,  # don't show boxes
    show_labels=False,  # don't show labels
    show=False  # don't show preview
)

# Process results and save masks
mask_info = []
for i, result in enumerate(results):
    # Get original image name
    orig_img_path = result.path
    filename = os.path.splitext(os.path.basename(orig_img_path))[0]
    
    # Get original image for size reference
    original_image = cv2.imread(orig_img_path)
    
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy()
        
        # Combine all masks for this image
        combined_mask = np.zeros_like(masks[0])
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask)
        
        # Save the combined mask
        mask_file = save_mask(combined_mask, original_image, output_dir, filename)
        
        # Store information about the mask
        mask_info.append({
            'original_image': orig_img_path,
            'mask_file': mask_file,
            'defects_found': len(masks),
            'classes': result.boxes.cls.cpu().tolist() if result.boxes else []
        })

# Save detection information
info_file = os.path.join(output_dir, 'detection_info.txt')
with open(info_file, 'w') as f:
    for info in mask_info:
        f.write(f"Original: {info['original_image']}\n")
        f.write(f"Mask: {info['mask_file']}\n")
        f.write(f"Number of defects: {info['defects_found']}\n")
        f.write(f"Defect classes: {info['classes']}\n")
        f.write("-" * 50 + "\n")

print(f"Masks saved to: {output_dir}") 