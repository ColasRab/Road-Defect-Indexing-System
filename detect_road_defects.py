from ultralytics import YOLOv10
import json
import datetime

# Load the trained model
model = YOLOv10('road_defects_yolov10.pt')

# Run inference on images
results = model.predict(
    source='path/to/test/images',  # directory of test images
    conf=0.25,  # confidence threshold
    iou=0.45,   # NMS IoU threshold
    save=True,  # save results
    save_txt=False,  # save results to *.txt
    save_conf=True,  # save confidences in --save-txt labels
    save_json=True,  # save results to JSON file
    project='runs/detect',  # save results to project/name
    name='exp',  # save results to project/name
    exist_ok=True  # existing project/name ok, do not increment
)

# Process results in COCO format
coco_results = []
for i, r in enumerate(results):
    boxes = r.boxes
    for j, box in enumerate(boxes):
        # Get box coordinates in COCO format [x, y, width, height]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        w = x2 - x1
        h = y2 - y1
        
        # Create COCO format annotation
        coco_ann = {
            'image_id': i,
            'category_id': int(box.cls),
            'bbox': [x1, y1, w, h],
            'score': float(box.conf),
            'area': w * h,
            'iscrowd': 0
        }
        coco_results.append(coco_ann)

# Save results in COCO format
output_file = f'predictions_coco_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
with open(output_file, 'w') as f:
    json.dump(coco_results, f) 