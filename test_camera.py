import cv2
import torch
import numpy as np
from ultralytics import YOLO

class Road_Yolov10_Model:
    def __init__(self, model_path):
        try:
            self.model = YOLO(model_path)  # Load the YOLO model
            self.model.model.eval()  # Set model to evaluation mode
            print(f"✅ Successfully loaded YOLOv10 model from {model_path}")
        except Exception as e:
            raise Exception(f"❌ Failed to load YOLOv10 model: {str(e)}")

    def forward(self, frame, conf_threshold=0.01):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        results = self.model(frame_rgb)
        
        for result in results:
            result.boxes = [box for box in result.boxes if box.conf[0] >= conf_threshold]
        
        return results


def initialize_model(model_path):
    return Road_Yolov10_Model(model_path)  # Ensure we return a valid model


def draw_detections(frame, results):
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = float(box.conf[0])  # Confidence score
            cls = int(box.cls[0])  # Class ID

            label = f"Class {cls} ({conf:.2f})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


if __name__ == "__main__":
    model_path = "C:/Users/bentl/Desktop/Road-Defect-Indexing-System/runs/train/road_defects/weights/best.pt"
    model = initialize_model(model_path)
    
    model_test = YOLO("C:/Users/bentl/Desktop/Road-Defect-Indexing-System/runs/train/road_defects/weights/best.pt")
    model_test.info()
        
    # Commented out webcam testing
    cap = cv2.VideoCapture(1)  # Change to 1 if using an external camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()

    while True:
        print(model.model.names)  # Check class labels
        ret, frame = cap.read()
        frame = cv2.resize(frame, (640, 640))  # Resize for better detection
        if not ret:
            print("Error: Failed to capture image.")
            break

        results = model.forward(frame)
        print(results)  # Debugging: print results
        draw_detections(frame, results)

        cv2.imshow('YOLOv10 Webcam', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Test with a static image
    # test_img_path = "C:/Users/bentl/Desktop/Road-Defect-Indexing-System/datasets/road_defects/images/val/India_000438_jpg.rf.cf7def1a1909ecbf75dd6ccc7704bce7.jpg"
    # test_img = cv2.imread(test_img_path)
    # if test_img is not None:
    #     test_img = cv2.resize(test_img, (640, 640))  # Resize for consistency
    #     test_results = model.forward(test_img)
    #     print(test_results)  # Debugging: print results
    #     print(test_results[0].boxes)  
    #     test_results[0].show()
    #     draw_detections(test_img, test_results)
    #     cv2.imshow("Test Image", test_img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    # else:
    #     print(f"Error: Could not read test image from {test_img_path}")
