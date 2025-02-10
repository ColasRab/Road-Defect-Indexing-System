import torch
import cv2
import numpy as np
from torchvision import transforms
from road_yolov10 import Road_Yolov10_Model, load_model

def main():
    yolo_model = load_model('road_defects_yolov10.pt')
    road_yolov10 = Road_Yolov10_Model(yolo_model, scale_factor=0.5, num_segments=100, k_clusters=3)
    road_yolov10.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    to_tensor = transforms.ToTensor()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(frame_rgb).unsqueeze(0)

        with torch.no_grad():
            detections = road_yolov10(input_tensor)

        print("Detections: ", detections)
        cv2.imshow("Road Defects Detection", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()