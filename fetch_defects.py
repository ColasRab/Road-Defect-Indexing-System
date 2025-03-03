import torch
import cv2
import numpy as np
from torchvision import transforms
from road_yolov10 import Road_Yolov10_Model, load_model

def main():
    # load the Road_YOLOv10 model for road defects detection.
    yolo_model = load_model('road_yolov10.pt')
    road_yolov10 = Road_Yolov10_Model(yolo_model, scale_factor=0.5, num_segments=100, k_clusters=3)
    road_yolov10.eval()

    # init the video capture device.
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    to_tensor = transforms.ToTensor()

    # Process the video stream frame by frame.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        input_tensor = to_tensor(frame_rgb).unsqueeze(0)

        # Perform road defects detection.
        with torch.no_grad():
            detections = road_yolov10(input_tensor)

        print("Detections: ", detections)
        cv2.imshow("Road Defects Detection", frame_resized)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture device and close the window.
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()