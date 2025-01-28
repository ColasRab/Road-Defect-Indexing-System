from yolov10 import YOLOv10
import cv2
import torch
import time

class RoadDefectDetector:
    def __init__(self, model_path=None, weights_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = YOLOv10(
            weights_path if weights_path else 'yolov10-l.pt',
            device=self.device
        )
        
        # Define NRDD2024 classes
        self.classes = [
            'longitudinal_crack',  # LC
            'transverse_crack',    # TC
            'alligator_crack',     # AC
            'pothole',            # PH
            'patch',              # PA
            'white_line',         # WL
            'sealed_crack'        # SC
        ]
        
        # Define colors for each class (BGR format)
        self.colors = {
            'longitudinal_crack': (0, 255, 0),    # Green
            'transverse_crack': (255, 0, 0),      # Blue
            'alligator_crack': (0, 0, 255),       # Red
            'pothole': (255, 255, 0),             # Cyan
            'patch': (255, 0, 255),               # Magenta
            'white_line': (0, 255, 255),          # Yellow
            'sealed_crack': (128, 128, 128)       # Gray
        }
        
    def process_video(self, source=0, conf=0.25):
        cap = cv2.VideoCapture(source)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties - FPS: {fps}, Resolution: {width}x{height}")
        
        while cap.isOpened():
            start_time = time.time()
            success, frame = cap.read()
            if not success:
                break
            
            # Preprocess frame
            frame = cv2.resize(frame, (640, 640))
            
            # Run YOLOv10 inference
            results = self.model.predict(
                frame,
                conf_thres=conf,
                iou_thres=0.45,
                classes=None,
                agnostic_nms=False,
                max_det=1000
            )
            
            # Draw results
            annotated_frame = self.model.draw_results(frame, results)
            
            # Calculate and display FPS
            fps = 1.0 / (time.time() - start_time)
            cv2.putText(annotated_frame, f'FPS: {int(fps)}', 
                       (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display
            cv2.imshow('NRDD2024 Road Defect Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        cap.release()
        cv2.destroyAllWindows()
    
    def train_model(self, data_yaml, epochs=100):
        """
        Train the model on NRDD2024 dataset
        """
        self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=640,
            batch=16,            # Reduce if running out of memory
            workers=8,           # Adjust based on your CPU
            device=self.device,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.001,          # Adjust learning rate if needed
            cos_lr=True,        # Cosine learning rate scheduling
            label_smoothing=0.1,
            mixup=0.2,          # Data augmentation
            copy_paste=0.3,     # Data augmentation
            augment=True        # Enable augmentation
        )

if __name__ == "__main__":
    # Initialize detector
    detector = RoadDefectDetector()
    
    # Start training
    detector.train_model(
        data_yaml='data.yaml',
        epochs=100  # Adjust based on your needs
    )
    
    # For webcam inference
    detector.process_video(source=0)
    
    # For video file inference
    # detector.process_video(source='path/to/video.mp4') 