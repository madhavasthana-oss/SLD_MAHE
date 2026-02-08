from ultralytics import YOLO    

class handDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Load the YOLO model
        self.model = YOLO(model_path)

    def detect_hands(self, frame):
        results = self.model(frame)
        return results