
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List
from VideoProcessors.LicensePlateReader import LicensePlateReader
from DatabaseManagers.DataClasses import Detection, Visit
from datetime import datetime
from pathlib import Path


def remove_duplicates(detections, iou_threshold=0.5):
    filtered = []
    detections = sorted(detections, key=lambda d: d.confidence, reverse=True)

    while detections:
        best = detections.pop(0)
        filtered.append(best)
        detections = [
            d for d in detections 
            if compute_iou(best.bbox, d.bbox) < iou_threshold
        ]

    return filtered

def compute_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1p, y1p, x2p, y2p = box2

    xi1, yi1 = max(x1, x1p), max(y1, y1p)
    xi2, yi2 = min(x2, x2p), min(y2, y2p)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2p - x1p) * (y2p - y1p)

    union = box1_area + box2_area - inter_area
    return inter_area / union if union > 0 else 0

class ObjectDetector:
    def __init__(self, model_size: str = "yolov8n.pt"):
        """Initialize YOLO model for object detection"""
        # Use smaller model for GTX 1650 Super
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            self.device = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(0).lower()
            if "1650" in gpu_name or "1660" in gpu_name:
                model_size = "yolov8n.pt"  # Nano model for lower-end GPUs
                print(f"Using nano model for {gpu_name}")
            elif any(x in gpu_name for x in ["3060", "3070", "4060", "4070"]):
                model_size = "yolov8s.pt"  # Small model for mid-range GPUs
                print(f"Using small model for {gpu_name}")
            else:
                model_size = "yolov8m.pt"  # Medium model for high-end GPUs
            
            self.device = "cuda"
        
        try:
            self.model = YOLO(model_size)
            self.model.to(self.device)
            print(f"YOLO model {model_size} loaded on {self.device}")
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            # Fallback to CPU
            self.device = "cpu"
            self.model = YOLO("yolov8n.pt")
            self.model.to(self.device)
        
        # Classes we're interested in
        self.target_classes = {
            0: 'person',
            1: 'bicycle', 
            2: 'car',
            3: 'motorcycle',
            5: 'bus',
            7: 'truck'
        }
        
        self.license_plate_reader = LicensePlateReader()
    
    def detect_objects(self, frame: np.ndarray, confidence_threshold: float = 0.5) -> List[Detection]:
        """Detect objects in frame and return Detection objects"""
        detections = []
        
        try:
            # Run inference
            results = self.model(frame, conf=confidence_threshold, device=self.device)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        class_id = int(box.cls.item())
                        confidence = box.conf.item()
                        
                        # Only process classes we're interested in
                        if class_id in self.target_classes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            class_name = self.target_classes[class_id]
                            
                            detection = Detection(
                                id=i,
                                class_name=class_name,
                                confidence=confidence,
                                bbox=(x1, y1, x2, y2),
                                timestamp=datetime.now()
                            )
                            
                            detections.append(detection)
            detections = remove_duplicates(detections, iou_threshold=0.5)
        except Exception as e:
            print(f"Error during detection: {e}")
        
        return detections
    
    def process_detections(self, frame: np.ndarray, detections: List[Detection], 
                          camera_id: str) -> List[Visit]:
        """Process detections to create visit records with thumbnails and license plates"""
        visits = []
        
        for detection in detections:
            # Create thumbnail
            x1, y1, x2, y2 = detection.bbox
            thumbnail = frame[y1:y2, x1:x2]
            
            # Save thumbnail
            thumbnail_dir = Path(f"thumbnails/{camera_id}/{detection.timestamp.strftime('%Y%m%d')}")
            thumbnail_dir.mkdir(parents=True, exist_ok=True)
            
            thumbnail_filename = f"{detection.timestamp.strftime('%H%M%S')}_{detection.class_name}_{detection.id}.jpg"
            thumbnail_path = thumbnail_dir / thumbnail_filename
            
            cv2.imwrite(str(thumbnail_path), thumbnail)
            
            # Try to read license plate for vehicles
            license_plate = None
            plate_confidence = None
            
            if detection.class_name in ['car', 'truck', 'bus', 'motorcycle']:
                license_plate, plate_confidence = self.license_plate_reader.read_plate(frame, detection.bbox)
            
            visit = Visit(
                detection=detection,
                thumbnail_path=str(thumbnail_path),
                license_plate=license_plate,
                plate_confidence=plate_confidence
            )
            
            visits.append(visit)
        
        return visits
    