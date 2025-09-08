# main.py - Main surveillance system
import cv2
import torch
import numpy as np
import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import threading
import time
from dataclasses import dataclass
from collections import defaultdict
import easyocr
from ultralytics import YOLO
import hashlib

@dataclass
class Detection:
    id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: datetime

@dataclass
class Visit:
    detection: Detection
    thumbnail_path: str
    license_plate: Optional[str] = None
    plate_confidence: Optional[float] = None

class DatabaseManager:
    def __init__(self, db_path: str = "surveillance.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Cameras table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cameras (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                source_path TEXT NOT NULL,
                location TEXT,
                status TEXT DEFAULT 'active',
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Visits table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS visits (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT,
                timestamp TIMESTAMP,
                object_type TEXT,
                confidence_score REAL,
                bbox_x1 INTEGER,
                bbox_y1 INTEGER,
                bbox_x2 INTEGER,
                bbox_y2 INTEGER,
                thumbnail_path TEXT,
                license_plate TEXT,
                plate_confidence REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (camera_id) REFERENCES cameras (id)
            )
        ''')
        
        # Reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_type TEXT,
                parameters TEXT,
                generated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_path TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_camera(self, camera_id: str, name: str, source_path: str, location: str = ""):
        """Add a new camera to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO cameras (id, name, source_path, location)
            VALUES (?, ?, ?, ?)
        ''', (camera_id, name, source_path, location))
        conn.commit()
        conn.close()
    
    def log_visit(self, visit: Visit, camera_id: str):
        """Log a visit to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        det = visit.detection
        cursor.execute('''
            INSERT INTO visits (
                camera_id, timestamp, object_type, confidence_score,
                bbox_x1, bbox_y1, bbox_x2, bbox_y2, thumbnail_path,
                license_plate, plate_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            camera_id, det.timestamp, det.class_name, det.confidence,
            det.bbox[0], det.bbox[1], det.bbox[2], det.bbox[3],
            visit.thumbnail_path, visit.license_plate, visit.plate_confidence
        ))
        
        conn.commit()
        conn.close()
    
    def get_visits(self, camera_id: str = None, start_date: str = None, end_date: str = None, 
                   object_type: str = None, license_plate: str = None):
        """Retrieve visits with filtering options"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        query = "SELECT * FROM visits WHERE 1=1"
        params = []
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)
        if object_type:
            query += " AND object_type = ?"
            params.append(object_type)
        if license_plate:
            query += " AND license_plate LIKE ?"
            params.append(f"%{license_plate}%")
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results

class LicensePlateReader:
    def __init__(self):
        """Initialize EasyOCR for license plate reading"""
        try:
            self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            print(f"EasyOCR initialized with GPU: {torch.cuda.is_available()}")
        except Exception as e:
            print(f"Error initializing EasyOCR: {e}")
            self.reader = None
    
    def read_plate(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[Optional[str], Optional[float]]:
        """Extract license plate text from image region"""
        if self.reader is None:
            return None, None
        
        try:
            x1, y1, x2, y2 = bbox
            # Expand bbox slightly for better OCR
            margin = 10
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(image.shape[1], x2 + margin)
            y2 = min(image.shape[0], y2 + margin)
            
            plate_region = image[y1:y2, x1:x2]
            
            if plate_region.size == 0:
                return None, None
            
            # Enhance image for better OCR
            plate_region = cv2.resize(plate_region, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            plate_region = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
            plate_region = cv2.threshold(plate_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            results = self.reader.readtext(plate_region)
            
            if results:
                # Get the result with highest confidence
                best_result = max(results, key=lambda x: x[2])
                text = best_result[1].strip()
                confidence = best_result[2]
                
                # Basic filtering for license plate patterns
                if len(text) >= 4 and confidence > 0.5:
                    return text.upper(), confidence
            
            return None, None
        
        except Exception as e:
            print(f"Error reading license plate: {e}")
            return None, None

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

class VideoProcessor:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.detector = ObjectDetector()
        self.processing = False
        self.frame_skip = 5  # Process every 5th frame for better performance
    
    def process_video(self, video_path: str, camera_id: str, camera_name: str):
        """Process a video file and detect objects"""
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        # Add camera to database
        self.db_manager.add_camera(camera_id, camera_name, video_path)
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"Processing video: {video_path}")
        print(f"Duration: {duration:.2f}s, FPS: {fps:.2f}, Total frames: {total_frames}")
        
        frame_count = 0
        processed_frames = 0
        
        self.processing = True
        
        try:
            while self.processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    continue
                
                processed_frames += 1
                
                # Resize frame for better performance on lower-end GPUs
                if frame.shape[1] > 1280:  # If width > 1280
                    scale = 1280 / frame.shape[1]
                    new_width = 1280
                    new_height = int(frame.shape[0] * scale)
                    frame = cv2.resize(frame, (new_width, new_height))
                
                # Detect objects
                detections = self.detector.detect_objects(frame)
                
                if detections:
                    print(f"Frame {frame_count}: Found {len(detections)} objects")
                    
                    # Process detections to create visits
                    visits = self.detector.process_detections(frame, detections, camera_id)
                    
                    # Log visits to database
                    for visit in visits:
                        self.db_manager.log_visit(visit, camera_id)
                        
                        # Print detection info
                        plate_info = f", License: {visit.license_plate}" if visit.license_plate else ""
                        print(f"  - {visit.detection.class_name} (conf: {visit.detection.confidence:.2f}){plate_info}")
                
                # Show progress
                if processed_frames % 50 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({processed_frames} frames processed)")
                
                # Display frame with detections (optional, comment out for headless operation)
                display_frame = frame.copy()
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    color = (0, 255, 0) if detection.class_name == 'person' else (255, 0, 0)
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{detection.class_name}: {detection.confidence:.2f}"
                    cv2.putText(display_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                cv2.imshow(f'Detection - {camera_name}', display_frame)
                
                # Press 'q' to quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.processing = False
            
            print(f"\nProcessing completed!")
            print(f"Total frames processed: {processed_frames}")

class ReportGenerator:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def generate_visit_frequency_report(self, days: int = 7):
        """Generate report of visit frequencies"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        # Get visits from last N days
        cursor.execute('''
            SELECT object_type, license_plate, COUNT(*) as visit_count,
                   MIN(timestamp) as first_visit, MAX(timestamp) as last_visit
            FROM visits 
            WHERE timestamp >= datetime('now', '-{} days')
            GROUP BY object_type, COALESCE(license_plate, 'unknown')
            ORDER BY visit_count DESC
        '''.format(days))
        
        results = cursor.fetchall()
        conn.close()
        
        print(f"\n=== Visit Frequency Report (Last {days} days) ===")
        for row in results:
            obj_type, license_plate, count, first_visit, last_visit = row
            plate_str = f" (Plate: {license_plate})" if license_plate else ""
            print(f"{obj_type.title()}{plate_str}: {count} visits")
            print(f"  First visit: {first_visit}")
            print(f"  Last visit: {last_visit}")
            print()
    
    def generate_camera_statistics(self):
        """Generate statistics by camera"""
        conn = sqlite3.connect(self.db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.name, v.object_type, COUNT(*) as count
            FROM cameras c
            LEFT JOIN visits v ON c.id = v.camera_id
            GROUP BY c.name, v.object_type
            ORDER BY c.name, count DESC
        ''')
        
        results = cursor.fetchall()
        conn.close()
        
        print("\n=== Camera Statistics ===")
        current_camera = None
        for row in results:
            camera_name, object_type, count = row
            if camera_name != current_camera:
                print(f"\n{camera_name}:")
                current_camera = camera_name
            if object_type:
                print(f"  {object_type}: {count} detections")

def main():
    """Main function to run the surveillance system"""
    # Initialize database
    db_manager = DatabaseManager()
    
    # Initialize video processor
    processor = VideoProcessor(db_manager)
    
    # Initialize report generator
    reporter = ReportGenerator(db_manager)
    
    # Test videos (you'll need to download these)
    test_videos = [
        {
            "path": "test_videos/traffic.mp4",
            "camera_id": "cam_01",
            "name": "Traffic Camera 1"
        },
        {
            "path": "test_videos/parking.mp4", 
            "camera_id": "cam_02",
            "name": "Parking Lot Camera"
        },
        {
            "path": "test_videos/entrance.mp4",
            "camera_id": "cam_03", 
            "name": "Entrance Camera"
        }
    ]
    
    print("=== Surveillance Detection System ===")
    print("Make sure you have downloaded test videos to test_videos/ folder")
    print("Recommended test videos:")
    print("1. Traffic scenes with cars and license plates")
    print("2. Parking lot footage")
    print("3. Entrance/pedestrian areas")
    print("\nPress Enter to continue or Ctrl+C to exit...")
    
    try:
        input()
        
        # Create test_videos directory
        os.makedirs("test_videos", exist_ok=True)
        os.makedirs("thumbnails", exist_ok=True)
        
        # Process each test video
        for video_config in test_videos:
            if os.path.exists(video_config["path"]):
                print(f"\nProcessing {video_config['name']}...")
                processor.process_video(
                    video_config["path"],
                    video_config["camera_id"], 
                    video_config["name"]
                )
                print(f"Completed processing {video_config['name']}")
            else:
                print(f"Video not found: {video_config['path']}")
        
        # Generate reports
        print("\nGenerating reports...")
        reporter.generate_visit_frequency_report(days=1)
        reporter.generate_camera_statistics()
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
    