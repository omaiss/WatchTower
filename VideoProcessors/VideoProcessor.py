from DatabaseManagers.DatabaseManager import DatabaseManager
from VideoProcessors.ObjectDetector import ObjectDetector
import cv2
import os

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
