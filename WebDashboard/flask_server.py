# app.py - FastAPI Web Dashboard Backend
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import sqlite3
import os
import cv2
import asyncio
import threading
from datetime import datetime
from pathlib import Path
import io
from collections import defaultdict
from DatabaseManagers.DatabaseManager import DatabaseManager
from VideoProcessors.ObjectDetector import ObjectDetector

app = FastAPI(title="Surveillance Dashboard API", version="1.0.0")

# CORS middleware for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/thumbnails", StaticFiles(directory="thumbnails"), name="thumbnails")

# Global variables
db_manager = DatabaseManager()
active_streams = {}
stream_threads = {}

# Pydantic models for API
class CameraConfig(BaseModel):
    id: str
    name: str
    rtsp_url: str
    location: str = ""
    active: bool = True

class DetectionFilter(BaseModel):
    camera_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    object_type: Optional[str] = None
    license_plate: Optional[str] = None
    limit: Optional[int] = 100

class ReportRequest(BaseModel):
    report_type: str  # "frequency", "license_plate", "camera_stats"
    parameters: Dict[str, Any]

# RTSP Stream Processor Class
class RTSPProcessor:
    def __init__(self, camera_config: CameraConfig, db_manager: DatabaseManager):
        self.camera_config = camera_config
        self.db_manager = db_manager
        self.detector = ObjectDetector()
        self.running = False
        self.cap = None
        self.frame_count = 0
        self.last_detection_time = {}
        self.detection_cooldown = 5  # seconds between detections for same object
        
    async def start_stream(self):
        """Start processing RTSP stream"""
        self.running = True
        
        # Add camera to database
        self.db_manager.add_camera(
            self.camera_config.id,
            self.camera_config.name,
            self.camera_config.rtsp_url,
            self.camera_config.location
        )
        
        # Start processing in background thread
        thread = threading.Thread(target=self._process_stream, daemon=True)
        thread.start()
        
        return {"status": "started", "camera_id": self.camera_config.id}
    
    def _process_stream(self):
        """Main stream processing loop"""
        try:
            self.cap = cv2.VideoCapture(self.camera_config.rtsp_url)
            
            # Configure capture properties for RTSP
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for real-time
            self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
            
            if not self.cap.isOpened():
                print(f"Failed to connect to RTSP stream: {self.camera_config.rtsp_url}")
                return
            
            print(f"Started processing camera: {self.camera_config.name}")
            
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    print(f"Lost connection to {self.camera_config.name}, attempting reconnect...")
                    self._reconnect()
                    continue
                
                self.frame_count += 1
                
                # Process every 10th frame for real-time performance
                if self.frame_count % 10 == 0:
                    self._process_frame(frame)
                
                # Store latest frame for live view
                active_streams[self.camera_config.id] = frame
                
        except Exception as e:
            print(f"Error processing stream {self.camera_config.name}: {e}")
        finally:
            if self.cap:
                self.cap.release()
    
    def _reconnect(self):
        """Attempt to reconnect to RTSP stream"""
        if self.cap:
            self.cap.release()
        
        # Wait before reconnecting
        threading.Event().wait(5)
        
        if self.running:
            self.cap = cv2.VideoCapture(self.camera_config.rtsp_url)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    def _process_frame(self, frame):
        """Process individual frame for detections"""
        try:
            # Resize for better performance
            if frame.shape[1] > 1280:
                scale = 1280 / frame.shape[1]
                new_width = 1280
                new_height = int(frame.shape[0] * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Detect objects
            detections = self.detector.detect_objects(frame, confidence_threshold=0.6)
            
            if detections:
                # Process detections
                visits = self.detector.process_detections(
                    frame, detections, self.camera_config.id
                )
                
                # Apply detection cooldown to prevent spam
                current_time = datetime.now()
                filtered_visits = []
                
                for visit in visits:
                    detection_key = f"{visit.detection.class_name}_{visit.detection.bbox}"
                    
                    if detection_key not in self.last_detection_time:
                        self.last_detection_time[detection_key] = current_time
                        filtered_visits.append(visit)
                    else:
                        time_diff = (current_time - self.last_detection_time[detection_key]).seconds
                        if time_diff >= self.detection_cooldown:
                            self.last_detection_time[detection_key] = current_time
                            filtered_visits.append(visit)
                
                # Log filtered visits to database
                for visit in filtered_visits:
                    self.db_manager.log_visit(visit, self.camera_config.id)
                    print(f"Logged {visit.detection.class_name} detection on {self.camera_config.name}")
        
        except Exception as e:
            print(f"Error processing frame: {e}")
    
    def stop_stream(self):
        """Stop processing stream"""
        self.running = False
        if self.cap:
            self.cap.release()
        
        if self.camera_config.id in active_streams:
            del active_streams[self.camera_config.id]

# API Endpoints

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve main dashboard page"""
    return FileResponse("static/index.html")

@app.post("/api/cameras/add")
async def add_camera(camera: CameraConfig, background_tasks: BackgroundTasks):
    """Add new camera and start processing"""
    try:
        # Create RTSP processor
        processor = RTSPProcessor(camera, db_manager)
        stream_threads[camera.id] = processor
        
        # Start processing in background
        background_tasks.add_task(processor.start_stream)
        
        return {"status": "success", "message": f"Camera {camera.name} added and started"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cameras")
async def get_cameras():
    """Get all cameras"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT id, name, source_path, location, status FROM cameras")
        cameras = []
        
        for row in cursor.fetchall():
            cameras.append({
                "id": row[0],
                "name": row[1],
                "rtsp_url": row[2],
                "location": row[3],
                "status": row[4],
                "is_active": row[0] in active_streams
            })
        
        conn.close()
        return cameras
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/cameras/{camera_id}")
async def remove_camera(camera_id: str):
    """Remove camera and stop processing"""
    try:
        # Stop processing if active
        if camera_id in stream_threads:
            stream_threads[camera_id].stop_stream()
            del stream_threads[camera_id]
        
        # Remove from database
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM cameras WHERE id = ?", (camera_id,))
        conn.commit()
        conn.close()
        
        return {"status": "success", "message": "Camera removed"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stream/{camera_id}")
async def get_stream(camera_id: str):
    """Get live stream frame"""
    if camera_id not in active_streams:
        raise HTTPException(status_code=404, detail="Camera stream not found")
    
    try:
        frame = active_streams[camera_id]
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        
        return StreamingResponse(
            io.BytesIO(frame_bytes),
            media_type="image/jpeg"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/detections")
async def get_detections(filter_params: DetectionFilter):
    """Get filtered detection history"""
    try:
        visits = db_manager.get_visits(
            camera_id=filter_params.camera_id,
            start_date=filter_params.start_date,
            end_date=filter_params.end_date,
            object_type=filter_params.object_type,
            license_plate=filter_params.license_plate
        )
        
        # Format results
        results = []
        for visit in visits[:filter_params.limit]:
            results.append({
                "id": visit[0],
                "camera_id": visit[1],
                "timestamp": visit[2],
                "object_type": visit[3],
                "confidence": visit[4],
                "bbox": [visit[5], visit[6], visit[7], visit[8]],
                "thumbnail_path": visit[9],
                "license_plate": visit[10],
                "plate_confidence": visit[11]
            })
        return {
            "detections": results,
            "total": len(results)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/generate")
async def generate_report(report_request: ReportRequest):
    """Generate various reports"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        if report_request.report_type == "frequency":
            days = report_request.parameters.get("days", 7)
            cursor.execute('''
                SELECT object_type, license_plate, COUNT(*) as visit_count,
                       MIN(timestamp) as first_visit, MAX(timestamp) as last_visit
                FROM visits 
                WHERE timestamp >= datetime('now', '-{} days')
                GROUP BY object_type, COALESCE(license_plate, 'unknown')
                ORDER BY visit_count DESC
            '''.format(days))
            
            results = cursor.fetchall()
            report_data = []
            
            for row in results:
                report_data.append({
                    "object_type": row[0],
                    "license_plate": row[1],
                    "visit_count": row[2],
                    "first_visit": row[3],
                    "last_visit": row[4]
                })
        elif report_request.report_type == "license_plate":
            plate = report_request.parameters.get("plate", "")
            cursor.execute('''
                SELECT v.*, c.name as camera_name
                FROM visits v
                JOIN cameras c ON v.camera_id = c.id
                WHERE v.license_plate LIKE ?
                ORDER BY v.timestamp DESC
                LIMIT 50
            ''', (f"%{plate}%",))
            
            results = cursor.fetchall()
            report_data = []
            
            for row in results:
                report_data.append({
                    "timestamp": row[2],
                    "camera_name": row[-1],
                    "license_plate": row[10],
                    "confidence": row[4]
                })
        
        elif report_request.report_type == "camera_stats":
            cursor.execute('''
                SELECT c.name, c.id, v.object_type, COUNT(*) as count,
                       MAX(v.timestamp) as last_detection
                FROM cameras c
                LEFT JOIN visits v ON c.id = v.camera_id
                WHERE v.timestamp >= datetime('now', '-7 days')
                GROUP BY c.name, c.id, v.object_type
                ORDER BY c.name, count DESC
            ''')
            
            results = cursor.fetchall()
            camera_stats = defaultdict(list)
            
            for row in results:
                camera_name, camera_id, object_type, count, last_detection = row
                camera_stats[camera_name].append({
                    "object_type": object_type,
                    "count": count,
                    "last_detection": last_detection
                })
            
            report_data = dict(camera_stats)
        
        else:
            raise HTTPException(status_code=400, detail="Invalid report type")
        
        conn.close()
        
        return {
            "report_type": report_request.report_type,
            "generated_at": datetime.now().isoformat(),
            "data": report_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stats/dashboard")
async def get_dashboard_stats():
    """Get dashboard statistics"""
    try:
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        # Total detections today
        cursor.execute('''
            SELECT COUNT(*) FROM visits 
            WHERE date(timestamp) = date('now')
        ''')
        today_detections = cursor.fetchone()[0]
        
        # Total detections this week
        cursor.execute('''
            SELECT COUNT(*) FROM visits 
            WHERE timestamp >= datetime('now', '-7 days')
        ''')
        week_detections = cursor.fetchone()[0]
        
        # Active cameras
        active_cameras = len(active_streams)
        
        # Recent detections by type
        cursor.execute('''
            SELECT object_type, COUNT(*) as count
            FROM visits 
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY object_type
            ORDER BY count DESC
        ''')
        detection_types = dict(cursor.fetchall())
        
        # Hourly activity for today
        cursor.execute('''
            SELECT strftime('%H', timestamp) as hour, COUNT(*) as count
            FROM visits 
            WHERE date(timestamp) = date('now')
            GROUP BY hour
            ORDER BY hour
        ''')
        hourly_activity = dict(cursor.fetchall())
        
        conn.close()
        return {
            "today_detections": today_detections,
            "week_detections": week_detections,
            "active_cameras": active_cameras,
            "detection_types": detection_types,
            "hourly_activity": hourly_activity
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/thumbnail/{path:path}")
async def get_thumbnail(path: str):
    """Serve thumbnail images"""
    thumbnail_path = Path() / path
    
    if not thumbnail_path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found")
    
    return FileResponse(thumbnail_path)

# WebSocket for real-time updates
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Send periodic updates
            stats = await get_dashboard_stats()
            await websocket.send_json({
                "type": "stats_update",
                "data": stats
            })
            await asyncio.sleep(5)  # Update every 5 seconds
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def main():
    import uvicorn
    
    # Create required directories
    os.makedirs("static", exist_ok=True)
    os.makedirs("thumbnails", exist_ok=True)
    
    print("Starting Surveillance Dashboard...")
    print("Dashboard will be available at: http://localhost:8000")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    