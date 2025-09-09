import sqlite3
from DatabaseManagers.DataClasses import Visit


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
        
        # if camera_id:
        #     query += " AND camera_id = ?"
        #     params.append(camera_id)
        # if start_date:
        #     query += " AND timestamp >= ?"
        #     params.append(start_date)
        # if end_date:
        #     query += " AND timestamp <= ?"
        #     params.append(end_date)
        # if object_type:
        #     query += " AND object_type = ?"
        #     params.append(object_type)
        # if license_plate:
        #     query += " AND license_plate LIKE ?"
        #     params.append(f"%{license_plate}%")
        
        query += " ORDER BY timestamp DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        conn.close()
        
        return results
